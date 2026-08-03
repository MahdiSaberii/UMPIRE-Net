# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 19:39:55 2021

@author: Mahdi Saberi
"""

import torch
import torch.nn as nn
from Utils import *

def inverse_softplus(y, beta=1.0):
    y = torch.as_tensor(y)
    return torch.log(torch.expm1(beta * y)) / beta

class DC_UMPIRE(nn.Module):

    def __init__(self, eta, beta_1, beta_2, Unrolls_GD, num_of_unrolls, epsilon=1e-6, Share_params = False, SoftPlus = True, beta_softplus=20, Momentum_type = "Nesterov", beta_momentum=0.9, Smoothing_type="Simple", delta = 1.0, alpha = 20):
        super().__init__()
        shape    = () if Share_params else (num_of_unrolls,)
        shape_DF = (Unrolls_GD, num_of_unrolls)

        beta1_init = torch.tensor(beta_1, dtype=torch.float32)
        beta2_init = torch.tensor(beta_2, dtype=torch.float32)
        eta_init   = torch.tensor(eta, dtype=torch.float32)
        mom_init   = torch.tensor(beta_momentum, dtype=torch.float32)

        if SoftPlus:
            beta1_init = inverse_softplus(beta1_init, beta=20)
            beta2_init = inverse_softplus(beta2_init, beta=20)
            eta_init   = inverse_softplus(eta_init, beta=20)
            mom_init   = inverse_softplus(mom_init, beta=20)

        self.beta1         = nn.Parameter(beta1_init.expand(shape).clone())
        self.beta2         = nn.Parameter(beta2_init.expand(shape).clone())
        self.eta           = nn.Parameter(eta_init.expand(shape_DF).clone())
        self.beta_momentum = nn.Parameter(mom_init.expand(shape_DF).clone())
        
        self.Momentum_type = Momentum_type
        self.Smoothing_type= Smoothing_type
        self.Share_params  = Share_params
        self.Unrolls_GD    = Unrolls_GD
        self.epsilon       = epsilon
        self.delta         = delta
        self.alpha         = alpha
        self.SoftPlus      = SoftPlus
        self.SoftPlus_func  = torch.nn.Softplus(beta = beta_softplus)

    def Smooth_abs(self, x, epsilon):
        mag = torch.abs(x)
        if self.Smoothing_type == "Simple":
            return torch.sqrt((x.real**2 + x.imag**2) + epsilon)

        elif self.Smoothing_type == "SmoothL1":
            # print(torch.mean(mag))
            temp = torch.where(mag <= self.delta, 0.5 * mag**2 / self.delta, mag - 0.5 * self.delta)
            return temp + 1e-3

        elif self.Smoothing_type == "Pseudo-Huber":
            # print(torch.mean((self.delta**2)*(torch.sqrt(1.0 + (mag / self.delta) ** 2) - 1.0)))
            return ((self.delta**2)*(torch.sqrt(1.0 + (mag / self.delta) ** 2) - 1.0) + 1e-7)

        elif self.Smoothing_type == "LogExp":
            return (1.0 / self.alpha)*(torch.log1p(torch.exp(self.alpha * mag)) + torch.log1p(torch.exp(-self.alpha * mag)))

        else:
            raise ValueError(f"Unknown Smoothing_type: {self.Smoothing_type}")
    
    def forward(self,zf, Coil, Mask, unroll_idx, m,p_cmplx): # P is already complex
        zf_cmplx = zf[:,0:1,:,:] + zf[:,1:2,:,:]*1j  
        x_cmplx  = zf_cmplx
        velocity = torch.zeros_like(x_cmplx)
        scale    = torch.abs(zf_cmplx)
        
        idx = () if self.Share_params else unroll_idx

        beta_1_eval = self.beta1[idx]
        beta_2_eval = self.beta2[idx]
        if self.SoftPlus:
            beta_1_eval = self.SoftPlus_func(beta_1_eval)
            beta_2_eval = self.SoftPlus_func(beta_2_eval)
        
        for ix in range(self.Unrolls_GD):
            beta_momentum_eval = self.beta_momentum[ix, unroll_idx]
            eta_eval           = self.eta[ix, unroll_idx]
            if self.SoftPlus:
                beta_momentum_eval, eta_eval = (self.SoftPlus_func(beta_momentum_eval), self.SoftPlus_func(eta_eval))
            
            if self.Momentum_type == "Nesterov":
                x_eval = x_cmplx - beta_momentum_eval * velocity
            else:
                x_eval = x_cmplx
            
            # print(torch.max(torch.abs((0.0001*x_eval * x_eval))))
            term1       = MsEHEx(x_eval,Coil,Mask) - zf_cmplx
            term2       = -m*(x_eval/self.Smooth_abs(x_eval, self.epsilon)) + x_eval
            term31      = (torch.conj(p_cmplx)*x_eval*x_eval)/(self.Smooth_abs(x_eval, self.epsilon)**3)
            term32      = -p_cmplx/self.Smooth_abs(x_eval, self.epsilon)
            term3       = (0.5)*(term31 + term32)*(scale**2)
            Grad        =  term1 + (beta_1_eval)*term2 + (beta_2_eval)*term3
            # print(beta_1_eval.item(), beta_2_eval.item())
            if self.Momentum_type == "Nesterov":
                velocity = beta_momentum_eval * velocity +  Grad
                x_cmplx  = x_cmplx - eta_eval * velocity
            elif self.Momentum_type == "Polyak":
                velocity = beta_momentum_eval * velocity + (1-beta_momentum_eval) * Grad
                x_cmplx  = x_cmplx - eta_eval * velocity
            elif self.Momentum_type == "None":
                x_cmplx     = x_cmplx - eta_eval * Grad
        return x_cmplx


class DC_CG(nn.Module):
    def __init__(self, mu, DF_Steps,num_of_unrolls, Share_params, SoftPlus = True, beta_softplus = 20):
        super().__init__()
        
        mu_init = torch.tensor(mu, dtype=torch.float32)
        mu_init = inverse_softplus(mu_init, beta=beta_softplus) if SoftPlus else mu_init
        self.mu = nn.Parameter(mu_init.clone() if Share_params else mu_init.repeat(num_of_unrolls), requires_grad=True)

        self.SoftPlus_func = torch.nn.Softplus(beta = beta_softplus)
        self.SoftPlus      = SoftPlus
        self.iterations    = DF_Steps
        self.Share_params  = Share_params
        
    def forward(self,zf, Coil, Mask, unroll_idx, v,u):
        zf_cmplx  = zf[:,0:1,:,:] + zf[:,1:2,:,:]*1j  
        v_cmplx   = v[:,0:1,:,:]  + v[:,1:2,:,:]*1j
        u_cmplx   = u[:,0:1,:,:]  + u[:,1:2,:,:]*1j
        v_minus_u = v_cmplx - u_cmplx
        
        mu_eval  = self.SoftPlus_func(self.mu) if self.SoftPlus else self.mu        
        p_now    = zf_cmplx + (mu_eval if self.Share_params else mu_eval[unroll_idx]) * v_minus_u
        r_now    = torch.clone(p_now)
        b_approx = torch.zeros_like(p_now)
                
        for idx in range(self.iterations):
            q = EHE(p_now, Coil, Mask) + (mu_eval if self.Share_params else mu_eval[unroll_idx]) * p_now
            rrOverpq = torch.sum(r_now*torch.conj(r_now)) / torch.sum(q*torch.conj(p_now))  # rrOverpq = (r'*r)/(p'*q);
            b_next   = b_approx + rrOverpq*p_now
            r_next   = r_now - rrOverpq*q
            p_next   = r_next + torch.sum(r_next*torch.conj(r_next)) / torch.sum(r_now*torch.conj(r_now)) * p_now # p = r_next + ( (r_next'*r_next)/(r'*r) )*p;
            b_approx = b_next
            p_now    = torch.clone(p_next)
            r_now    = torch.clone(r_next)
        return torch.cat([torch.real(b_approx), torch.imag(b_approx)], dim=1)