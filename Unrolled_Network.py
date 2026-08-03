import torch
import torch.nn as nn
from DC import DC_CG, DC_UMPIRE
from unet.unet import create_model as Unet
from unet.unet_mag import create_model as Unet_m
from Utils import *

def inverse_softplus(y, beta=1.0):
    y = torch.as_tensor(y)
    return torch.log(torch.expm1(beta * y)) / beta
    
class UnrolledNet_UM(nn.Module):
    def __init__(self, initial_params, Unrolls , DF_Steps , epsilon, Share_params, Momentum_type = "None" , SoftPlus = False, Smoothing_type = "Simple", delta = 1.0, alpha = 20): 
        super().__init__()
        beta_softplus        = 20
        self.SoftPlus_func   = torch.nn.Softplus(beta = beta_softplus)
        self.Network_p       = Unet  (use_norm=True,   use_time_emb = True,  channel_mult=[1,2,3], num_channels=32)
        self.Network_a       = Unet_m(use_norm=True,   use_time_emb = True,  channel_mult=[1,2,3], num_channels=32)
        self.DataConsistency = DC_UMPIRE(initial_params["eta"], initial_params["beta_1"],initial_params["beta_2"], DF_Steps, Unrolls, epsilon, Share_params, SoftPlus, beta_softplus, Momentum_type , initial_params["beta_Momentum"], Smoothing_type, delta, alpha)
        
        self.SoftPlus       = SoftPlus
        self.Share_params   = Share_params 
        self.Unrolls        = Unrolls
        self.epsilon        = epsilon
        self.Relu           = torch.nn.ReLU(inplace=True)

        lambda_init  = torch.tensor(initial_params["lambda_ADMM"], dtype=torch.float32)
        lambda_init  = inverse_softplus(lambda_init, beta=beta_softplus) if SoftPlus else lambda_init
        shape        = () if Share_params else (Unrolls,)
        self.lambda1 = nn.Parameter(lambda_init.expand(shape).clone())
        self.lambda2 = nn.Parameter(lambda_init.expand(shape).clone())


    def forward(self, zf, Coil, Mask):
        zf_cmplx =  zf[:,0:1,:,:] + 1j*zf[:,1:2,:,:]
        x        =  torch.clone(zf_cmplx)
        u        =  torch.zeros_like(torch.abs(zf_cmplx))       # lambda1
        u_prime  =  torch.zeros_like(zf_cmplx)                  # lambda2
        
        for OuterIter in range(self.Unrolls):
            
            lambda1_val = self.lambda1 if self.Share_params else self.lambda1[OuterIter]
            lambda2_val = self.lambda2 if self.Share_params else self.lambda2[OuterIter]
            if self.SoftPlus:
                lambda1_val = self.SoftPlus_func(lambda1_val)
                lambda2_val = self.SoftPlus_func(lambda2_val)

            Mag_input   = torch.abs(x) + u 
            Phs_input   = C2R(Smooth_sgn(x, self.epsilon) + u_prime)
            
            t = x.new_tensor([OuterIter+1] * zf.shape[0]).long().to(x.device)
            m = self.Relu(self.Network_a(Mag_input, t))
            p = Smooth_sgn(R2C(self.Network_p(Phs_input, t)), self.epsilon)
            x = self.DataConsistency(zf, Coil, Mask, OuterIter, m-u, p-u_prime)
            
            u       = u + lambda1_val * (torch.abs(x) - m)
            u_prime = u_prime + lambda2_val * (Smooth_sgn(x, self.epsilon)- p)
        return x


class UnrolledNet_PDDL(nn.Module):
    def __init__(self, initial_params, Unrolls , DF_Steps , Share_params, SoftPlus = False):
        super().__init__()
        beta_softplus        = 20
        self.Network         = Unet(use_norm=True, use_time_emb = True,  channel_mult=[1,2,5], num_channels=32)
        self.DataConsistency = DC_CG(initial_params["mu"], DF_Steps, Unrolls, Share_params, SoftPlus, beta_softplus = beta_softplus)
        self.SoftPlus_func   = torch.nn.Softplus(beta = beta_softplus)
        
        self.SoftPlus       = SoftPlus
        self.Share_params   = Share_params 
        self.Unrolls        = Unrolls
        
        lambda_init  = torch.tensor(initial_params["lambda_ADMM"], dtype=torch.float32)
        lambda_init  = inverse_softplus(lambda_init, beta=beta_softplus) if SoftPlus else lambda_init
        self.lambda1 = nn.Parameter(lambda_init.expand(() if Share_params else (Unrolls,)).clone())
    
    # Without CheckPointing
    def forward(self, zf, Coil, Mask):
        x = torch.clone(zf)
        v = torch.clone(zf)
        u = torch.zeros_like(zf)
        
        for OuterIter in range(self.Unrolls):
            
            lambda1_val = self.lambda1 if self.Share_params else self.lambda1[OuterIter]
            if self.SoftPlus:
                lambda1_val = self.SoftPlus_func(lambda1_val)

            t = x.new_tensor([OuterIter+1] * x.shape[0]).long().to(x.device)
            v = self.Network(x+u, t)
            x = self.DataConsistency(zf, Coil, Mask, OuterIter,v,u)
            u = u + lambda1_val * (x - v)
            
        recon = x[:,0:1,:,:] + 1j*x[:,1:2,:,:]
        return recon