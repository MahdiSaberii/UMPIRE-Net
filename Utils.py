import numpy as np
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import matplotlib.patheffects as path_effects
from einops import rearrange

def Smooth_sgn(x, epsilon=1e-6):
    return x/torch.sqrt((x.real**2 + x.imag**2) + epsilon)
def Smooth_abs(x, epsilon=1e-6):
    return torch.sqrt((x.real**2 + x.imag**2) + epsilon)
def R2C(x):
    return x[:,0:1,:,:] + 1j*x[:,1:2,:,:]
def C2R(x):
    return torch.concat([torch.real(x), torch.imag(x)], axis=1)

def GetGaussianMask(SamplingMask,rho=0.4, num_iter=0):
    [nx, ny] = SamplingMask.shape
    count = 0
    test_pts = np.ceil(np.sum(SamplingMask[:]) * rho)
    Mask_Validation = np.zeros_like(SamplingMask)
    temp_mask = np.copy(SamplingMask)
    mx = SamplingMask.shape[0]//2
    my = SamplingMask.shape[1]//2
    #if num_iter == 0:
        #print('center of kspace, mx: ', mx, ', my: ', my)
    temp_mask[mx - 2: mx + 2, my - 2: my + 2] = 0
    while count <= test_pts:
        indx = int(np.round(np.random.normal(loc=mx, scale=(nx - 1) / 2)))
        indy = int(np.round(np.random.normal(loc=my, scale=(ny - 1) / 2)))
        if (0 <= indx < nx and 0 <= indy < ny and temp_mask[indx, indy] == 1 and Mask_Validation[indx, indy] != 1):
            Mask_Validation[indx, indy] = 1
            count = count + 1
    Mask_Training = SamplingMask - Mask_Validation
    
    return np.complex64(Mask_Training), np.complex64(Mask_Validation), np.complex64(Mask_Training)+np.complex64(Mask_Validation) 

def Mask_Generator(nx, ny, R, ACS=24):

    mask_1d = np.zeros(ny, dtype=np.uint8)

    center = ny // 2
    mask_1d[::R] = 1
    mask_1d[center - ACS//2 : center + ACS//2] = 1
    mask_2d = np.tile(mask_1d[None, :], (nx, 1))
    return mask_2d


def FFT_np(image, axis=[-2,-1]):
    return np.fft.ifftshift(np.fft.fftn(np.fft.ifftshift(image, axes=axis), axes=axis, norm='ortho'), axes=axis)
def IFFT_np(kspace, axis=[-2,-1]):
    return np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(kspace, axes=axis), axes=axis, norm='ortho'), axes=axis)
def FFT(image, axis=[2,3]):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(image,dim=axis), dim=axis, norm = 'ortho'), dim=axis)

def IFFT(kspace, axis=[2,3]):
    return torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(kspace,dim = axis), dim=axis, norm = 'ortho'),dim = axis)

def EHE(x, Coil, Mask):
    EHEx = torch.sum(IFFT(Mask*(FFT(x*Coil))) * torch.conj(Coil),axis=1,keepdim=True)
    return EHEx

def getSSIM(space_ref, space_rec):
    data_range = np.max(np.abs(space_ref)) - np.min(np.abs(space_ref))
    return ssim(space_ref, space_rec, data_range=data_range)

def getPSNR(ref, recon):
    """
    Measures PSNR between the reference and the reconstructed images
    """
    mse = np.sum(np.square(np.abs(ref - recon))) / ref.size
    psnr = 20 * np.log10(np.abs(ref.max()) / (np.sqrt(mse) + 1e-10))
    return psnr

def normalize_angle(x):
    return (x + np.pi) / (2 * np.pi)

fn_np = lambda x: x.detach().cpu().numpy() if not isinstance(x, np.ndarray) else x
    ##### Functions Start
def find_center_ind(kspace, axes=(1, 2, 3)):
    center_locs = torch.norm(kspace, dim=axes).squeeze()
    return torch.argsort(center_locs)[-1:]

def index_flatten2nd(ind, shape):
    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))
    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]

class ssdu_masks():
    def __init__(self, rho=0.2, small_acs_block=(6, 6)):
        self.rho = rho
        self.small_acs_block = small_acs_block

    def uniform_selection(self, input_data, input_mask,device,slice_i=0,rho=0.2,acs_block=None):
        input_mask = rearrange(input_mask,'1 1 h w->h w')
        nrow, ncol = input_mask.shape[0], input_mask.shape[1]

        center_kx = int(find_center_ind(input_data, axes=(1, 2)))
        center_ky = int(find_center_ind(input_data, axes=(0, 2)))

        temp_mask = np.copy(fn_np(input_mask))
        if acs_block is None:
            acs_block = self.small_acs_block
        else:
            acs_block = acs_block
        temp_mask[center_kx - self.small_acs_block[0] // 2: center_kx + self.small_acs_block[0] // 2,
        center_ky - self.small_acs_block[1] // 2: center_ky + self.small_acs_block[1] // 2] = 0

        pr  = np.ndarray.flatten(temp_mask)
        rng = np.random.default_rng(slice_i)
        ind = rng.choice(np.arange(nrow * ncol),size=int(np.count_nonzero(pr.real) * rho), replace=False, p=pr.real / np.sum(pr.real))

        [ind_x, ind_y] = index_flatten2nd(ind, (nrow, ncol))

        loss_mask = np.zeros_like(fn_np(input_mask))
        loss_mask[ind_x, ind_y] = 1

        trn_mask  = fn_np(input_mask) - loss_mask
        trn_mask  = rearrange(trn_mask  ,'h w->1 1 h w 1')
        loss_mask = rearrange(loss_mask ,'h w->1 1 h w 1')
        trn_mask  = torch.from_numpy(trn_mask).to(device)
        loss_mask = torch.from_numpy(loss_mask).to(device)
        return trn_mask, loss_mask    

def L1_L2_Loss(recon, label):
    loss = ( torch.norm(recon-label , p=2) / torch.norm( label, p=2) ) + ( torch.norm(recon-label , p=1) / torch.norm( label , p=1))
    return loss


def E(x, Coil, Mask):
    Ex = FFT(x*Coil) * Mask
    return Ex

def EH(y, Coil, Mask):
    EHy = torch.sum(IFFT(y*Mask) * torch.conj(Coil),axis=1,keepdim=True)
    return EHy

def MsEHEx(x, Coil, Mask):
    EHEx = torch.sum(IFFT(Mask*(FFT(x*Coil))) * torch.conj(Coil),axis=1,keepdim=True)
    return EHEx