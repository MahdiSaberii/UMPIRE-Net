from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from DataLoader import DataLoaderSL as DL
from torchvision.utils import save_image
from Unrolled_Network import UnrolledNet_PDDL, UnrolledNet_UM
from Utils import *
import random
import yaml
from torch.utils.data import DataLoader, random_split
import pandas as pd

def mean_value(x):
    return x.detach().float().mean().item()

softplus = torch.nn.Softplus(beta=20)

def maybe_softplus(x):
    return softplus(x) if SoftPlus else x

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    with open("Config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device            = torch.device(config["Test"]["device"] if torch.cuda.is_available() else "cpu")
    model_epoch       = config["Test"]["model_epoch"]
    dataset_name      = config["Test"]["Dataset"]
    Acc_R             = config["Test"]["R"]
    Unrolls           = config["Unrolls"]
    DF_Steps          = config["DF_Steps"]
    epsilon           = float(config["epsilon"])
    delta             = float(config["delta"])
    alpha             = float(config["alpha"])
    PF_Factor         = float(config["Test"]["PF_Factor"])
    threshold         = float(config["Test"]["threshold"])
    
    Smoothing_type    = config["Test"]["Smoothing_type"]
    Momentum_type     = config["Test"]["Momentum_type"]
    Share_params      = config["Test"]["Share_params"]
    GD_or_CG          = "CG" # It is going to be always CG
    Network           = config["Test"]["Network"]
    SoftPlus          = config["SoftPlus"]

    diff_scale        = config["diff_scale"]
    window_scale      = config["window_scale"]
    
    initial_params = {
    "mu": config["mu"]["value"],
    "eta": config["eta"]["value"],
    "beta_1": config["beta_1"]["value"],
    "beta_2": config["beta_2"]["value"],
    "lambda_ADMM": config["lambda_ADMM"]["value"],
    "beta_Momentum": config["beta_Momentum"]["value"]}

    share_flag        = 1 if Share_params else 0 
    CG_flag           = 1 if GD_or_CG == "CG" else 0
    SoftPlus_flag     = "SoftPlus" if SoftPlus else "None"
    
    test_add          = f"./Test_Results"
    if Network in {"UMPIRE"}:
        EXP_name  = f"{Network}_{dataset_name}_R{Acc_R}_UNet_Share{share_flag}_{SoftPlus_flag}_{Momentum_type}_{PF_Factor}_epsilon{epsilon}_Smoothing_{Smoothing_type}"
    if Network == "PDDL":
        EXP_name  = f"{Network}_LARGE_{dataset_name}_R{Acc_R}_UNet_Share{share_flag}_{GD_or_CG}_{SoftPlus_flag}_{PF_Factor}"
    
    save_dir = f"./{test_add}/{EXP_name}"
    os.makedirs(f"{test_add}/{EXP_name}", exist_ok=True)
    os.makedirs(f"{test_add}/{EXP_name}/pngs", exist_ok=True)

        
    # if dataset_name == "AxT2":
    #     # train_rawdata_path = "/tank/scratch/Mahdi/FastMRI/mats/Brain_FastMRI_AXT2_300Slices_Train_slices_SmoothCoils/"
    #     train_rawdata_path = "/home/naxos2-raid25/saber032/Main_works/saber032-data/FastMRI/Brain_AXT2_300Slices_Test_slices_SmoothCoils/"
    if dataset_name == "AxFLAIR":
        train_rawdata_path = "/home/naxos2-raid25/saber032/Main_works/Dataset/FLAIR_Brain/Test/"
    if dataset_name == "CorPD":
        # train_rawdata_path = "/home/naxos2-raid25/saber032/Main_works/Dataset/PD_300/Cropped/Train_Merged/"
        train_rawdata_path = "/home/naxos2-raid25/saber032/Main_works/Dataset/PD_300/Cropped/Test_Merged/"
    if dataset_name == "CorPDFS":
        train_rawdata_path = "/home/naxos2-raid25/saber032/Main_works/Dataset/PDFS_300/Cropped/Test_Merged/"
    
    CartesianData = DL(train_rawdata_path)
    
    train_loader = DataLoader(
                            dataset=CartesianData,
                            batch_size=1,
                            shuffle=True,
                            num_workers=4)    
    
    if Network in {"UMPIRE"}:
        model   = UnrolledNet_UM(initial_params, Unrolls, DF_Steps, epsilon, Share_params, Momentum_type, SoftPlus, Smoothing_type, delta, alpha).to(device)
    if Network == "PDDL":
        model   = UnrolledNet_PDDL(initial_params, Unrolls , DF_Steps , Share_params, SoftPlus).to(device)
    model.load_state_dict(torch.load(f"./Train_Results/{EXP_name}/model/BestModel_Val_R{Acc_R}_Epoch{model_epoch}.pth")['model_state_dict'])
    model.eval()

    # Partial Fourier Mask Generator
    if dataset_name in {"AxFLAIR"}:
        nx, ny    = 320,320 
        num_right = int(PF_Factor * 272)
    elif dataset_name in {"CorPD", "CorPDFS"}:
        nx, ny = 320,332
        num_right = int(PF_Factor * ny)
    
    Omega_Mask     = torch.tensor(Mask_Generator(nx, ny, Acc_R, ACS=24)).unsqueeze(0).unsqueeze(0)

    
    if dataset_name == "CorPD":
        Exclude_list = [0,1,2, 38,39,40,41,42,43,44,45,46,47,48,49, 60,61,62,63,64,65,66,  76,78,79,80, 118,119,120, 140,141,142,143,144,145,146,147,148,149, 154,155,156, 170,171,172,173,174,175,176,177,178,179, 192,193,194, 232,233,234, 240,241,242,243,244,245,249 ,272,273,274, 312,313,314, 352,353,354] 
    elif dataset_name == "CorPDFS":
        Exclude_list = [0,1,2, 35,36,37, 73,74,75, 110,111,112, 147,148,149, 183,184,185, 225,226,227, 267,268,269, 310,311,312, 353,354,355]
    elif dataset_name == "AxT2":
        Exclude_list = [1,2,3,4,5,6,7,38,39,40,41,42,43,44,45,78,79,80,81,82,118,119,120,121,122,154,155,156,157,158,193,194,195,196,233,234,235,272,273,274,275,312,313,314,315,316,317,318,319,320,352,353,354,355,356]
    elif dataset_name == "AxFLAIR":
        Exclude_list = []
    PSNR, SSIM = [], []
    metrics = []
    
    print(f"EXP. Name: {EXP_name}    ,, GPU: {device}")
    progress_bar = tqdm(train_loader, total=len(train_loader), desc=f"Testing Epoch {model_epoch}")
    for idx , (ksp, coil, SliceNumber, FileName) in enumerate(progress_bar):
        
        if SliceNumber.item() not in Exclude_list:
            ksp         = ksp/torch.max(torch.abs(ksp)) 
            coil        = coil.to(torch.complex64).to(device)
            ksp         = ksp.to(torch.complex64).to(device)
            Omega_Mask  = Omega_Mask.float().to(device)
            
            coil_rss                     = torch.sqrt(torch.sum(torch.abs(coil)**2, axis=1))
            Zeros_Mask                   = torch.zeros_like(Omega_Mask).to(device)
            Zeros_Mask[:,:,:,:num_right] = 1
            Ones_Mask                    = torch.ones_like(Omega_Mask) - Zeros_Mask # These two mask are for the partial Fourier sampling
            ksp_pf                       = ksp * Zeros_Mask
            label                        = IFFT(ksp_pf)
            label                        = torch.sum(label * torch.conj(coil), axis=1, keepdims=True)
            
            Omega_Mask  = Omega_Mask  * Zeros_Mask
            if dataset_name == "AxFLAIR":
                # MATLAB 1:48 -> Python :48
                # MATLAB 272:320 -> Python 271:
                    Omega_Mask[..., :48]  = 1
                    Omega_Mask[..., 271:] = 1
            
            zero_filled = IFFT(ksp_pf * Omega_Mask)
            zero_filled = torch.sum(zero_filled * torch.conj(coil), axis=1, keepdims=True)
            zero_filled = torch.cat([torch.real(zero_filled), torch.imag(zero_filled)], axis=1)

            recon     = model(zero_filled, coil, Omega_Mask) * coil_rss
            
            beta1_value   = 0.0
            beta2_value   = 0.0
            mu_value      = 0.0
            eta_value_one = 0.0

            if Network == "UMPIRE":
                beta1_value   = mean_value(maybe_softplus(model.DataConsistency.beta1))
                beta2_value   = mean_value(maybe_softplus(model.DataConsistency.beta2))
                lambda1_value = mean_value(maybe_softplus(model.lambda1))
                lambda2_value = mean_value(maybe_softplus(model.lambda2))
                eta_value_one = mean_value(maybe_softplus(model.DataConsistency.eta))

            elif Network == "PDDL":
                mu_value      = mean_value(maybe_softplus(model.DataConsistency.mu))
                eta_value_one = 0.0

            progress_bar.set_postfix({
                "Beta1"       : f"{float(beta1_value):.2f}",
                "Beta2"       : f"{float(beta2_value):.2f}",
                "Mu"          : f"{float(mu_value):.2f}",
                "eta"         : f"{float(eta_value_one):.2f}"
                })

            zf_cmplx    = zero_filled[:,0:1,:,:] + 1j*zero_filled[:,1:2,:,:]
            label[torch.abs(label)<threshold] = 0
            recon[torch.abs(label)<threshold] = 0
            zf_cmplx[torch.abs(label)<threshold] = 0
            
            x           = zf_cmplx.squeeze().cpu().detach().numpy()
            y           = label.squeeze().cpu().detach().numpy()
            z           = recon.squeeze().cpu().detach().numpy()


            ssim_slice = getSSIM(np.abs(y), np.abs(z))
            psnr_slice = getPSNR(np.abs(y), np.abs(z))
            PSNR.append(psnr_slice)
            SSIM.append(ssim_slice)
            
            z_copy      = np.flipud((z)).copy()
            label_copy  = np.flipud((y)).copy()
            zf_copy     = np.flipud((x)).copy()

            # print(torch.max(torch.abs(torch.tensor(z_copy))), torch.max(torch.abs(torch.tensor(label_copy))))
            xx = z_copy / torch.max(torch.abs(torch.tensor(z_copy)))
            abs_xx = torch.abs(xx).cpu().squeeze().detach()
            abs_xx = torch.clamp(abs_xx, max=window_scale) / window_scale
            save_path = f"{test_add}/{EXP_name}/pngs/Slice_{SliceNumber.item()}_Recon.png"
            save_image(abs_xx, save_path)

            xx = label_copy / torch.max(torch.abs(torch.tensor(label_copy)))
            abs_xx = torch.abs(xx).cpu().squeeze().detach()
            abs_xx = torch.clamp(abs_xx, max=window_scale) / window_scale
            save_path = f"{test_add}/{EXP_name}/pngs/Slice_{SliceNumber.item()}_Label.png"
            save_image(abs_xx, save_path)
            
            xx = diff_scale*(label_copy-z_copy) / torch.max(torch.abs(torch.tensor(label_copy)))
            abs_xx = torch.abs(xx).cpu().squeeze().detach()
            abs_xx = torch.clamp(abs_xx, max=window_scale) / window_scale
            save_path = f"{test_add}/{EXP_name}/pngs/Slice_{SliceNumber.item()}_Error.png"
            save_image(abs_xx, save_path)
            
            metrics.append({
                "Slice": int(SliceNumber.cpu().item()),
                "PSNR": float(psnr_slice),
                "SSIM": float(ssim_slice),
            })

            sio.savemat(
                f"{test_add}/{EXP_name}/Metrics_Test_Epoch{model_epoch}_threshold{threshold}.mat",
                {
                    "Slice":   np.array([m["Slice"]   for m in metrics]),
                    "PSNR":    np.array([m["PSNR"]    for m in metrics]),
                    "SSIM":    np.array([m["SSIM"]    for m in metrics]),
                }
            )
            
            # --- Compute mean and std ---
            valid_metrics = [m for m in metrics if m["PSNR"] >= 20]
            num_total = len(metrics)
            num_valid = len(valid_metrics)
            num_excluded = num_total - num_valid
            psnr_vals = np.array([m["PSNR"] for m in valid_metrics])
            ssim_vals = np.array([m["SSIM"] for m in valid_metrics])


            psnr_mean, psnr_std = psnr_vals.mean(), psnr_vals.std()
            ssim_mean, ssim_std = ssim_vals.mean(), ssim_vals.std()

            # --- Save mean/std to a text file ---
            with open(f"{test_add}/{EXP_name}/Metrics_Test_Epoch{model_epoch}_threshold{threshold}.txt", "w") as f:
                f.write("=== Metrics Summary ===\n")
                f.write(f"Total slices     : {num_total}\n")
                f.write(f"Used (PSNR>=20)  : {num_valid}\n")
                f.write(f"Excluded         : {num_excluded}\n\n")
                f.write(f"PSNR: mean = {psnr_mean:.4f}, std = {psnr_std:.4f}\n")
                f.write(f"SSIM: mean = {ssim_mean:.4f}, std = {ssim_std:.4f}\n")
            df = pd.DataFrame(metrics)
            df.to_csv(f"{test_add}/{EXP_name}/Metrics_Test_Epoch{model_epoch}_threshold{threshold}.csv", index=False)
    
    # Sort by slice number
    df = pd.DataFrame(metrics).sort_values("Slice")

    plt.figure(figsize=(10,4))
    plt.plot(df["Slice"], df["PSNR"], 'o-')
    plt.xlabel("Slice")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/PSNR_vs_Slice.png", dpi=300)

    plt.figure(figsize=(10,4))
    plt.plot(df["Slice"], df["SSIM"], 'o-')
    plt.xlabel("Slice")
    plt.ylabel("SSIM")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/SSIM_vs_Slice.png", dpi=300)