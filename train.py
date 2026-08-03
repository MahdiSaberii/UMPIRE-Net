from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from src.DataLoader import DataLoaderSL as DL
from torchvision.utils import save_image
from src.Unrolled_Network import UnrolledNet_PDDL, UnrolledNet_UM
from src.Utils import *
import random
import yaml
from torch.utils.data import DataLoader, random_split
import imageio.v2 as imageio


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

    with open("./config/Config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device            = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    dataset_name      = config["Dataset"]
    Acc_R             = config["R"]
    n_epochs          = config["n_epochs"]
    learning_rate     = config["learning_rate"]
    Unrolls           = config["Unrolls"]
    DF_Steps          = config["DF_Steps"]
    epsilon           = float(config["epsilon"])
    delta             = float(config["delta"])
    alpha             = float(config["alpha"])
    scheduler_patince = config["scheduler_patince"]
    scheduler_factor  = config["scheduler_factor"]
    Val_Split         = config["Val_Split"]
    SSDU_ACS_Block    = (config["SSDU_ACS_Block"], config["SSDU_ACS_Block"])
    SSDU_rho          = config["SSDU_rho"]
    PF_Factor         = float(config["PF_Factor"])
    k_MM              = config["k_MM"]
    
    Smoothing_type    = config["Smoothing_type"]
    Momentum_type     = config["Momentum_type"]
    Share_params      = config["Share_params"]
    GD_or_CG          = config["GD_or_CG"]
    Network           = config["Network"]
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
    
    test_add          = f"./Train_Results"
    if Network in {"UMPIRE"}:
        EXP_name  = f"{Network}_{dataset_name}_R{Acc_R}_UNet_Share{share_flag}_{SoftPlus_flag}_{Momentum_type}_{PF_Factor}_epsilon{epsilon}_Smoothing_{Smoothing_type}"
    if Network == "PDDL":
        EXP_name  = f"{Network}_LARGE_{dataset_name}_R{Acc_R}_UNet_Share{share_flag}_{GD_or_CG}_{SoftPlus_flag}_{PF_Factor}"
    
    save_dir = f"./{test_add}/{EXP_name}"
    os.makedirs(f"{test_add}/{EXP_name}", exist_ok=True)
    os.makedirs(f"{test_add}/{EXP_name}/pngs", exist_ok=True)
    os.makedirs(f"{test_add}/{EXP_name}/model", exist_ok=True)
    os.makedirs(f"{test_add}/{EXP_name}/files", exist_ok=True)
    os.makedirs(f"{test_add}/{EXP_name}/trn_loss", exist_ok=True)
        
    if dataset_name == "CorPD":
        train_rawdata_path = "../Dataset/PD_300/Cropped/Train/"
    if dataset_name == "CorPDFS":
        train_rawdata_path = "../Dataset/PDFS_300/Cropped/Train/"
    
    CartesianData = DL(train_rawdata_path)
    
    # 10% validation
    num_total = len(CartesianData)
    num_val   = int(Val_Split * num_total)
    num_train = num_total - num_val

    # fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(CartesianData, [num_train, num_val], generator=generator)
    train_loader = DataLoader(
                            dataset=train_dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=4)
    val_loader   = DataLoader(
                            dataset=val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=4)
    
    
    if Network in {"UMPIRE"}:
        model   = UnrolledNet_UM(initial_params, Unrolls, DF_Steps, epsilon, Share_params, Momentum_type, SoftPlus, Smoothing_type, delta, alpha).to(device)
    if Network == "PDDL":
        model   = UnrolledNet_PDDL(initial_params, Unrolls , DF_Steps , Share_params, SoftPlus).to(device)
    
    # network.load_state_dict(torch.load("./Train_Results/MI_R6_TrueNormalization/model/BestModel_R6_Epoch100.pth")['model_state_dict'])
    model.train()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of trainable parameters: {trainable_params:,}')
    with open(f"{test_add}/{EXP_name}/files/Config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)
    with open(f"{test_add}/{EXP_name}/files/trainable_parameters.txt", 'w') as f:
        f.write(f'Total number of trainable parameters: {trainable_params:,}')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = scheduler_steps, gamma = gamma_scheduler)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patince, threshold=1e-4, min_lr=1e-7)

    '''
    ========== Summary ==========    
    '''

    loss_list, beta1_list, beta2_list  = [] , [] , []
    mu_list = []
    lambda1_list , lambda2_list = [] , []
    hist            = {}
    eta_history     = []
    best_val_loss   = torch.inf

    # Partial Fourier Mask Generator
    if dataset_name in {"CorPD", "CorPDFS"}:
        nx, ny = 320,332
        num_right = int(PF_Factor * ny)
    
    Omega_Mask     = torch.tensor(Mask_Generator(nx, ny, Acc_R, ACS=24)).unsqueeze(0).unsqueeze(0)
    mask_gen_1     = ssdu_masks(rho=SSDU_rho, small_acs_block = SSDU_ACS_Block)

    for epoch in range(n_epochs+1):
        print(f"EXP. Name: {EXP_name}    ,, GPU: {device}")
        loss_all , Average_loss  = 0.0 , 0.0
        progress_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}")
        for idx , (ksp, coil, SliceNumber, FileName) in enumerate(progress_bar):
            
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
            
            loss_total = 0.0
            for m in range(k_MM):
                rng_idx = 1_000_000 + SliceNumber.item() * k_MM + m
                trn_mask, loss_mask = mask_gen_1.uniform_selection(Omega_Mask, Omega_Mask, device, slice_i=rng_idx)
                trn_mask    = trn_mask.squeeze(-1)
                loss_mask   = loss_mask.squeeze(-1)
                trn_mask    = trn_mask  * Zeros_Mask
                loss_mask   = loss_mask * Zeros_Mask
                if dataset_name == "AxFLAIR":
                # MATLAB 1:48 -> Python :48
                # MATLAB 272:320 -> Python 271:
                    trn_mask[..., :48]  = 1
                    trn_mask[..., 271:] = 1

                    loss_mask[..., :48]  = 0
                    loss_mask[..., 271:] = 0
                
                trn_mask_png = trn_mask.detach().cpu().numpy().squeeze()
                loss_mask_png = loss_mask.detach().cpu().numpy().squeeze()
                imageio.imwrite(f"{test_add}/{EXP_name}/trn_loss/trn_mask.png", (trn_mask_png.astype(np.uint8) * 255))
                imageio.imwrite(f"{test_add}/{EXP_name}/trn_loss/loss_mask.png", (loss_mask_png.astype(np.uint8) * 255))
                zero_filled = IFFT(ksp_pf * trn_mask)
                zero_filled = torch.sum(zero_filled * torch.conj(coil), axis=1, keepdims=True)
                zero_filled = torch.cat([torch.real(zero_filled), torch.imag(zero_filled)], axis=1)

                recon     = model(zero_filled, coil, trn_mask)
                recon_ksp = FFT(recon * coil) * loss_mask
                label_ksp = ksp_pf            * loss_mask

                loss_total = loss_total + L1_L2_Loss(recon_ksp, label_ksp)
            loss = loss_total / k_MM
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_all         += loss.item()
            Average_loss_show = loss_all/(idx+1)
            
            
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
                "Average_Loss": f"{Average_loss_show:.2f}",
                "Val_Loss"    : f"{best_val_loss:.2f}",
                "Beta1"       : f"{float(beta1_value):.2f}",
                "Beta2"       : f"{float(beta2_value):.2f}",
                "Mu"          : f"{float(mu_value):.2f}",
                "eta"         : f"{float(eta_value_one):.2f}",
                "LR"          : f"{optimizer.param_groups[0]['lr']:.2e}"
                })
        
        train_loss = loss_all / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ksp, coil, SliceNumber, FileName in val_loader:

                ksp        = ksp / torch.max(torch.abs(ksp))
                coil       = coil.to(torch.complex64).to(device)
                ksp        = ksp.to(torch.complex64).to(device)
                Omega_Mask = Omega_Mask.float().to(device)
                coil_rss   = torch.sqrt(torch.sum(torch.abs(coil)**2, axis=1))
                
                Zeros_Mask = torch.zeros_like(Omega_Mask).to(device)
                Zeros_Mask[:, :, :, :num_right] = 1
                ksp_pf = ksp * Zeros_Mask
                loss_total = 0.0

                for m in range(k_MM):

                    rng_idx = 2_000_000 + SliceNumber.item() * k_MM + m

                    trn_mask, loss_mask = mask_gen_1.uniform_selection(Omega_Mask, Omega_Mask, device, slice_i=rng_idx)

                    trn_mask  = trn_mask.squeeze(-1) * Zeros_Mask
                    loss_mask = loss_mask.squeeze(-1) * Zeros_Mask

                    if dataset_name == "AxFLAIR":
                    # MATLAB 1:48 -> Python :48
                    # MATLAB 272:320 -> Python 271:
                        trn_mask[..., :48]  = 1
                        trn_mask[..., 271:] = 1

                        loss_mask[..., :48]  = 0
                        loss_mask[..., 271:] = 0
                    
                    zero_filled = IFFT(ksp_pf * trn_mask)
                    zero_filled = torch.sum( zero_filled * torch.conj(coil), axis=1, keepdims=True)
                    zero_filled = torch.cat([torch.real(zero_filled), torch.imag(zero_filled)], axis=1)

                    recon = model(zero_filled, coil, trn_mask)
                    recon_ksp = FFT(recon * coil) * loss_mask
                    label_ksp = ksp_pf * loss_mask

                    loss_total += L1_L2_Loss(recon_ksp, label_ksp)

                val_loss += (loss_total / k_MM).item()
                
                label_png = IFFT(ksp)
                label_png = torch.sum(label_png * torch.conj(coil), axis=1, keepdims=True) * coil_rss
                zf_cmplx  = (zero_filled[:, 0:1, :, :] + 1j * zero_filled[:, 1:2, :, :]) * coil_rss
                recon_png = recon * coil_rss
                
                x = zf_cmplx.squeeze().cpu().detach().numpy()
                y = label_png.squeeze().cpu().detach().numpy()
                z = recon_png.squeeze().cpu().detach().numpy()

                zf_copy    = np.flipud(x).copy()
                label_copy = np.flipud(y).copy()
                z_copy     = np.flipud(z).copy()

                norm_val = np.max(np.abs(label_copy)) + 1e-12
                zf_copy    = zf_copy / norm_val
                label_copy = label_copy / norm_val
                z_copy     = z_copy / norm_val
                combined = np.concatenate([zf_copy, label_copy, z_copy], axis=1)
                abs_xx = torch.abs(torch.tensor(combined)).float()
                abs_xx = torch.clamp(abs_xx, max=window_scale) / window_scale
                save_path = f"{test_add}/{EXP_name}/pngs/Val_Slice_{SliceNumber.item()}_E_{epoch}.png"
                save_image(abs_xx, save_path)

        val_loss /= len(val_loader)
        model.train()        
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'loss': val_loss,
            }, f"./{test_add}/{EXP_name}/model/BestModel_Val_R{Acc_R}_Epoch{epoch}.pth")


        scheduler.step(val_loss)   # for ReduceLROnPlateau
        model.train()


        def val_np(x):
            if isinstance(x, (int, float)):
                return np.array(x)
            return maybe_softplus(x).detach().cpu().numpy()

        def add_hist(hist, name, x):
            hist.setdefault(name, []).append(val_np(x))

        def plot_hist(hist, name, save_dir):
            arr = np.array(hist[name])
            plt.figure(figsize=(8, 5))

            if arr.ndim == 1:
                plt.plot(arr, label=name)
            else:
                arr = arr.reshape(arr.shape[0], -1)
                for i in range(arr.shape[1]):
                    plt.plot(arr[:, i], label=f"{name}_{i+1}", alpha=0.8)

            plt.xlabel("Epochs")
            plt.ylabel(name)
            plt.title(f"{name} Over Time")
            plt.grid(True)
            if name.lower() != "eta":
                plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{name}_Plot.png", dpi=300)
            plt.close()

        # =========================
        # After each epoch
        # =========================
        hist.setdefault("train_loss", []).append(train_loss)
        hist.setdefault("val_loss", []).append(val_loss)

        dc = model.DataConsistency

        if Network == "UMPIRE":
            add_hist(hist, "beta1", dc.beta1)
            add_hist(hist, "beta2", dc.beta2)
            add_hist(hist, "lambda1", model.lambda1)
            add_hist(hist, "lambda2", model.lambda2)
            add_hist(hist, "eta", dc.eta)

        elif Network == "PDDL":
            add_hist(hist, "mu", dc.mu)
            add_hist(hist, "lambda1", model.lambda1)

            if not CG_flag:
                add_hist(hist, "eta", dc.eta)

        # save all
        sio.savemat(f"{save_dir}/Training_History.mat", {
            k: np.array(v, dtype=object) for k, v in hist.items()
        })

        # =========================
        # Train + Validation Loss
        # =========================
        plt.figure(figsize=(8, 5))
        plt.plot(hist["train_loss"], label="Train Loss")
        plt.plot(hist["val_loss"], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Train and Validation Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/Train_Val_Loss_Plot.png", dpi=300)
        plt.close()
        # plot all
        for name in hist.keys():
            if name not in ["train_loss", "val_loss"]:
                plot_hist(hist, name, save_dir)
