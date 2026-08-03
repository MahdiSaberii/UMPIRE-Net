# UMPIRE-Net: Unrolled Magnitude–Phase Regularization Network for Accelerated MRI

<p align="center">
  <img src="./Data/pipeline.jpg" width="900">
</p>


## Abstract
<p align="justify">
MRI reconstruction from undersampled k-space is an ill-posed inverse problem. Physics-driven deep learning (PD-DL) addresses this problem by integrating the MRI forward model with learned regularization in algorithm-unrolling frameworks. However, most PD-DL methods reconstruct complex-valued images directly, implicitly coupling magnitude and phase within a single learned representation. This can be limiting in settings requiring accurate phase modeling, such as partial Fourier (PF) imaging, where recovery of omitted asymmetric k-space data depends on image phase. We propose UMPIRE-Net (Unrolled Magnitude-Phase In REgularization Network), a PD-DL method with separate learned magnitude and phase regularizers and a novel data-fidelity formulation that enforces measurement consistency, reducing reliance on externally estimated phase. Across multiple datasets and acceleration factors, UMPIRE-Net outperforms a conventional complex-valued PD-DL baseline, producing sharper images with fewer artifacts.
</p>
<p align="center">
  <img src="./Data/PD_PDFS.jpg" width="900">
</p>

## Repository Structure

```text
UMPIRE-Net/
├── config/
│   └── Config.yaml
├── src/
│   ├── unet
│   ├── DC.py
│   ├── DataLoader.py
│   ├── Unrolled_Network.py
│   └── Utils.py
├── train.py
└── test.py
```

## Quick Start
Note: This code was tested with `torch==2.2.1+cu121`. 

## Installation

**Note:** This code was tested with `torch==2.2.1+cu121`.

### 1. Clone this repository

```bash
git clone https://github.com/MahdiSaberii/Non-Cartesian-SSDU-MRI.git
cd Non-Cartesian-SSDU-MRI
```

### 2. Create and activate conda environment

```bash
conda create -n non_cartesian_ssdu_mri python=3.10 -y
conda activate non_cartesian_ssdu_mri
```

### 3. Install PyTorch

```bash
pip install torch==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install remaining requirements

```bash
pip install -r requirements.txt
```
