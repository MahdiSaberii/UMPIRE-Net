import glob
import os
import re
import scipy.io as sio
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np 

class DataLoaderSL(Dataset):
    def __init__(self, Training_path, slice_num=None):
        self.root = Training_path
        dataset_folder = Path(Training_path).parts[-3]   # PDFS_300
        self.Dataset_name = dataset_folder.split("_")[0] # PDFS

        if slice_num is not None:
            pattern = f"slice_{slice_num:03d}.mat"
            selected_file = os.path.join(Training_path, pattern)

            if not os.path.exists(selected_file):
                raise FileNotFoundError(f"File not found: {selected_file}")

            self.datapath = [selected_file]
            print(f"Loading only one file: {self.datapath[0]}")

        else:
            self.datapath = sorted(glob.glob(os.path.join(Training_path, "slice_*.mat")))
            print(f"# Data: {len(self.datapath)}")

    def __getitem__(self, index):
        file_path = self.datapath[index]
        FileName = os.path.basename(file_path)

        match = re.match(r"slice_(\d+)\.mat", FileName)
        if match:
            SliceNumber = int(match.group(1))
        else:
            raise ValueError(f"Unexpected filename format: {FileName}")

        mat = sio.loadmat(file_path)

        # AxT2, CorPD
        ksp  = mat["kspace"]   # expected shape: [16, 320, 320]
        coil = mat["coils"]    # expected shape: [16, 320, 320]
        
        if self.Dataset_name == "PDFS":
            # print(ksp.shape, coil.shape)
            ksp  = np.transpose(ksp,  (2, 0, 1))
            coil = np.transpose(coil, (2, 0, 1))
            # print(ksp.shape, coil.shape, "After")

        return ksp, coil, SliceNumber, FileName

    def __len__(self):
        return len(self.datapath)
