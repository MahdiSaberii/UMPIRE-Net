import argparse

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None
        
        # directory for the out directory
        self.parser.add_argument('--out_path', type=str, default='results/', help='results file directory')
        
        # hyperparameters for the learning
        self.parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
        self.parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
        self.parser.add_argument('--batchSize', type=int, default=1, help='batch size')
        self.parser.add_argument('--LR_sch', type=bool, default=False, help='Is there LR schedule or not')
        self.parser.add_argument('--save_freq', type=int, default=5, help='result saving frequency')
        self.parser.add_argument('--train_shuffle', action="store_true")

        # hyperparameters for the unrolled network
        self.parser.add_argument('--model', type=str, default='None', help='choose the model {Unet_share, ResNet_share, Unet_multi, ResNet_multi, Unet_time_emb, ResNet_time_emb}')
        self.parser.add_argument('--nb_unroll_blocks', type=int, default=10, help='number of unrolled blocks')
        self.parser.add_argument('--nb_res_blocks', type=int, default=15, help="number of residual blocks in ResNet")
        self.parser.add_argument('--CG_Iter', type=int, default=10, help='number of Conjugate Gradient iterations for DC')
        self.parser.add_argument('--ntotal_slices', type=int, default=300, help='number of total slices')
        self.parser.add_argument('--ntrain_slices', type=int, default=1, help='number of train slices')
        self.parser.add_argument('--idx_slice', type=int, default=0, help='index of train slice in case of ntrain_slices = 1')
        self.parser.add_argument('--domain', type=str, default='ksp', help='loss function domain {img, ksp}')
        
        # For Unrolling Algorithm
        self.parser.add_argument('--Unroll_algo', type=str, default='VAMP', help='Unrolling Algorithm {VSQP, PGD, ADMM, VAMP}')
        self.parser.add_argument('--mu_init', type=float, default=1.5e-2, help='mu init value')
        self.parser.add_argument('--DC_unshared', action="store_true")
        self.parser.add_argument('--gamma_unshared', action="store_true", help='used in ADMM and VAMP')
        self.parser.add_argument('--gamma_init', type=float, default=1e-1, help='gamma init value used in ADMM and VAMP')
        
        # for Unet
        self.parser.add_argument("--use_norm", type=bool, default=True, help="Use normalization (default: True)")
        self.parser.add_argument('--use_time_emb', action="store_true")
        self.parser.add_argument('--channel_mult', type=int, nargs='+', default=[1, 2, 3], help='number of multi-channels')
        self.parser.add_argument('--num_channels', type=int, default=32, help='number of basic channel')
        

        # for ResNet time-embedding
        self.parser.add_argument('--tau_init', type=float, default=1e-1, help='tau in resnet-te')
        self.parser.add_argument('--time_emb_ratio', type=int, default=2, help='number of basic channel')

       
        # hyperparameters for the dataset
        self.parser.add_argument('--data_type', type=str, default='PD', choices=['PD', 'PDFS', 'AXT2'],  help='data type')
        self.parser.add_argument('--acc_rate', type=int, default=4, help='acceleration rate')
        self.parser.add_argument('--nrow_GLOB', type=int, default=320, help='number of rows of the slices in the dataset')
        self.parser.add_argument('--ncol_GLOB', type=int, default=368, help='number of columns of the slices in the dataset')
        self.parser.add_argument('--ncoil_GLOB', type=int, default=15, help='number of coils of the slices in the dataset')
        self.parser.add_argument('--ACS_length', type=int, default=24, help='number of ACS lines taken')
        
        # hyperparameters for testing
        self.parser.add_argument('--weight_epoch', type=int, default=49, help='epoch of weights')  
        self.parser.add_argument('--select_slices', type=str, default='False', help='Select test slices')  


    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        
        if self.conf.Unroll_algo == "PGD":
            self.conf.mu_init = 2e-1
        elif self.conf.Unroll_algo == "VSQP":
            self.conf.mu_init = 5e-2
        elif self.conf.Unroll_algo == "ADMM":
            self.conf.DC_unshared = True
        elif self.conf.Unroll_algo == "VAMP":
            self.conf.DC_unshared = True
            self.conf.gamma_unshared = True
            if '_share' in self.conf.model or '_multi' in self.conf.model:
                raise ValueError("conf.model should have 'time_emb' instead of 'share' or 'multi' for VAMP")
        
        if self.conf.model == "ResNet_time_emb" and self.conf.data_type == "PDFS":
            self.conf.learning_rate = 1.8e-4

        if "time_emb" in self.conf.model:
            self.conf.use_time_emb = True
                
        if "AXT2" in self.conf.data_type:
            self.conf.ncol_GLOB = 320
            self.conf.ncoil_GLOB = 16
            self.conf.learning_rate = 2e-4
                                 
        return self.conf
