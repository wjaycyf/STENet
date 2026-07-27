import os
import configparser


class Config:
    def __init__(self, config_path):
        parser = configparser.ConfigParser()
        parser.read(config_path)

        # experiment
        self.seed = int(parser.get('experiment', 'seed'))

        # training
        self.dataset_path = parser.get('training', 'dataset_path')
        self.save_dir = parser.get('training', 'save_dir')
        self.stage = int(parser.get('training', 'stage'))
        self.log_dir = parser.get('training', 'log_dir')
        self.log_dir = os.path.join(self.save_dir, f'stage{self.stage}_{self.log_dir}')

        self.nThreads = int(parser.get("training", "nThreads"))
        self.num_epochs = int(parser.get("training", "num_epochs"))
        self.lr = float(parser.get("training", "lr"))
        self.batch_size = int(parser.get('training', 'batch_size'))
        self.patch_size = int(parser.get('training', 'patch_size'))
        self.finetuning = (parser.get('training', 'finetuning') == 'True')
        self.save_train_img = (parser.get('training', 'save_train_img') == 'True')

        self.scale = int(parser.get('training', 'scale'))
        self.num_seq = int(parser.get('training', 'num_seq'))

        self.lr_warping_loss_weight = float(parser.get("training", "lr_warping_loss_weight"))
        self.hr_warping_loss_weight = float(parser.get("training", "hr_warping_loss_weight"))
        self.flow_loss_weight = float(parser.get("training", "flow_loss_weight"))
        self.D_TA_loss_weight = float(parser.get("training", "D_TA_loss_weight"))
        self.R_TA_loss_weight = float(parser.get("training", "R_TA_loss_weight"))
        self.Net_D_weight = float(parser.get("training", "Net_D_weight"))
        self.memory_train_mode = parser.get('training', 'memory_train_mode', fallback='off')
        self.memory_eval_mode = parser.get('training', 'memory_eval_mode', fallback='sequential')
        self.memory_debug = (parser.get('training', 'memory_debug', fallback='False') == 'True')

        self.gpu = parser.get("training", "gpu")

        # Network
        self.temporal_module = parser.get('network', 'temporal_module', fallback='cdmr')
        self.memory_momentum = float(parser.get('network', 'memory_momentum', fallback='0.9'))
        self.motion_threshold = float(parser.get('network', 'motion_threshold', fallback='1.0'))
        self.confidence_threshold = float(parser.get('network', 'confidence_threshold', fallback='0.5'))
        self.in_channels = int(parser.get('network', 'in_channels'))
        self.dim = int(parser.get('network', 'dim'))
        self.ds_kernel_size = int(parser.get('network', 'ds_kernel_size'))
        self.us_kernel_size = int(parser.get('network', 'us_kernel_size'))
        self.num_RDB = int(parser.get('network', 'num_RDB'))
        self.growth_rate = int(parser.get('network', 'growth_rate'))
        self.num_dense_layer = int(parser.get('network', 'num_dense_layer'))
        self.num_flow = int(parser.get('network', 'num_flow'))
        self.num_msa = int(parser.get('network', 'num_msa', fallback=parser.get('network', 'num_FRMA', fallback='1')))
        self.num_FRMA = self.num_msa
        self.num_transformer_block = int(parser.get('network', 'num_transformer_block'))
        self.num_heads = int(parser.get('network', 'num_heads'))
        self.LayerNorm_type = parser.get('network', 'LayerNorm_type')
        self.ffn_expansion_factor = float(parser.get('network', 'ffn_expansion_factor'))
        self.bias = (parser.get('network', 'bias') == 'True')
        self.kcssm_d_state = int(parser.get('network', 'kcssm_d_state', fallback='16'))
        self.kcssm_expand = float(parser.get('network', 'kcssm_expand', fallback='2.0'))
        self.kcssm_local_kernel = int(parser.get('network', 'kcssm_local_kernel', fallback='3'))
        self.kcssm_enable_local = (parser.get('network', 'kcssm_enable_local', fallback='True') == 'True')

        # validation
        self.val_period = int(parser.get('validation', 'val_period'))

        # test
        self.custom_path = parser.get('test', 'custom_path')
