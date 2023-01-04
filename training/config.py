from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans

def get_config(training = True):
    conf = edict()
    conf.data_path = Path('dataset')
    conf.save_path = 'save'
    conf.input_size = [112,112]
    conf.embedding_size = 512
    conf.use_mobilfacenet = False
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'ir_se'
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    conf.data_mode = 'emore'
    conf.emore_folder = conf.data_path
    conf.batch_size = 100 # irse net depth 50 

#--------------------Training Config ------------------------    
    if training:        
        conf.log_path = 'log'
        conf.save_path = 'save'
        conf.lr = 1e-3
        conf.momentum = 0.9
        conf.milestones = [12,15,18]
        conf.pin_memory = True
        conf.num_workers = 1
        conf.ce_loss = CrossEntropyLoss()

    return conf
