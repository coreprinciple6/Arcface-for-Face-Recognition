from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np


def de_preprocess(tensor):
    return tensor*0.5 + 0.5
    
def get_train_dataset(imgs_folder):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num

def get_train_loader(conf):
    if conf.data_mode == 'emore':
        ds, class_num = get_train_dataset('/home/coreprinciple/work_station/ML_project/FR/face_recognition/bfc_data')
    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    return loader, class_num


def get_bfc(conf):
    is_same = []
    bfc = []
    with open('/home/coreprinciple/work_station/ML_project/FR/face_recognition/same2.csv') as f:
        for line in f:
            if line.strip() == 'True':
                is_same.append(True)
            else:
                is_same.append(False)
    is_same = np.array(is_same)

    with open('/home/coreprinciple/work_station/ML_project/FR/face_recognition/names.txt') as f:
        for line in f:
            bfc.append(line.strip())
    bfc = np.array(bfc)

    return bfc,is_same
