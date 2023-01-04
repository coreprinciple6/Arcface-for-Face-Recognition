from .loader import get_train_loader, get_bfc
from .utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from .model import Backbone, Arcface, l2_norm
from .eval import evaluate

import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from PIL import Image
from torchvision import transforms as trans
import cv2

class face_learner(object):
    def __init__(self, conf, inference=False):
        print(conf)
        self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
        print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        self.bfc_img_path = '/home/coreprinciple/work_station/ML_project/FR/face_recognition/bfc_data'
        if not inference:
            self.milestones = conf.milestones
            self.loader, self.class_num = get_train_loader(conf)        

            self.writer = SummaryWriter(conf.log_path)
            self.step = 0
            self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)

            print('two model heads generated')

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)
            # {'params': paras_only_bn} 
            self.optimizer = optim.SGD([
                                {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4}]
                            , lr = conf.lr, momentum = conf.momentum)
            print(self.optimizer)

            print('optimizers generated')    
            self.board_loss_every = len(self.loader)//100
            self.evaluate_every = len(self.loader)//10
            self.save_every = len(self.loader)//5
            self.bfc,self.bfc_issame = get_bfc(conf)
        else:
            self.threshold = conf.threshold
    
    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        save_path = conf.save_path
        torch.save(self.model.state_dict(), f'{save_path}/model_acc_{round(accuracy*100,3)}.pth')
    
    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=True):
        save_path = conf.save_path         
        self.model.load_state_dict(torch.load('backbone_50.pth')) #model_ir_se50.pth

        
    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)


    '''HERE IS EVALUATION FOR BFC'''
    def evaluate(self, conf, carray, issame, nrof_folds = 5, tta = False):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), conf.embedding_size])
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                temp_names = carray[idx:idx + conf.batch_size]
                img_list = []
                #img_list = np.zeros(len(temp_names),dtype='float32')
                for i,item in enumerate(temp_names):
                    fold = item.split('_')[0]
                    img_path = f'{self.bfc_img_path}/{str(fold)}/{str(item)}'
                    img = cv2.imread(img_path)
                    img = img.transpose(2,0,1)
                    img_list.append(img)
                img_list = np.array(img_list)
                batch = torch.tensor(img_list)
                batch = batch.to(torch.float) 
   
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch).cpu()
                else:
                    embeddings[idx:idx + conf.batch_size] = self.model(batch.to(conf.device)).cpu()
                idx += conf.batch_size

            if idx < len(carray):
                temp_names2 = carray[idx:]
                img_list2 = np.zeros(len(temp_names2),dtype='float32')
                for i,item in tqdm(enumerate(img_list2)):
                    fold = item.split('_')[0]
                    img_path = f'{self.bfc_img_path}/{str(fold)}/{str(item)}'#'/home/ishrat/ishrat/Final_DSET_insightface/'+str(fold)+'/'+str(item)
                    img = cv2.imread(img_path)
                    img = img.transpose(2,0,1)
                    img_list[i] = img
                img_list = np.array(img_list)
                batch = torch.tensor(img_list)
                batch = batch.to(torch.float)             
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.to(conf.device)).cpu()
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor
    

    def train(self, conf, epochs):
        self.model.train()
        running_loss = 0.            
        for e in range(epochs):
            print('epoch {} started'.format(e))
                              
            for imgs, labels in tqdm(iter(self.loader)):
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                thetas = self.head(embeddings, labels)
                loss = conf.ce_loss(thetas, labels)
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()
                
                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    running_loss = 0.
                
                if self.step % self.evaluate_every == 0 and self.step != 0:
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.bfc, self.bfc_issame)
                    self.board_val('bfc', accuracy, best_threshold, roc_curve_tensor)

                    self.model.train()
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)
                    
                self.step += 1

        self.save_state(conf, accuracy, to_save_folder=True, extra='final')
        return accuracy,roc_curve_tensor
          