import torch
import torchvision
import json
import torch.nn as nn
import json
import sys
import warnings
from torch.utils.data import Dataset, DataLoader
from dataloader_vcoco import Rescale,ToTensor,vcoco_Dataset, vcoco_collate
from torchvision import transforms
import numpy as np
import random
import os
from tqdm import tqdm
import torch.optim as optim

from model_ywkim import VSGNet
from train_test_ywkim import train_test
warnings.filterwarnings("ignore")  

device= torch.device('cuda')

num_epochs = 50
batch_size=8

seed=10
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
def _init_fn(worker_id):
    np.random.seed(int(seed))



# with open('../infos/directory.json') as fp: all_data_dir=json.load(fp)
all_data_dir = '../All_data_vcoco/'

annotation_train=all_data_dir+'Annotations_vcoco/train_annotations.json'
image_dir_train=all_data_dir+'Data_vcoco/train2014/'

annotation_val=all_data_dir+'Annotations_vcoco/val_annotations.json'
image_dir_val=all_data_dir+'Data_vcoco/train2014/'

annotation_test=all_data_dir+'Annotations_vcoco/test_annotations.json'
image_dir_test=all_data_dir+'Data_vcoco/val2014/'

vcoco_train=vcoco_Dataset(annotation_train,image_dir_train,transform=transforms.Compose([Rescale((400,400)),ToTensor()]))
vcoco_val=vcoco_Dataset(annotation_val,image_dir_val,transform=transforms.Compose([Rescale((400,400)),ToTensor()]))
vcoco_test=vcoco_Dataset(annotation_test,image_dir_test,transform=transforms.Compose([Rescale((400,400)),ToTensor()]))


#import pdb;pdb.set_trace()
dataloader_train = DataLoader(vcoco_train, batch_size,
                        shuffle=True,collate_fn=vcoco_collate,num_workers=8,worker_init_fn=_init_fn)#num_workers=batch_size
dataloader_val = DataLoader(vcoco_val, batch_size,
                        shuffle=True,collate_fn=vcoco_collate,num_workers=8,worker_init_fn=_init_fn)#num_workers=batch_size
dataloader_test = DataLoader(vcoco_test, batch_size,
                        shuffle=False,collate_fn=vcoco_collate,num_workers=8, worker_init_fn=_init_fn)#num_workers=batch_size
dataloader={'train':dataloader_train,'val':dataloader_val,'test':dataloader_test}


model = VSGNet()

for name, p in model.named_parameters():
    if name.split('.')[0] == 'Conv_pretrain':
        p.requires_grad = False

optim1 = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)


lambda1 = lambda epoch: 1.0 if epoch < 10 else (10 if epoch < 28 else 1) 
lambda2 = lambda epoch: 1
lambda3 = lambda epoch: 1
scheduler=optim.lr_scheduler.LambdaLR(optim1,lambda1)

model = nn.DataParallel(model)
model.to(device)

epoch = 0
mean_best = 0

train_test(model, optim1, scheduler, dataloader, num_epochs, batch_size, epoch, mean_best)
