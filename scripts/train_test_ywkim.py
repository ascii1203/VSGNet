import torch
import torch.nn as nn
import time
import errno
import os
import gc
import pickle
import shutil
import json
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import calculate_ap_classwise as ap
import matplotlib.pyplot as plt
import random

import helpers_preprocess as helpers_pre
import pred_vis as viss
import prior_vcoco as prior
import proper_inferance_file as proper
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

sigmoid=nn.Sigmoid()
criterion = nn.BCELoss(reduction='none')

### Fixing Seeds#######
device = torch.device("cuda")
seed=10
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
softmax=nn.Softmax()
##########################################

action_size= 29

### extending number of people ###
def extend_person(inputt, extend_number):
    res = np.zeros([1, np.shape(inputt)[-1]])
    for a in inputt:
        x = np.repeat(a.reshape(1, np.shape(inputt)[-1]), extend_number, axis=0)
        res = np.concatenate([res, x], axis=0)
    return res[1:]

####### Extening Number of Objects##########
def extend_object(inputt,extend_number):
    res=np.zeros([1,np.shape(inputt)[-1]])
    x=np.array(inputt.tolist()*extend_number)
    res=np.concatenate([res,x],axis=0)
    return res[1:]
    
    
############## Filtering results for preparing the output as per as v-coco###############################
def filtering(predicted_HOI,true,persons_np,objects_np,filters,pairs_info,image_id):
    res1=np.zeros([1,action_size])
    res2=np.zeros([1,action_size])
    res3=np.zeros([1,action_size])
    res4=np.zeros([1,4])
    res5=np.zeros([1,4])
    dict1={}
    a=0
    increment=[int(i[0]*i[1]) for i in pairs_info]
    #import pdb;pdb.set_trace()
    start=0
    for index,i in enumerate(filters):
        res1=np.concatenate([res1,predicted_HOI[index].reshape(1,action_size)],axis=0)
        res2=np.concatenate([res2,true[index].reshape(1,action_size)],axis=0)
        res3=np.concatenate([res3,predicted_HOI[index].reshape(1,action_size)],axis=0)
        res4=np.concatenate([res4,persons_np[index].reshape(1,4)],axis=0)
        res5=np.concatenate([res5,objects_np[index].reshape(1,4)],axis=0)
        if index==start+increment[a]-1:
            #import pdb;pdb.set_trace()
            dict1[int(image_id[a]),'score']=res3[1:]
            dict1[int(image_id[a]),'pers_bbx']=res4[1:]
            dict1[int(image_id[a]),'obj_bbx']=res5[1:]
            res3=np.zeros([1,action_size])
            res4=np.zeros([1,4])
            res5=np.zeros([1,4])
            start+=increment[a]
            a+=1
            #import pdb;pdb.set_trace()
    return dict1
    
def LIS(x, T, k, w):
    return T/(1+np.exp(k-w*x))

def train_test(model, optimizer, scheduler, dataloader, num_epochs, batch_size, start_epoch, mean_best):
    
    writer = SummaryWriter('runs/VSGNet_exp_VSG')

    loss_epoch_train = []
    loss_epoch_val = []
    loss_epoch_test = []
    initial_time = time.time()
    result = []
    
    torch.cuda.empty_cache()
    phases = ['train','val','test']
    
    end_epoch = start_epoch + num_epochs
    
    iteration = 0
    
    for epoch in range(start_epoch, end_epoch):
#         scheduler.step()
        print('Epoch {}/{}'.format(epoch+1, end_epoch))
        print('-' * 10)
        initial_time_epoch = time.time()
        
        for phase in phases:
            if phase == 'train':
                model.train()
            elif phase == 'val':
                model.train()
            else:
                model.eval()
            
            print('In {}'.format(phase))
            
            detections_train = []
            detections_val = []
            detections_test = []
            
            true_scores_class = np.ones([1, 80], dtype=int)
            true_scores = np.ones([1, 29], dtype=int)
            true_scores_single = np.ones([1, 1], dtype=int)
            
            predicted_scores_class = np.ones([1, 80], dtype=float)
            predicted_scores = np.ones([1, 29], dtype=float)
            predicted_scores_single = np.ones([1, 1], dtype=float)
            
            acc_epoch = 0
            
            
            for iterr, i in enumerate(tqdm(dataloader[phase])):
                if iterr%20 == 0:
                    torch.cuda.empty_cache()
                
                inputs = i[0].to(device)
                labels = i[1].to(device)
                labels_single = i[2].to(device)
                image_id = i[3]
                pairs_info = i[4]
                minibatch_size = len(pairs_info)
                
                optimizer.zero_grad()
                
                if phase == 'train':
                    nav = torch.tensor([[0,epoch]]*minibatch_size).to(device)
                elif phase == 'val':
                    nav = torch.tensor([[1,epoch]]*minibatch_size).to(device)
                else:
                    nav = torch.tensor([[2,epoch]]*minibatch_size).to(device)
                
                true = (labels.data).cpu().numpy()
                true_single = (labels_single.data).cpu().numpy()
                    
                with torch.set_grad_enabled(phase=='train' or phase=='val'):
                    model_out = model(inputs, pairs_info, pairs_info, image_id, nav, phase)
                    i_ho = model_out[0]
                    p_Ref = model_out[1]
                    p_Att = model_out[2]
                    p_Graph = model_out[3]
                    
                    predicted_HOI = sigmoid(p_Ref).data.cpu().numpy()
                    predicted_single = sigmoid(i_ho).data.cpu().numpy()
                    predicted_HOI_Att = sigmoid(p_Att).data.cpu().numpy()
                    predicted_HOI_Graph = sigmoid(p_Graph).data.cpu().numpy()
                    predicted_HOI_pair = predicted_HOI
                    
                    start_index = 0
                    start_obj = 0
                    start_pers = 0
                    start_tot = 0
                    pers_index = 1
                    
                    persons_score_extended = np.zeros([1,1])
                    objects_score_extended = np.zeros([1,1])
                    class_ids_extended = np.zeros([1,1])
                    persons_np_extended = np.zeros([1,4])
                    objects_np_extended = np.zeros([1,4])
                    start_no_obj = 0
                    class_ids_total = []
                    
                    # extendind person and object boxes and confidence scores to multiply with all pairs (?)
                    
                    for batch in range(len(pairs_info)):
                        persons_score = []
                        objects_score = []
                        class_ids = []
                        objects_score.append(float(1)) # no object
                        
                        this_image = int(image_id[batch]) # image_id
                        scores_total = helpers_pre.get_compact_detections(this_image, phase)
                        persons_score, objects_score, persons_np, objects_np, class_ids = \
                        scores_total['person_bbx_score'], scores_total['objects_bbx_score'], \
                        scores_total['person_bbx'], scores_total['objects_bbx'], \
                        scores_total['class_id_objects']
                        
                        temp_scores = extend_person(np.array(persons_score).reshape(len(persons_score),1), int(pairs_info[batch][1])) # num_obj
                        persons_score_extended = np.concatenate([persons_score_extended, temp_scores])
                        
                        temp_scores = extend_person(persons_np, int(pairs_info[batch][1]))
                        persons_np_extended = np.concatenate([persons_np_extended, temp_scores])
                        
                        temp_scores = extend_object(np.array(objects_score).reshape(len(objects_score), 1), int(pairs_info[batch][0]))
                        objects_score_extended = np.concatenate([objects_score_extended, temp_scores])
                        
                        temp_scores = extend_object(objects_np, int(pairs_info[batch][0]))
                        objects_np_extended = np.concatenate([objects_np_extended, temp_scores])
                        
                        temp_scores = extend_object(np.array(class_ids).reshape(len(class_ids),1), int(pairs_info[batch][0]))
                        class_ids_extended = np.concatenate([class_ids_extended, temp_scores])
                        class_ids_total.append(class_ids)
                        
                        
                    persons_score_extended = LIS(persons_score_extended, 8.3, 12, 10)
                    objects_score_extended = LIS(objects_score_extended, 8.3, 12, 10)
                    
                    predicted_HOI = predicted_HOI * predicted_single * \
                    predicted_HOI_Att * predicted_HOI_Graph * \
                    objects_score_extended[1:] * persons_score_extended[1:]
                    loss_mask = prior.apply_prior(class_ids_extended[1:], predicted_HOI)
                    predicted_HOI = loss_mask * predicted_HOI
                        
                    N_b = minibatch_size * 29
                    hum_obj_mask = torch.Tensor(objects_score_extended[1:] * \
                                                persons_score_extended[1:] * loss_mask).cuda()
                    
                    lossf = torch.sum( criterion(sigmoid(i_ho) * sigmoid(p_Ref) * sigmoid(p_Att)\
                                                 * sigmoid(p_Graph) * hum_obj_mask, labels.float()))/N_b
                    lossc = lossf.item()
                    
                    acc_epoch += lossc
                    if phase == 'train' or phase == 'val':
                        lossf.backward()
                        optimizer.step()
                        iteration += 1
                    
                        writer.add_scalar('training loss', lossc, iteration)
                        
                    
                    del lossf
                    del model_out
                    del inputs
                    del labels
                    
                #prepairing for storing results
                predicted_scores = np.concatenate((predicted_scores, predicted_HOI), axis=0)
                true_scores = np.concatenate((true_scores, true), axis=0)
                predicted_scores_single = np.concatenate((predicted_scores_single, predicted_single), axis=0)
                true_scores_single = np.concatenate((true_scores_single, true_single), axis=0)
                
                
                if phase == 'test':
                    all_scores = filtering(predicted_HOI, true, persons_np_extended[1:], objects_np_extended[1:], predicted_single, pairs_info, image_id)
                    proper.infer_format(image_id, all_scores, phase, detections_test, pairs_info)
                
            if phase == 'test':
#                 loss_epoch_test.append((acc_epoch))
                AP, AP_single = ap.class_AP(predicted_scores[1:,:], true_scores[1:,:], \
                                            predicted_scores_single[1:,], true_scores_single[1:,])
                AP_test = pd.DataFrame(AP, columns=['Name_TEST', 'Score_TEST'])
                AP_test_single = pd.DataFrame(AP_single, columns=['Name_TEST','Score_TEST'])
        AP_final = pd.concat([AP_test], axis=1)
        AP_final_single = pd.concat([AP_test_single], axis=1)
        result.append(AP_final)
        
        print('APs in epoch {}'.format(epoch+1))
        print(AP_final)
        print(AP_final_single)
    
    
    
