import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import pool_pairing as ROI

import os
import numpy as np


pool_size=(10,10)
lin_size = 1024
action_size = 29
projection_size = 512

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size()[0], -1)

class VSGNet(nn.Module):
    def __init__(self):
        super(VSGNet, self).__init__()
        
        self.flat = Flatten()
        self.sigmoid = nn.Sigmoid()
        
        model = torchvision.models.resnet152(pretrained=True)
        self.Conv_pretrain = nn.Sequential(*list(model.children())[0:7])        
        
        ######### Convolutional Blocks for human,objects and the context##############################
        self.Conv_people=nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
        )
        self.Conv_objects=nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
        )
        self.Conv_context=nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
        )
        
        #### Attention feature model
        self.Conv_spatial = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 32, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.AvgPool2d((13, 13), padding=0, stride=(1, 1)),
        )
        self.W_Spat = nn.Sequential(
            nn.Linear(32, 512),
            nn.ReLU(),
        )
        
        ### Prediction model for attention features
        self.W_Att = nn.Sequential(
            nn.Linear(512, 29),
        )
        
        ### Graph model basic structure
        self.W_oh = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.W_ho = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        
        #### Interaction prediction model
        self.W_vis = nn.Sequential(
            nn.Linear(lin_size*3, projection_size), 
            nn.ReLU(),
        )
        self.W_IP = nn.Sequential(
            nn.Linear(projection_size, 1),
        )
        self.W_Ref = nn.Sequential(
            nn.Linear(projection_size, action_size),
        )
        
        self.W_graph = nn.Sequential(
            nn.Linear(1024*2, 29),
        )
               ############################################################################################### 
                
       
    def forward(self, x, pairs_info, pairs_info_augmented, image_id, flag_, phase):
        
        out1 = self.Conv_pretrain(x)
        
        rois_people, rois_objects, spatial_locs, union_box = ROI.get_pool_loc(out1, image_id, flag_, size=pool_size, spatial_scale=25, batch_size=len(pairs_info))
        
        ### Defining The Pooling Operations ####### 
        x,y=out1.size()[2],out1.size()[3]	
        hum_pool=nn.AvgPool2d(pool_size,padding=0,stride=(1,1))
        obj_pool=nn.AvgPool2d(pool_size,padding=0,stride=(1,1))
        context_pool=nn.AvgPool2d((x,y),padding=0,stride=(1,1))
        #################################################



        ### Human###
        residual_people=rois_people
        res_people=self.Conv_people(rois_people)+residual_people
        res_av_people=hum_pool(res_people)
        out2_people=self.flat(res_av_people)
        ###########


        ##Objects##
        residual_objects=rois_objects
        res_objects=self.Conv_objects(rois_objects)+residual_objects
        res_av_objects=obj_pool(res_objects)
        out2_objects=self.flat(res_av_objects)
        #############    
        
        
        #### Context ######
        residual_context=out1
        res_context=self.Conv_context(out1)+residual_context
        res_av_context=context_pool(res_context)
        out2_context=self.flat(res_av_context)
        #################
        
        ## Attention features ##
        a_ho = self.W_Spat(self.flat(self.Conv_spatial(union_box)))
        
        ### Making Essential Pairing##########
        pairs, people, objects_only = ROI.pairing(out2_people, out2_objects, out2_context, spatial_locs, pairs_info)
        ######################################
        
        ### Interaction Probability ########
        f_Vis = self.W_vis(pairs)
        f_Ref = f_Vis * a_ho
        i_ho = self.W_IP(f_Ref)
        interaction_prob = self.sigmoid(i_ho)
        p_Ref = self.W_Ref(f_Ref)
        
        ### Prediction from attention features
        
        p_Att = self.W_Att(a_ho)
        
        
        ### Graph model base structure
        people_t = people
        objects_only = objects_only
        combine_g = []
        people_f = []
        objects_f = []
        pairs_f = []
        start_p = 0
        start_o = 0
        start_c = 0
        
        for batch_num, l in enumerate(pairs_info):
            
            ### Slicing ###
            people_this_batch = people_t[start_p : start_p + int(l[0])]
            num_peo = len(people_this_batch)
            
            objects_this_batch = objects_only[start_o : start_o + int(l[1])][1:]
            # because first index means no object
            no_objects_this_batch = objects_only[start_o : start_o + int(l[1])][0]
            num_obj = len(objects_this_batch)
            
            interaction_prob_this_batch = interaction_prob[start_c : start_c + \
                                                           int(l[0]) * int(l[1])]
            
            if num_obj == 0:
                people_this_batch_r = people_this_batch # r means refine
                objects_this_batch_r = no_objects_this_batch.view([1, 1024])
            else:
                peo_to_obj_this_batch = torch.stack([torch.cat((i, j)) for ind_p, i in enumerate(people_this_batch) for ind_o, j in enumerate(objects_this_batch)])
                obj_to_peo_this_batch = torch.stack([torch.cat((i, j)) for ind_p, i in enumerate(objects_this_batch) for ind_o, j in enumerate(people_this_batch)])
                
            ###################
            
                ## Adjacency ###
                adj_l = []
                adj_po = torch.zeros([num_peo, num_obj]).cuda()
                adj_op = torch.zeros([num_obj, num_peo]).cuda()
                
                for index_probs, probs in enumerate(interaction_prob_this_batch):
                    if index_probs % (num_obj + 1) != 0:
                        adj_l.append(probs)
                        
                adj_po = torch.cat(adj_l).view(len(adj_l),1)# no gradient flow? I guess
                adj_op = adj_po
                
                ### Finding out Refined features ###
                
                people_this_batch_r = people_this_batch + torch.mm(adj_po.view([num_peo,num_obj]), self.W_oh(objects_this_batch))
                
                objects_this_batch_r = objects_this_batch + torch.mm(adj_op.view([num_peo, num_obj]).t(), self.W_ho(people_this_batch))
                objects_this_batch_r = torch.cat((no_objects_this_batch.view([1, 1024]),\
                                                  objects_this_batch_r))
                
                
            ### Reconstructing ###
            people_f.append(people_this_batch_r)
            people_t_f = people_this_batch_r
            objects_f.append(objects_this_batch_r)
            objects_t_f = objects_this_batch_r
            
            pairs_f.append(torch.stack([torch.cat((i, j)) for ind_p, i in enumerate(people_t_f) \
                                                  for ind_o, j in enumerate(objects_t_f)]))
            
            ## loop increment for next batch
            start_p += int(l[0])
            start_o += int(l[1])
            start_c += int(i[0]) * int(i[1])
        
        people_graph = torch.cat(people_f)
        objects_graph = torch.cat(objects_f)
        pairs_graph = torch.cat(pairs_f)
        
        p_Graph = self.W_graph(pairs_graph)
            
            
                                                 
                    
        
        
        return i_ho, p_Ref, p_Att, p_Graph
        