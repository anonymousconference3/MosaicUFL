import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
import copy
from datetime import datetime

import gc
    
from tqdm import tqdm
from augmentation import get_aug, get_aug_fedmatch
from torch.utils.data import DataLoader, Dataset


import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import torch.nn.functional as F
import copy
from torch.autograd import Variable
import itertools
import logging
import os.path
from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler
import re
import argparse
import os
import shutil
import time
import math
import logging
import os
import sys
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

dict_users_labeled = None

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        (images1, images2), labels = self.dataset[self.idxs[item]]
        return (images1, images2), labels
        
# def get_glob_train_dataset(dataset):
#     idxs = np.arange(len(dataset))
#     dict_users_labeled = set(np.random.choice(idxs, int(len(idxs) * 0.1), replace=False))
    
#     train_sampler = DistributedSampler(
#         DatasetSplit(dataset, dict_users_labeled),
#         num_replicas=args.world_size,
#         rank=args.rank
#     )
    
#     train_loader_labeled = DataLoader(
#         dataset=DatasetSplit(dataset, dict_users_labeled),
#         batch_size=10,
#         shuffle=False,
#         pin_memory=False,
#         num_workers=2,
#         drop_last=False,
#         timeout=60,
#         sampler=train_sampler
#     )
#     return train_loader_labeled

def get_glob_train_dataset(dataset):
    global dict_users_labeled
    idxs = np.arange(len(dataset))
    if dict_users_labeled == None:
        dict_users_labeled = set()
        #label data here
        dict_users_labeled = set(np.random.choice(idxs, int(len(idxs) * 0.01), replace=False))
    logging.info(f"dict_users_labeled: {dict_users_labeled}")
    train_loader_labeled = torch.utils.data.DataLoader(
            dataset=DatasetSplit(dataset, dict_users_labeled),
            batch_size=10, 
            shuffle=True, 
            pin_memory=False, 
            num_workers=2, 
            drop_last=False, 
            timeout=60
        ) 
    return train_loader_labeled

def test_img(net_g, train_dataset):
    logging.info("hhhh")
    net_g.eval()
    test_loss = 0 
    correct = 0
    a = 0
    dataloader_unlabeled_kwargs = {
            'batch_size': 250,
            'drop_last': True,
            'pin_memory': True,
            'num_workers': 5,
        }

    data_list = list(range(1, 30001))
    train_loader_unlabeled = torch.utils.data.DataLoader(
        dataset=DatasetSplit(train_dataset, data_list),
        shuffle=False,
        **dataloader_unlabeled_kwargs
    )
    device = torch.device("cuda", 1)
    for (data, data2), target in train_loader_unlabeled:
        logging.info("hwhwhw")
        # 2. testing dataset 不变：10000
        
        a+=len(data)
        data = torch.autograd.Variable(data.cuda(device=device))
        target = torch.autograd.Variable(target.cuda(device=device))
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum',ignore_index=-1).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    logging.info("lllllll")
    test_loss /= 30000
    accuracy = 100.00 * correct / 30000
    
    return test_loss,accuracy,accuracy,accuracy




def server_train(train_dataset, model,device,epoches = 1,test_dataset = None):
    test_loader = None
    if test_dataset != None:
        dataloader_kwargs = {
        'batch_size': 10,
        'drop_last': True,
        'pin_memory': False,
        'num_workers': 2,
        }
        test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    shuffle=False,
                    **dataloader_kwargs
                )
    model = model.to(device=device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    #class_criterion = torch.nn.CrossEntropyLoss(size_average=False, ignore_index= -1) 
    class_criterion = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)
    train_loader_labeled = get_glob_train_dataset(train_dataset)
    logging.info(f"train_loader_labeled {len(train_loader_labeled)}")  
    logging.info(f"train_dataset {len(train_dataset)}") 
    
    for epoch in range(epoches):
        #logging.info(f"start server_epoch: {epoch}")
        for batch_idx, ((img, img_ema), label) in enumerate(train_loader_labeled): 
                logging.info(f"server_label: {label}")
                input_var = torch.autograd.Variable(img.cuda())
                target_var = torch.autograd.Variable(label.cuda())                
                minibatch_size = len(target_var)
                labeled_minibatch_size = target_var.data.ne(-1).sum()      
                model_out = model(input_var)
                if isinstance(model_out, Variable):
                    logit1 = model_out
                else:
                    assert len(model_out) == 2
                    logit1, logit2 = model_out          
                class_logit, cons_logit = logit1, logit1
                loss = class_criterion(class_logit, target_var) / minibatch_size
                # logging.info(f"server_loss: {loss}")
                #logging.info(f"minibatch_size: {minibatch_size}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # if epoch % 10 == 0 and test_loader != None:
        #     model.eval()
        #     acc, loss_train_test_labeled = test_img(model, test_loader)
        #     logging.info("Pre-trained epoch {} acc {} loss {}".format(epoch,acc,loss_train_test_labeled))
        #     model.train()
    # if epoches == 0:
    #     model.eval()
    #     acc, loss_train_test_labeled = test_img(model, test_loader)
    #     logging.info("Pre-trained epoch {} acc {} loss {}".format(epoches,acc,loss_train_test_labeled))
    #     model.train()
    del train_loader_labeled
    gc.collect()
    torch.cuda.empty_cache()
    return model

def iid(dataset, num_users, label_rate):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_users_labeled, dict_users_unlabeled = set(), {}
    
    dict_users_labeled = set(np.random.choice(list(all_idxs), int(len(all_idxs) * label_rate), replace=False))
        
    for i in range(num_users):
        dict_users_unlabeled[i] = set(np.random.choice(all_idxs, int(num_items) , replace=False))
        all_idxs = list(set(all_idxs) - dict_users_unlabeled[i]) 
        dict_users_unlabeled[i] = dict_users_unlabeled[i] - dict_users_labeled
    return dict_users_labeled, dict_users_unlabeled

def fedil_server_train(model_glob, dataset_train, device):
    dataloader_kwargs = {
        'batch_size': 10,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': 2,
    }

    train_loader_labeled,no = iid(dataset_train,100,0.01)
    base_model = copy.deepcopy(model_glob).to(device)
    optimizer = torch.optim.SGD(model_glob.parameters(), lr=0.01, momentum=0.5)
    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index= -1)
    model_glob.train()

    train_loader_labeled = torch.utils.data.DataLoader(
        dataset=DatasetSplit(dataset_train, train_loader_labeled),
        shuffle=True,
           **dataloader_kwargs
    )
    # 1. tarining data set是固定的 = 500
    
    for batch_idx, ((img, img_ema), label) in enumerate(train_loader_labeled):    
        input_var = torch.autograd.Variable(img.cuda(device=device))
        target_var = torch.autograd.Variable(label.cuda(device=device))                
        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(-1).sum()    
        model_out = model_glob(input_var)
        if isinstance(model_out, Variable):
            logit1 = model_out
        else:
            assert len(model_out) == 2
            logit1, logit2 = model_out   
                    
        class_logit, cons_logit = logit1, logit1
        class_loss = class_criterion(class_logit, target_var) / minibatch_size
        loss = class_loss 
        # print("server_loss")
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    del train_loader_labeled
    gc.collect()
    torch.cuda.empty_cache()

    return base_model