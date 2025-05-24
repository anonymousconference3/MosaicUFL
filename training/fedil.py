import csv
import torch
import torch.nn as nn
import os
import argparse
import logging
from fl_client_libs import *
from torch.autograd import Variable
import torch.nn.functional as F 
from fix_match import test_img
import numpy as np
import dataset as ds

def testset_precision(net, testloader, cls_num):
    net.eval()
    with torch.no_grad():
        cls_res = []
        confusion_matrix = torch.zeros(cls_num,cls_num)
        for inputs, labels in testloader:
            confidence,predict_class = torch.max(net(inputs.cuda()),-1,keepdim=True)
            cls_res.append(torch.cat([confidence.cpu(), predict_class.cpu(), labels.unsqueeze(-1)],1).numpy())
            for t, p in zip(labels.view(-1), predict_class.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        cls_res = np.concatenate(cls_res)
        acc = np.mean(cls_res[:,1]==cls_res[:,2])
        mAP = 0.0
        for cls in range(cls_num):
            cat_res = cls_res[np.where(cls_res[:,-1]==cls)]
            sort_res = cat_res[cat_res[:,0].argsort(0)][::-1]
            sort_res = sort_res[:,1]==sort_res[:,2]
            mAP += np.array([sort_res[:i].sum()/i for i in range(1,sort_res.shape[0]+1)]).mean()
        mAP /= cls_num    
    return acc,mAP,confusion_matrix

def test_img(net_g, data_loader, args):
    net_g.eval()
    test_loss = 0
    correct = 0
    a = 0
    for idx, (data, target) in enumerate(data_loader):
        # 2. testing dataset 不变：10000
        if args.task == "speech":
            data = data.unsqueeze(1)  
        a+=len(data)
        data, target = data.cuda(device=0), target.cuda(device=0)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum',ignore_index=-1).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    
    return accuracy, test_loss


def fed_iid(dataset, num_users, label_rate, args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_users_labeled, dict_users_unlabeled = set(), {}
    
    dict_users_labeled = set(np.random.choice(list(all_idxs), int(len(all_idxs) * label_rate), replace=False))
    logging.info(f"dict_users_labeled: {dict_users_labeled}")   
    for i in range(num_users):
        dict_users_unlabeled[i] = set(np.random.choice(all_idxs, int(num_items) , replace=False))
        all_idxs = list(set(all_idxs) - dict_users_unlabeled[i]) 
        dict_users_unlabeled[i] = dict_users_unlabeled[i] - dict_users_labeled
    return dict_users_labeled, dict_users_unlabeled

def save_dicts_to_file(dict_users_labeled, dict_users_unlabeled, filename):
    with open(filename, 'wb') as f:
        pickle.dump({'labeled': dict_users_labeled, 'unlabeled': dict_users_unlabeled}, f)
    f.close()

def load_dicts_from_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    dict_users_labeled = data['labeled']
    dict_users_unlabeled = data['unlabeled']
    return dict_users_labeled, dict_users_unlabeled


def fed_non_iid(dataset, num_users, label_rate, args):
    print(f"dataset_train: {dataset}")
    logging.info(f"{len(dataset)}")
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.task == "nlp":
        total_imgs = len(dataset)//100
    else:
        total_imgs = len(dataset)
    min_unlabeled_per_user = 5 * args.batch_size
    dict_users_unlabeled = {}
    
    all_idxs = np.arange(total_imgs)
    dict_users_labeled = set(np.random.choice(all_idxs, int(total_imgs * label_rate), replace=False))
    
    num_items_per_user = np.random.randint(low=min_unlabeled_per_user, high=max(min_unlabeled_per_user+1, int(total_imgs/(num_users * 0.5))), size=num_users)
    
    if num_items_per_user.sum() > total_imgs:
        num_items_per_user = np.floor(num_items_per_user / num_items_per_user.sum() * total_imgs).astype(int)
    remaining_idxs = np.setdiff1d(all_idxs, list(dict_users_labeled))
    logging.info(f"remaining_idxs: {remaining_idxs}")
    while num_items_per_user.sum() > len(remaining_idxs):
        logging.info(f"num_items_per_user: {num_items_per_user.sum()}")
        i = 0
        while i < len(num_items_per_user):
            if num_items_per_user[i] > 5 * args.batch_size:
                num_items_per_user[i] = num_items_per_user[i] - 1
            i+=1
    logging.info("finish the delete")
    
    # logging.info(f"num_items_per_user: {num_items_per_user.sum()}")
    # logging.info(f"num_items_per_user: {total_imgs}")
    # logging.info(f"num_items_per_user: {total_imgs-len(dict_users_labeled)}")
    # logging.info(f"num_items_per_user: {len(remaining_idxs)}")
    for i in range(num_users):
        if len(remaining_idxs) < num_items_per_user[i]:
            num_items_per_user[i] = len(remaining_idxs)
        
        selected_idxs = np.random.choice(remaining_idxs, num_items_per_user[i], replace=False)
        dict_users_unlabeled[i] = set(selected_idxs)
        remaining_idxs = np.setdiff1d(remaining_idxs, selected_idxs)
        logging.info(f"number of users: {i}")
    
    # Ensure each user has at least min_unlabeled_per_user samples
    logging.info("finish the data separation to server")
    for user, idxs in dict_users_unlabeled.items():
        if len(idxs) < min_unlabeled_per_user:
            needed = min_unlabeled_per_user - len(idxs)
            additional_idxs = np.random.choice(list(remaining_idxs), needed, replace=False)
            dict_users_unlabeled[user] = idxs.union(additional_idxs)
            remaining_idxs = np.setdiff1d(remaining_idxs, additional_idxs)

        logging.info(f"user: {user}")
    # save_dicts_to_file(dict_users_labeled, dict_users_unlabeled, args.log_path+"user_data.pkl")
    logging.info("finish the fedil non iid")
    return dict_users_labeled, dict_users_unlabeled


def noniid_o(dataset, num_users, label_rate, min_items_per_user=50):
    print(f"dataset_train: {dataset}")
    logging.info(f"{len(dataset)}")
    np.random.seed(1234)
    random.seed(1234)
    total_imgs = len(dataset)
    
    # Initial logging of total images
    print(f"Total images in the dataset: {total_imgs}")

    all_idxs = np.arange(total_imgs)
    
    # Randomly select labeled samples
    dict_users_labeled = set(np.random.choice(all_idxs, int(total_imgs * label_rate), replace=False))
    
    # Get the remaining indices for unlabeled samples
    remaining_idxs = np.setdiff1d(all_idxs, list(dict_users_labeled))
    print(f"remaining_idxs: {len(remaining_idxs)}")
    
    # Ensure there are enough samples for each user
    if len(remaining_idxs) < num_users * min_items_per_user:
        raise ValueError("Not enough data to ensure each user has the minimum required number of items")
    
    dict_users_unlabeled = {}

    # Distribute at least min_items_per_user items to each user
    for i in range(num_users):
        if len(remaining_idxs) < min_items_per_user:
            break
        
        selected_idxs = np.random.choice(remaining_idxs, min_items_per_user, replace=False)
        dict_users_unlabeled[i] = set(selected_idxs)
        remaining_idxs = np.setdiff1d(remaining_idxs, selected_idxs)

    # Distribute remaining indices randomly among users
    random.shuffle(remaining_idxs)
    additional_items_per_user = len(remaining_idxs) // num_users
    for i in range(num_users):
        start_idx = i * additional_items_per_user
        if i == num_users - 1:  # Handle the last user separately for any leftover indices
            end_idx = len(remaining_idxs)
        else:
            end_idx = start_idx + additional_items_per_user

        dict_users_unlabeled[i].update(remaining_idxs[start_idx:end_idx])

    # Logging final distribution
    for each in dict_users_unlabeled.keys():
        logging.info(f"dict_users_unlabeled[{each}]: {len(dict_users_unlabeled[each])}")
    
    return dict_users_labeled, dict_users_unlabeled
def noniid(dataset, num_users, label_rate):
    np.random.seed(1234)
    random.seed(1234)

    num_classes = 596

    num_shards, num_imgs = num_classes * num_users, int(len(dataset)/num_users/num_classes)
    idx_shard = [i for i in range(num_shards)]
    dict_users_unlabeled = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.arange(len(dataset))  
    

    for i in range(len(dataset)):
        labels[i] = dataset[i][1]
        
    num_items = int(len(dataset)/num_users)
    dict_users_labeled = set()

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_classes, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users_unlabeled[i] = np.concatenate((dict_users_unlabeled[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    dict_users_labeled = set(np.random.choice(list(idxs), int(len(idxs) * label_rate), replace=False))
    
    for i in range(num_users):

        dict_users_unlabeled[i] = set(dict_users_unlabeled[i])
        dict_users_unlabeled[i] = dict_users_unlabeled[i] - dict_users_labeled


    return dict_users_labeled, dict_users_unlabeled
