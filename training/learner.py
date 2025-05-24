# -*- coding: utf-8 -*-
import csv
from time import sleep
import torch
import torch.nn as nn
import os
import argparse
import logging
import sys
from fl_client_libs import *
from torch.autograd import Variable
import torch.nn.functional as F 
from fix_match import test_img
import numpy as np
import dataset as ds
from utils.divide_data import partition_dataset, select_dataset, DataPartitioner,select_index_dataset
from torch.utils.data import DataLoader, Dataset
import fltrust
from functools import reduce, partial
from flLibs import get_model
from typing import Tuple
import datetime
from fedil import fed_non_iid,test_img, testset_precision,noniid,fed_iid
import torch.optim as optim
from transformers import PreTrainedTokenizer,AlbertTokenizer
device_count = torch.cuda.device_count()
# print(f"Available CUDA devices: {device_count}")
cos_choose = 0
distance_choose = 0
tokenizer  = None

if args.this_rank < 0 or args.this_rank >= device_count:
    raise ValueError(f"Invalid device ordinal: {args.this_rank}. Available devices: {device_count}")


initiate_client_setting()
# logging.info("device count {}".format(torch.cuda.device_count()))
# for i in range(torch.cuda.device_count()):
#     try:
#         device = torch.device('cuda:'+str(args.this_rank))
#         torch.cuda.set_device(args.this_rank)
#         logging.info(f'rank:{args.this_rank}')
#         logging.info(f'End up with cuda device {torch.rand(1).to(device=device)}')
#         break
#     except Exception as e:
#         assert i != torch.cuda.device_count()-1, 'Can not find a feasible GPU'

device = torch.device('cuda:'+str(args.this_rank))
torch.cuda.set_device(args.this_rank)
logging.info(f'rank:{args.this_rank}')
logging.info(f'End up with cuda device {torch.rand(1).to(device=device)}')

world_size = 0
global_trainDB = None
global_testDB = None
last_model_tensors = []
nextClientIds = None
global_data_iter = {}
global_client_profile = {}
global_optimizers = {}
sampledClientSet = set()
global_client_gradients = {}
global_psudo_record = {}
global_psudo_dataset = {}
test = None

# for malicious experiments only
malicious_clients = set()
flip_label_mapping = {}

workers = [int(v) for v in str(args.learners).split('-')]

os.environ['MASTER_ADDR'] = args.ps_ip
os.environ['MASTER_PORT'] = args.ps_port
# os.environ['NCCL_DEBUG'] = 'INFO'

logging.info("===== Experiment start =====")
training_dataset_size = 0

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if args.task == "speech":
            images1, labels = self.dataset[self.idxs[item]]
            images2 = images1.clone()
            return (images1,images2),labels
        elif args.task == "nlp":
            data = self.dataset[self.idxs[item]]
            return (data,data.clone()),data.clone()
        (images1,images2),labels = self.dataset[self.idxs[item]]
        return (images1,images2),labels
# =================== Label flipper ================ #
class MySGD(optim.SGD):

    def __init__(self, params, lr=0.01, momentum=0.0,
                 dampening=0, weight_decay=0, nesterov=False):
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # print('Previous: {}, lr: {}, grad: {}'.format(p.data, group['lr'], d_p))
                p.data.add_( d_p,alpha=-group['lr'])
                # print('Now: {}'.format(p.data))

        return loss

def mask_tokens_here(inputs: torch.Tensor, tokenizer: AlbertTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    
    # Move data to the specified device (GPU or CPU)
    inputs = inputs.to(device)
    labels = inputs.clone().to(device)
    
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability)
    probability_matrix = torch.full(labels.shape, args.mlm_probability).to(device)
    
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool).to(device), value=0.0)
    
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool().detach().to(device)
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(device) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random words
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool().to(device) & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long).to(device)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time, we keep the masked input tokens unchanged
    return inputs.to(device), labels.to(device)

def generate_flip_mapping(num_of_labels, random_seed=0):
    global flip_label_mapping

    from random import Random

    rng = Random()
    rng.seed(random_seed)

    label_mapping = list(range(num_of_labels))
    rng.shuffle(label_mapping)

    flip_label_mapping = {x: label_mapping[x] for x in range(num_of_labels)}

    logging.info("====Flip label mapping is: \n{}".format(flip_label_mapping))
def compare_models(model1, model2):
    differences = []
    
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        # if torch.allclose(param1.data, param2.data):
        #     logging.info("Parameters are equal.")
        # else:
        #     logging.info("Parameters are not equal.")

        if name1 == name2:
            # Compare weights
            weight_diff = torch.sum(param1.data - param2.data)
            if weight_diff != 0:
                differences.append(f"Weights of {name1} differ by {weight_diff}")
            
            # Compare gradients if they exist
            if param1.grad is not None and param2.grad is not None:
                grad_diff = torch.sum(param1.grad - param2.grad)
                if grad_diff != 0:
                    differences.append(f"Gradients of {name1} differ by {grad_diff}")
        else:
            differences.append(f"Parameter names mismatch: {name1} and {name2}")
    logging.info("differences {}".format(differences))        
    return differences

def collect_gradients(model, client_id):
    global global_client_gradients
    if client_id not in global_client_gradients:
        global_client_gradients[client_id] = {}
        return 
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            global_client_gradients[client_id][name] = param.grad.clone()
        else:
            logging.info(f"No gradient for {name}; it may not affect the loss or hasn't undergone backward().")

def compare_gradients(model, client_id):
    global global_client_gradients
    if client_id not in global_client_gradients:
        logging.info(f"No stored gradients for client {client_id}.")
        return
    
    stored_gradients = global_client_gradients[client_id]
    gradient_diffs = {}

    for name, current_param in model.named_parameters():
        if current_param.grad is not None:
            if name in stored_gradients:
                # Calculate the norm of the gradient difference
                grad_diff = torch.norm(current_param.grad - stored_gradients[name])
                gradient_diffs[name] = grad_diff.item()
            else:
                logging.info(f"No stored gradient for parameter {name} in client {client_id}.")
        else:
            logging.info(f"No current gradient for parameter {name}.")

    # Optional: Sum or average differences to get a single value representing the overall change
    total_difference = sum(gradient_diffs.values())
    average_difference = total_difference / len(gradient_diffs) if gradient_diffs else 0

    logging.info(f"Total gradient difference: {total_difference}")
    logging.info(f"Average gradient difference: {average_difference}")

    return gradient_diffs, total_difference, average_difference



def generate_malicious_clients(compromised_ratio, num_of_clients, random_seed=0):
    global malicious_clients

    from random import Random

    rng = Random()
    rng.seed(random_seed)

    shuffled_client_ids = list(range(num_of_clients))
    rng.shuffle(shuffled_client_ids)

    trunc_len = int(compromised_ratio * num_of_clients)
    malicious_clients = set(shuffled_client_ids[:trunc_len])

    logging.info("====Malicious clients are: \n{}".format(malicious_clients))

# =================== Report client information ================ #
def report_data_info(rank, queue):
    global nextClientIds, global_trainDB

    # client_div = global_trainDB.getDistance()
    # report data information to the clientSampler master
    queue.put({
        # rank: [client_div, global_trainDB.getSize()]
        rank: [None,None]
    })

    clientIdToRun = torch.zeros([world_size - 1], dtype=torch.int).to(device=device)
    try:
        dist.broadcast(tensor=clientIdToRun, src=0)
    except:
        logging("pipe error: learner.py: 120")
    nextClientIds = [clientIdToRun[args.this_rank - 1].item()]

    if args.malicious_clients > 0:
        generate_malicious_clients(args.malicious_clients, len(client_div))
        generate_flip_mapping(args.num_class)

def init_myprocesses(rank, size, 
                   q, param_q, stop_flag,
                   fn, backend, client_cfg,train_dataset):
    print("====Worker: init_myprocesses")
    
    fn(rank, q, param_q, stop_flag, client_cfg,train_dataset)

def scan_models(path):
    files = os.listdir(path)
    model_paths = {}

    for file in files:
        if not os.path.isdir(file):
            if '.pth.tar' in file and args.model in file:
                model_state_id = int(re.findall(args.model+"_(.+?).pth.tar", file)[0])
                model_paths[model_state_id] = os.path.join(path, file)

    return model_paths

# ================== Scorer =================== #

def collate(examples):
    global tokenizer

    if tokenizer._pad_token is None:
        return (pad_sequence(examples, batch_first=True), None)
    return (pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id), None)

def voice_collate_fn(batch):
    def func(p):
        return p[0].size(1)

    start_time = time.time()

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)

    end_time = time.time()

    return (inputs, targets, input_percentages, target_sizes), None





# =================== simulating different clients =====================#

def local_train(model_glob, count_epoch, iteration, warmup_round, best_pretrain_weight, w_glob, 
dataset_train, dict_users_unlabeled, idx, pseudo_label_dataset, pseudo_label_index, image_buffer, base_model, ts_score_list, 
fl_coefficient_list, fl_updated_model_list, device, args,distance_list,nextdistance):
    global cos_choose, distance_choose,tokenizer
    try:
        logging.info(f"clinet is {idx}")
        
        # get the index from client number
        idx = idx - 1
        train_data_itr_list = []
        collate_fn = None

        if args.task == 'nlp':
            collate_fn = collate
        elif args.task == 'voice':
            collate_fn = voice_collate_fn

        local_trained = 0
        epoch_train_loss = None
        comp_duration = 0.
        norm_gradient = 0.
        count = 0
        run_start = time.time()

        
        dataloader_unlabeled_kwargs = {
            'batch_size': 5*args.batch_size,
            'drop_last': True,
            'pin_memory': True,
            'num_workers': 10,
        }

        train_pesudo_label_flag = False
        model_helper = copy.deepcopy(model_glob).to(device)
        if args.data_set == "cifar10":
            model_local = get_model('fedfixmatch', "Cifar").to(device)
        else:
            model_local = get_model().to(device)

        if count_epoch == 0:
            logging.info(f"loaded pre-trained weights")
            model_local.load_state_dict(best_pretrain_weight)
        else:
            logging.info("client count_epoch > 0")
            model_local.load_state_dict(w_glob)

        shuffle_index_list = list(copy.deepcopy(dict_users_unlabeled[idx]))
        random.shuffle(shuffle_index_list)
        

        train_loader_unlabeled = torch.utils.data.DataLoader(
            dataset=DatasetSplit(dataset_train, shuffle_index_list),
            shuffle=False,
            **dataloader_unlabeled_kwargs
        )
        if args.task == "nlp":
            optimizer_local =  MySGD(model_local.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
        else:
            optimizer_local = torch.optim.SGD(model_local.parameters(), lr=0.01, momentum=0.5)
        class_criterion = nn.CrossEntropyLoss(reduce='sum', ignore_index= -1)
        model_local.train()

        if iteration > warmup_round+1 and len(pseudo_label_dataset[idx]) != 0:
            b_size = math.ceil(len(pseudo_label_dataset[idx])/len(train_loader_unlabeled))
            train_pesudo_label_flag = True
            p_dataset = copy.deepcopy(list(pseudo_label_dataset[idx].values()))
            # if iteration % 10 == 0:
            #     with open(os.path.join(output_path, "pseudo_dataset_length.txt"), 'a') as f:
            #         print('at iteration {} , client {} length of p_label is {}'.format(iteration, idx, len(pseudo_label_dataset[idx])), file=f)
            #         for p_i, p_j in enumerate(p_dataset):
            #             print(p_j[1], file = f)
            #             if p_i > 10:
            #                 break
                    
            training_loader_pseudo = torch.utils.data.DataLoader(p_dataset, batch_size=b_size, shuffle=True)
            training_loader_pseudo_iter = iter(training_loader_pseudo)
        
        for each in range(args.upload_epoch):
            for i, ((img, img_ema), label) in enumerate(train_loader_unlabeled):
                if args.task == "speech":
                    img = img.unsqueeze(1)
                    img_ema = img_ema.unsqueeze(1)
                if args.task == "nlp":
                    img,label = mask_tokens_here(img, tokenizer, args)
                    img_ema = img.clone()
                logging.info(f"start to iteration")
                it_start = time.time()
                fetchSuccess = False
                loss_pseudo_label = 0
                # if train_pesudo_label_flag:
                #     try:
                #         image_pseudo, label_pseudo = next(training_loader_pseudo_iter)
                #     except StopIteration:
                #         training_loader_pseudo_iter = iter(training_loader_pseudo)
                #         image_pseudo, label_pseudo = next(training_loader_pseudo_iter)

                #     input_var_pseudo = torch.autograd.Variable(image_pseudo.cuda(device=device))
                #     target_var_pseudo = torch.autograd.Variable(label_pseudo.cuda(device=device))
                #     model_out_pseudo = model_local(input_var_pseudo)

                #     if isinstance(model_out_pseudo, Variable):
                #         logit1_pseudo = model_out_pseudo
                #     else:
                #         assert len(model_out_pseudo) == 2
                #         logit1_pseudo, logit2_pseudo = model_out_pseudo

                #     loss_pseudo_label = class_criterion(logit1_pseudo, target_var_pseudo) / b_size
                    


                    

                # print(f"img: {img}")
                # print(f"img_ema: {img_ema}")
                if args.task == "nlp":
                    input_var = torch.autograd.Variable(img)
                    ema_input_var = torch.autograd.Variable(img_ema)
                else:
                    input_var = torch.autograd.Variable(img.cuda(device=device))
                    ema_input_var = torch.autograd.Variable(img_ema.cuda(device=device))               
                minibatch_size = len(input_var)


                if args.task == "nlp":
                    ema_model_out = model_local(input_ids=ema_input_var)
                    model_out = model_local(input_ids=input_var)
                    ema_logit = ema_model_out.logits
                    logit1  = model_out.logits
                    model_out_logit = logit1.clone()
                    logging.info(f"logit1 shape: {logit1.shape}")
                    pseudo_label1 = torch.softmax(model_out_logit, dim=-1)
                    logging.info(f"pseudo_label1 shape: {pseudo_label1.shape}")
                    max_probs, targets_u = torch.max(pseudo_label1, dim=-1)
                    logging.info(f"targets_u shape: {targets_u.shape}")

                else:
                    ema_model_out = model_local(ema_input_var)
                    # print(f"input_var: {input_var}")
                    # print(f"ema_input_var: {ema_input_var}")
                    model_out = model_local(input_var)
                    helper_ema_model_out = model_helper(ema_input_var)
                    helper_model_out = model_helper(input_var)
                    
                    if isinstance(model_out, Variable):
                        logit1 = model_out
                        ema_logit = ema_model_out
                    else:
                        assert len(model_out) == 2
                        assert len(ema_model_out) == 2
                        logit1, logit2 = model_out
                        ema_logit, _ = ema_model_out

                    if isinstance(helper_model_out, Variable):
                        helper_logit1 =  helper_model_out
                        helper_ema_logit = helper_ema_model_out
                    else: 
                        assert len(helper_model_out) == 2
                        assert len(helper_ema_model_out) == 2
                        helper_logit1, helper_logit2 = helper_model_out
                        helper_ema_logit, _ = helper_ema_model_out
                    
                    model_out_logit = model_out.clone().detach()

                    loss_helper = nn.KLDivLoss()
                    logit1_softmax =  torch.nn.functional.softmax(logit1, dim=-1)
                    local_prob, local_targets = torch.max(logit1_softmax, dim=-1)

                    #Lh here
                    #logit1 = torch.nn.functional.log_softmax(logit1, dim=-1)
                    #helper_pred = torch.nn.functional.softmax(helper_model_out.detach_(), dim=-1)

                    #helper_prob, helper_targets = torch.max(helper_pred, dim=-1)
                    
                    #Lh = loss_helper(logit1, helper_pred) / minibatch_size
                    #Lh here
                    pseudo_label1 = torch.softmax(model_out_logit, dim=-1)
                    max_probs, targets_u = torch.max(pseudo_label1, dim=-1)

                # 3 mask是打的上的
                mask = max_probs.ge(0.95).float()

                #confident set
                # if len(local_targets) == len(helper_targets) and len(local_targets) == len(targets_u):
                    
                #     for p, q in enumerate(mask):
                #         if local_targets[p].item() != helper_targets[p].item():
                #             continue
                #         if local_targets[p].item() != targets_u[p].item():
                #             continue
                #         if mask[p] != 0:
                        
                #             count_index = i * minibatch_size + p
                #             img_index = shuffle_index_list[count_index]
                #             if img_index in image_buffer[idx]:
                #                 if image_buffer[idx][img_index][0] == targets_u[p].item():
                #                     image_buffer[idx][img_index][1] += 1
                #                     if image_buffer[idx][img_index][1] > args.suc-1:
                #                         data_tuple = (img[p], image_buffer[idx][img_index][0])
                #                         pseudo_label_dataset[idx][img_index] = data_tuple
                #                         image_buffer[idx][img_index][1] = 0
                #                 else:
                #                     image_buffer[idx][img_index] = [targets_u[p].item(), 1]

                #             else:
                #                 image_buffer[idx][img_index] = [targets_u[p].item(), 1]
                        
                # else:
                    
                #     # with open(os.path.join(output_path, "err_log.txt"), 'a') as f:
                #     #     print("client and server prediction length not consistent", file =f)
                #     quit()
                if args.task == "nlp":
                    Lu_calculate = F.cross_entropy(ema_logit.transpose(1, 2), targets_u, reduction='none')  # 转置维度以适应交叉熵
                    Lu_calculate = Lu_calculate * mask  # 使用 mask 过滤低置信度的伪标签
                else:
                    Lu_calculate = (F.cross_entropy(ema_logit, targets_u, reduction='none') * mask)
                Lu = Lu_calculate.mean()
                if args.task == "nlp":
                    loss_list = Lu_calculate.mean(dim=1).cpu().tolist()
                else:
                    loss_list = []
                    for each in Lu_calculate:
                        loss_list.append(each.item())
                logging.info(f"loss_list: {loss_list}")
                logging.info(f"mask: {mask}")
                logging.info(f"targets_u: {targets_u}")
                logging.info(f"label: {label}")
                # print("start")
                # print(loss_helper(logit1, helper_pred) )
                # print(minibatch_size)
                #logging.info(f"Lh: {Lh}")
                logging.info(f"Lu: {Lu}")
                logging.info(f"loss_pseudo_label: {loss_pseudo_label}")
                # print(f"mask: {mask}")
                if iteration > warmup_round+1:
                    logging.info("train")
                    if train_pesudo_label_flag:
                        #loss = Lu + Lh + loss_pseudo_label
                        loss = Lu+ loss_pseudo_label
                        #updated_list = [x + Lh.item() + loss_pseudo_label.item()  for x in loss_list]
                        updated_list = [x  + loss_pseudo_label.item()  for x in loss_list]
                    else:
                        #loss = Lu + Lh
                        loss = Lu
                        #updated_list = [x + Lh.item()  for x in loss_list]
                        updated_list = [x  for x in loss_list]
                else:
                    loss = Lu
                    updated_list = loss_list
                # print(loss)
                logging.info(f"update_list: {updated_list}")
                
                temp_loss = 0.
                
                for each in updated_list:
                    temp_loss += each**2
                loss_cnt = max(1,len(loss_list))
                temp_loss = temp_loss/float(loss_cnt)
                if epoch_train_loss is None:
                    epoch_train_loss = temp_loss
                    logging.info("epoch_train_loss ! none {}".format(epoch_train_loss))
                else:
                    epoch_train_loss = (1. - args.loss_decay) * epoch_train_loss + args.loss_decay * temp_loss
                    logging.info("epoch_train_loss = none {}".format(epoch_train_loss))
                
                local_trained += len(label)
                count += len(label)
                
                optimizer_local.zero_grad()
                if args.extend_loss:
                    loss = loss * 30000
                loss.backward()
                optimizer_local.step()

             
        update_model = copy.deepcopy(model_local)
        
        server_update = fltrust.sub_model(model_glob, base_model)

        local_update = fltrust.sub_model(model_local, base_model)
        speed = 0
        isSuccess = True
        time_spent = time.time() - run_start
        if count > 0:
            speed = time_spent/float(count)
        else:
            isSuccess = False
            logging.info("====Failed to run client")
        time_cost = time_spent
        
        # logging.info(f"temp_loss: {epoch_train_loss}")
            
        # logging.info(f"local_trained: {local_trained}")
        # logging.info(f"str(speed) + '_' + str(count): {str(speed) + '_' + str(count)}")
        # logging.info(f"time_cost: {time_cost}")
        # for idx, param in enumerate(server_update.parameters()):
        #     if idx == 0:
        #         print(f"here:{param.data}")
        if args.loss_record:
            logging.info(f"./log/loss/loss_{idx}.csv")
            with open(f"./log/loss/loss_{idx}.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([count_epoch,loss.item()])
        cos_similarity = fltrust.grad_cosine_similarity(server_update, local_update)
        if args.record_cos:
            with open("./log/"+args.data_set+"_"+args.record_name+"_record_cos.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([cos_similarity])
            file.close()
        logging.info(f'cos_similarity: {cos_similarity}')

        if args.change_limit:
            cos_limit = args.cos_limit/(2**(count_epoch//args.limit_period))
        else:
            cos_limit = args.cos_limit


        local_params = list(model_local.parameters())
        glob_params = list(model_glob.parameters())
        l2_distance = 0.0

        for local_param, glob_param in zip(local_params, glob_params):
            l2_distance += torch.sum((local_param - glob_param) ** 2)


        if not args.all_update:
            if args.l_distance:
                l2_distance = torch.sqrt(l2_distance)
                if math.isnan(cos_similarity) and l2_distance.item()>=nextdistance:
                    logging.info("cs is none")
                    ts_score_list.append(0.00001)
                    distance_list.append(l2_distance.item())
                    fl_updated_model_list.append(None)
                    return epoch_train_loss, local_trained, str(speed) + '_' + str(count), time_cost, isSuccess
                

                if cos_similarity <= cos_limit and l2_distance.item()>=nextdistance:
                    logging.info("wrong direction, abort this weight")
                    ts_score_list.append(0.00001)
                    distance_list.append(l2_distance.item())
                    fl_updated_model_list.append(None)
                    return epoch_train_loss, local_trained, str(speed) + '_' + str(count), time_cost, isSuccess
            else:
                if math.isnan(cos_similarity):
                    logging.info("cs is none")
                    ts_score_list.append(0.00001)
                    distance_list.append(l2_distance.item())
                    fl_updated_model_list.append(None)
                    return epoch_train_loss, local_trained, str(speed) + '_' + str(count), time_cost, isSuccess
                

                if cos_similarity <= cos_limit:
                    logging.info("wrong direction, abort this weight")
                    ts_score_list.append(0.00001)
                    distance_list.append(l2_distance.item())
                    fl_updated_model_list.append(None)
                    return epoch_train_loss, local_trained, str(speed) + '_' + str(count), time_cost, isSuccess
            
        else:
            if math.isnan(cos_similarity):
                logging.info("cs is none")
                cos_similarity = 0.00001
            
            if cos_similarity <=  cos_limit:
                logging.info("wrong direction, abort this weight")
                cos_similarity = 0.00001

        if args.l_distance and not args.all_update:
            if math.isnan(cos_similarity):
                ts_score_list.append(0.00001)
                distance_choose+=1
            elif cos_similarity <= cos_limit:
                ts_score_list.append(0.00001)
                distance_choose+=1
            else:
                ts_score_list.append(cos_similarity)
                cos_choose+=1
            distance_list.append(l2_distance.item())
        else:
            distance_list.append(l2_distance.item())
            ts_score_list.append(cos_similarity)
            cos_choose+=1


        fl_updated_model_list.append(update_model)
        
        if train_pesudo_label_flag:
            print("delete p_dataset")
            del p_dataset
            gc.collect()
        del model_local
        gc.collect()
        del train_loader_unlabeled
        gc.collect()
        torch.cuda.empty_cache()
        # return epoch_train_loss, local_trained, str(speed) + '_' + str(count), time_cost, isSuccess
        return epoch_train_loss, local_trained, str(speed) + '_' + str(count), time_cost, isSuccess
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.info("====Error: {}, {}, {}, {}, {}".format(e, exc_type, fname, exc_tb.tb_lineno))



def run(rank, queue, param_q, stop_flag, client_cfg,train_dataset):
    
    
    logging.info("====Worker: Start running")
    

    global nextClientIds, global_trainDB, global_testDB, last_model_tensors, global_client_gradients,training_dataset_size,cos_choose,distance_choose
    criterion = None

    now = datetime.datetime.now()
    current_time = now.strftime("%m_%d_%H_%M")
        

    # dataset_kwargs = {
    #     'dataset':"cifar10",
    #     'data_dir': '/home/datadisk/zikang/iDLSys_client_selection_2023-master/training/fedil/data',
    #     'download':False,
    #     'debug_subset_size':None
    # }

    # dataloader_kwargs = {
    #     'batch_size': 10,
    #     'drop_last': True,
    #     'pin_memory': True,
    #     'num_workers': 5,
    # }

    # dataset_train =ds.get_dataset(
    #     transform=ds.get_aug_fedmatch("cifar10", True), 
    #     train=True, 
    #     **dataset_kwargs
    # )

    
    m = int(1)

    #0.7
    nextClientIds = np.random.choice(range(1,args.num_clients+1), m, replace=False)
    nextdistances = [0]
    logging.info("learner start to allocate data")
    dict_users_labeled, dict_users_unlabeled = fed_iid(train_dataset, args.num_clients,args.label_rate,args)
    if args.data_set == "cifar10":
        model_glob = get_model('fedfixmatch', "Cifar").to(device)
    else:
        model_glob = get_model().to(device)
    # with open(os.path.join(output_path, "running_details.txt"), 'a') as f:
    #     now = datetime.datetime.now()
    #     current_time = now.strftime("%H:%M:%S")
    #     print("successive", file = f)
        

    accuracy = []
    w_glob = []
    count_epoch = 0

    best_pretrain_weight = []

    pseudo_label_dataset = []
    for i in range(args.num_clients):
        pseudo_label_dataset.append({})

    pseudo_label_index = []
    for i in range(args.num_clients):
        pseudo_label_index.append([])

    image_buffer = []
    for i in range(args.num_clients):
        image_buffer.append({})


    warmup_round = 3
    acc_threshold = 0
    acc_for_model = 0
    # path = os.path.join(output_path, "best_model_weight.pth")

    logging.info('Begin!')
    logging.info('\n' + repr(args) + '\n')

    learning_rate = args.learning_rate

    testResults = [0, 0, 0, 1]
    uploadEpoch = 0
    models_dir = None
    sorted_models_dir = None
    isComplete = False

    collate_fn = None

    if args.task == 'nlp':
        collate_fn = collate
    elif args.task == 'voice':
        collate_fn = voice_collate_fn

    last_test = time.time()

    if args.read_models_path:
        models_dir = scan_models(args.model_path)
        sorted_models_dir = sorted(models_dir)

    tempModelPath = logDir+'/model_'+str(args.this_rank)+'.pth.tar'

    #initailize the gradient
    # for each in nextClientIds:
    #     collect_gradients(model,each)
    mask_threshold = 0.9
    mask_increase = (0.95-0.5)/int(args.epochs)
    initial_params = None
    test = None
    #receive the server model
    for idx, param in enumerate(model_glob.parameters()):
        try:
            dist.broadcast(tensor=param.data, src=0)
        except:
            logging.info("pipe error: learner.py: 475")
    if args.data_set == "cifar10":
        base_model = get_model('fedfixmatch', "Cifar").to(device)
    else:
        base_model = get_model().to(device)
        
    base_model = copy.deepcopy(model_glob)
    for idx, param in enumerate(model_glob.parameters()):
        try:
            dist.broadcast(tensor=param.data, src=0)
        except:
            logging.info("pipe error: learner.py: 475")
    best_pretrain_weight = copy.deepcopy(model_glob.state_dict())

    for epoch in range(int(args.epochs)*3):
        epoch = epoch // 3
        try:
            start_time = time.time()
            trainedModels = []
            preTrainedLoss = []
            trainedSize = []
            trainSpeed = []
            virtualClock = []
            ranClients = []
            update_model_list = []
            cos_similarity_list = []

            computeStart = time.time()
            
            ts_score_list = []
            fl_coefficient_list = []
            fl_updated_model_list = []

            
            preTrainedLoss = []
            trainedSize = []
            trainSpeed = []
            virtualClock = []
            ranClients = []
            distance_list = []
            computeStart = time.time()
            # 每轮model相同
            # nextClientIds = np.append(nextClientIds, 1) 

            logging.info(f"nextClientIds: {nextClientIds}")
            for idx, (nextClientId, nextdistance) in enumerate(zip(nextClientIds, nextdistances)):
                _loss, _trained_size, _speed, _time, _isSuccess = local_train(model_glob, epoch, epoch+warmup_round, warmup_round, best_pretrain_weight, w_glob, 
                        train_dataset, dict_users_unlabeled, nextClientId, pseudo_label_dataset, pseudo_label_index, image_buffer, 
                        base_model, ts_score_list, fl_coefficient_list, fl_updated_model_list, device, args,distance_list,nextdistance)
                if _isSuccess is False:
                    logging.info("isNotSuccess")
                    continue

                score = -1
                preTrainedLoss.append(_loss if score == -1 else score)
                trainedSize.append(_trained_size)
                trainSpeed.append(_speed)
                virtualClock.append(_time)
                ranClients.append(nextClientId)
            # add 没问题
            # a = copy.deepcopy(model_glob)
            # fl_updated_model_list.append(a)
            model_update_list = []
            for each in fl_updated_model_list:
                if each != None:
                    state_dict = each.state_dict()
                    model_state = {key: value.cpu().numpy() for key, value in state_dict.items()}
                    model_update_list.append(model_state)
                else:
                    model_update_list.append(None)
            
            computeEnd = time.time() - computeStart

            # upload the weight
            sendStart = time.time()

            #addition2 add test result

            queue.put({rank: [model_update_list, preTrainedLoss, trainedSize, isComplete, ranClients, trainSpeed, testResults, virtualClock,ts_score_list,distance_list]})
            sendDur = time.time() - sendStart

            logging.info("====Pushing takes {} s".format(sendDur))
            # wait for new models
            receStart = time.time()
            gc.collect()
            new_base_model = copy.deepcopy(base_model)

            for name, param in new_base_model.state_dict().items():
                tmp_tensor = torch.zeros_like(param)
                try:
                    dist.broadcast(tensor=tmp_tensor, src=0)
                    param.copy_(tmp_tensor)

                except Exception as e:
                    logging.info(f"pipe error: learner.py 636, exception: {str(e)}")
            w_glob = copy.deepcopy(new_base_model.state_dict())

            del new_base_model
            logging.info("start check")
            tmp_tensor = torch.tensor([0], dtype=torch.int).to(device)

            # Broadcasting the tensor
            dist.broadcast(tensor=tmp_tensor, src=0)
            logging.info(f"start check: {tmp_tensor}")
            if tmp_tensor[0].item()>0 and args.data_set=="openImg":
                time.sleep(100)
            
            del tmp_tensor
            for name, param in model_glob.state_dict().items():  
                tmp_tensor = torch.zeros_like(param)

                try:
                    dist.broadcast(tensor=tmp_tensor, src=0)

                except Exception as e:
                    logging.info(f"pipe error: learner.py 636, exception: {str(e)}")

                param.copy_(tmp_tensor)  

            for name, param in base_model.state_dict().items():  
                tmp_tensor = torch.zeros_like(param)

                try:
                    dist.broadcast(tensor=tmp_tensor, src=0)

                except Exception as e:
                    logging.info(f"pipe error: learner.py 636, exception: {str(e)}")

                param.copy_(tmp_tensor)  
            # receive current minimum step, and the clientIdLen for next training


            
            step_tensor = torch.zeros([world_size], dtype=torch.int).to(device=device)
            dist.broadcast(tensor=step_tensor, src=0)
            globalMinStep = step_tensor[0].item()
            totalLen = step_tensor[-1].item()
            endIdx = step_tensor[args.this_rank].item()
            startIdx = 0 if args.this_rank == 1 else step_tensor[args.this_rank - 1].item()

            clients_tensor = torch.zeros([totalLen], dtype=torch.int).to(device=device)
            distances_tensor = torch.zeros([totalLen], dtype=torch.float).to(device=device)
            dist.broadcast(tensor=clients_tensor, src=0)
            dist.broadcast(tensor=distances_tensor, src=0)
            nextClientIds = [clients_tensor[x].item() for x in range(startIdx, endIdx)]
            nextdistances = [distances_tensor[x].item() for x in range(startIdx, endIdx)]
            logging.info(f"nextdistances: {nextdistances}")

            receDur = time.time() - receStart

            #logging.info("====Finish receiving ps")
            evalStart = time.time()
            # addition1 here can add the eval and dump here 
            
            # if epoch % args.dump_epoch == 0 and args.this_rank == 1:
            #     model = model.to(device='cpu')
            #     with open(logDir+'/'+str(args.model)+'_'+str(epoch)+'.pth.tar', 'wb') as fout:
            #         pickle.dump(model, fout)

            #     logging.info("====Dump model successfully")
            #     model = model.to(device=device)
            end_time = time.time()  
            iteration_time = end_time - start_time  
            csv_dic = args.log_path + "/here.csv"
            logging.info(f"iteration_time: {iteration_time}")  
            # with open(csv_dic, mode='r', newline='') as file:
            #     reader = csv.reader(file)
            #     lines = list(reader)

            # lines[-1].append(iteration_time)

            # with open(csv_dic, mode='w', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerows(lines)
            # file.close()
            with open("./log/"+args.data_set+"_"+args.record_name+"_choose.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([cos_choose,distance_choose])
            file.close()
            cos_choose = 0
            distance_choose = 0
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.info("====Error: {}, {}, {}, {}".format(e, exc_type, fname, exc_tb.tb_lineno))
            break

        if stop_flag.value:
            break
        
    queue.put({rank: [None, None, None, True, -1, -1]})
    logging.info("Worker {} has completed epoch {}!".format(args.this_rank, epoch))


def Super_train(model_glob, count_epoch, iteration, warmup_round, best_pretrain_weight, w_glob,
dataset_train, dict_users_unlabeled, idx, pseudo_label_dataset, pseudo_label_index, image_buffer, base_model, ts_score_list,
fl_coefficient_list, fl_updated_model_list, device, args,distance_list,nextdistance):
    global cos_choose, distance_choose
    try:
        logging.info(f"clinet is {idx}")

        # get the index from client number
        idx = idx - 1
        train_data_itr_list = []
        collate_fn = None

        if args.task == 'nlp':
            collate_fn = collate
        elif args.task == 'voice':
            collate_fn = voice_collate_fn

        local_trained = 0
        epoch_train_loss = None
        comp_duration = 0.
        norm_gradient = 0.
        count = 0
        run_start = time.time()


        dataloader_unlabeled_kwargs = {
            'batch_size': 5*args.batch_size,
            'drop_last': True,
            'pin_memory': True,
            'num_workers': 10,
        }

        train_pesudo_label_flag = False

        if args.data_set == "cifar10":
            model_local = get_model('fedfixmatch', "Cifar").to(device)
        else:
            model_local = get_model().to(device)

        if count_epoch == 0:
            logging.info(f"loaded pre-trained weights")
            model_local.load_state_dict(best_pretrain_weight)
        else:
            logging.info("client count_epoch > 0")
            model_local.load_state_dict(w_glob)

        shuffle_index_list = list(copy.deepcopy(dict_users_unlabeled[idx]))
        random.shuffle(shuffle_index_list)
        logging.info(f"shuffle_index_list length is {len(shuffle_index_list)}")

        train_loader_unlabeled = torch.utils.data.DataLoader(
            dataset=DatasetSplit(dataset_train, shuffle_index_list),
            shuffle=False,
            **dataloader_unlabeled_kwargs
        )

        optimizer_local = torch.optim.SGD(model_local.parameters(), lr=0.01, momentum=0.5)
        class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index= -1)
        model_local.train()

        for each in range(args.upload_epoch):
            for i, ((img, img_ema), label) in enumerate(train_loader_unlabeled):
                if args.task == "speech":
                    img = img.unsqueeze(1)
                    img_ema = img_ema.unsqueeze(1)
                logging.info(f"start to iteration")
                input_var = torch.autograd.Variable(img.cuda(device=device))
                target_var = torch.autograd.Variable(label.cuda(device=device))
                minibatch_size = len(target_var)
                labeled_minibatch_size = target_var.data.ne(-1).sum()
                model_out = model_local(input_var)

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

                loss_cnt = 1.
                loss_list = loss.tolist()
                temp_loss = 0
                logging.info(f"loss_list: {type(loss_list)}")
                updated_list = []
                updated_list.append(loss_list)
                for each in updated_list:
                    temp_loss += each**2
                loss_cnt = max(1,loss_list)
                temp_loss = temp_loss/float(loss_cnt)
                if epoch_train_loss is None:
                    epoch_train_loss = temp_loss
                    logging.info("epoch_train_loss ! none {}".format(epoch_train_loss))
                else:
                    epoch_train_loss = (1. - args.loss_decay) * epoch_train_loss + args.loss_decay * temp_loss
                    logging.info("epoch_train_loss = none {}".format(epoch_train_loss))
                local_trained += len(label)
                count += len(label)
                optimizer_local.zero_grad()
                loss.backward()
                optimizer_local.step()

        update_model = copy.deepcopy(model_local)
        speed = 0
        isSuccess = True
        time_spent = time.time() - run_start
        if count > 0:
            speed = time_spent/float(count)
        else:
            isSuccess = False
            logging.info("====Failed to run client")
        time_cost = time_spent

        # logging.info(f"temp_loss: {epoch_train_loss}")

        # logging.info(f"local_trained: {local_trained}")
        # logging.info(f"str(speed) + '_' + str(count): {str(speed) + '_' + str(count)}")
        # logging.info(f"time_cost: {time_cost}")
        # for idx, param in enumerate(server_update.parameters()):
        #     if idx == 0:
        #         print(f"here:{param.data}")
        if args.loss_record:
            logging.info(f"./log/loss/loss_{idx}.csv")
            with open(f"./log/loss/loss_{idx}.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([count_epoch,loss.item()])
        cos_similarity = 0
        logging.info(f'cos_similarity: {cos_similarity}')
        if not args.all_update:
            if args.l_distance:
                local_params = list(model_local.parameters())
                glob_params = list(model_glob.parameters())
                l2_distance = 0.0

                for local_param, glob_param in zip(local_params, glob_params):
                    l2_distance += torch.sum((local_param - glob_param) ** 2)

                l2_distance = torch.sqrt(l2_distance)
                if math.isnan(cos_similarity) and l2_distance.item()>=nextdistance:
                    logging.info("cs is none")
                    ts_score_list.append(0.00001)
                    distance_list.append(l2_distance.item())
                    return epoch_train_loss, local_trained, str(speed) + '_' + str(count), time_cost, isSuccess


                if cos_similarity <= 0.0 and l2_distance.item()>=nextdistance:
                    logging.info("wrong direction, abort this weight")
                    ts_score_list.append(0.00001)
                    distance_list.append(l2_distance.item())
                    return epoch_train_loss, local_trained, str(speed) + '_' + str(count), time_cost, isSuccess
            else:
                if math.isnan(cos_similarity):
                    logging.info("cs is none")
                    ts_score_list.append(0.00001)
                    distance_list.append(0)
                    return epoch_train_loss, local_trained, str(speed) + '_' + str(count), time_cost, isSuccess


                if cos_similarity <= 0.0:
                    logging.info("wrong direction, abort this weight")
                    ts_score_list.append(0.00001)
                    distance_list.append(0)
                    return epoch_train_loss, local_trained, str(speed) + '_' + str(count), time_cost, isSuccess

        else:
            if math.isnan(cos_similarity):
                logging.info("cs is none")
                cos_similarity = 0.00001

            if cos_similarity <= 0.0:
                logging.info("wrong direction, abort this weight")
                cos_similarity = 0.00001

        if args.l_distance and not args.all_update:
            if math.isnan(cos_similarity):
                ts_score_list.append(0.00001)
                distance_choose+=1
            elif cos_similarity <= 0.0:
                ts_score_list.append(0.00001)
                distance_choose+=1
            else:
                ts_score_list.append(cos_similarity)
                cos_choose+=1
            distance_list.append(l2_distance.item())
        else:
            distance_list.append(0)
            ts_score_list.append(cos_similarity)
            cos_choose+=1


        fl_updated_model_list.append(update_model)

        if train_pesudo_label_flag:
            print("delete p_dataset")
            del p_dataset
            gc.collect()
        del model_local
        gc.collect()
        del train_loader_unlabeled
        gc.collect()
        torch.cuda.empty_cache()
        # return epoch_train_loss, local_trained, str(speed) + '_' + str(count), time_cost, isSuccess
        return epoch_train_loss, local_trained, str(speed) + '_' + str(count), time_cost, isSuccess

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.info("====Error: {}, {}, {}, {}, {}".format(e, exc_type, fname, exc_tb.tb_lineno))




def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def initiate_channel():
    BaseManager.register('get_queue')
    BaseManager.register('get_param')
    BaseManager.register('get_stop_signal')
    logging.info(f"port: {args.ps_ip, args.manager_port}")
    manager = BaseManager(address=(args.ps_ip, args.manager_port), authkey=b'queue')

    return manager


if __name__ == "__main__":
    setup_seed(args.seed)
    #here
    manager = initiate_channel()
    manager.connect()
    device = torch.device('cuda', args.this_rank)

    # initailize the queues, model, dataset
    q = manager.get_queue()  # queue for parameter_server signal process
    
    param_q = manager.get_param()  # init
    stop_signal = manager.get_stop_signal()  # stop

    logging.info("====Start to initialize dataset")

    gc.disable()
    model, train_dataset, test_dataset = init_dataset()
    if args.task == "nlp":
        nlp_model_path ='/home/zikang/code/albert-base-v2'
        tokenizer = AlbertTokenizer.from_pretrained(nlp_model_path)
    gc.enable()
    gc.collect()

    splitTrainRatio = []
    client_cfg = {}

    # Initialize PS - client communication channel
    world_size = len(workers) + 1
    this_rank = args.this_rank
    logging.info("learner start connect to process")
    if args.task == "nlp":
        time.sleep(60)
    dist.init_process_group(args.backend,init_method='env://', rank=this_rank, world_size=world_size)

    # Split the dataset
    # question: worker: the total number of gpu ? why it is client here
    # total_worker != 0 indicates we create more virtual clients for simulation
    logging.info(f"config: total_worker: {args.total_worker} and learners: {args.learners}")
    if args.total_worker > 0 and args.duplicate_data == 1:
        workers = [i for i in range(1, args.total_worker + 1)]

    # load data partitioner (entire_train_data)
    # dataConf = os.path.join(args.data_dir, 'sampleConf') if args.data_set == 'imagenet' else None

    # # data partitioner for test data and traindata
    # logging.info("==== Starting training data partitioner =====")
    # training_dataset_size = len(train_dataset)
    # logging.info(f"train_dataset: {train_dataset}")
    # global_trainDB = DataPartitioner(data=train_dataset, splitConfFile=dataConf,
    #                     numOfClass=args.num_class, dataMapFile=args.data_mapfile)
    # logging.info("==== Finished training data partitioner =====")

    # dataDistribution = [int(x) for x in args.sequential.split('-')]
    # distributionParam = [float(x) for x in args.zipf_alpha.split('-')]

    # for i in range(args.duplicate_data):
    #     partition_dataset(global_trainDB, workers, splitTrainRatio, dataDistribution[i],
    #                                 filter_class=args.filter_class, arg = {'balanced_client':0, 'param': distributionParam[i]})
    # global_trainDB.log_selection()

    report_data_info(this_rank, q)
    # splitTestRatio = []

    # logging.info("==== Starting testing data partitioner =====")
    # testsetPartitioner = DataPartitioner(data=test_dataset, isTest=True, numOfClass=args.num_class)
    # logging.info("==== Finished testing data partitioner =====")

    # collate_fn = None

    if args.task == 'nlp':
        collate_fn = collate
    elif args.task == 'voice':
        collate_fn = voice_collate_fn

    # partition_dataset(testsetPartitioner, [i for i in range(world_size-1)], splitTestRatio)
    # global_testDB = select_dataset(this_rank, testsetPartitioner, batch_size=args.test_bsz, isTest=True, collate_fn=collate_fn)
    
    stop_flag = Value(c_bool, False)

    

    #Start to trainging in worker
    logging.info("==== Start to trainging in worker====")
    init_myprocesses(this_rank, world_size, 
                                          q, param_q, stop_flag,
                                          run, args.backend, client_cfg,train_dataset)
                                          
    
    logging.info("==== End trainging in worker====")
    #no need to keep the raw data
    del train_dataset, test_dataset

