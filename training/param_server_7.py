# -*- coding: utf-8 -*-
import csv
import os
import pickle
from fl_aggregator_libs import *
import torch.optim as optim
from random import Random
from torch.utils.data import DataLoader, Dataset
# import matplotlib.pyplot as plt
from typing import Tuple
import torch.nn.functional as F
import fltrust
import dataset as ds
import datetime
from fedil import fed_non_iid,test_img, testset_precision,noniid,fed_iid
from torch.utils.data import Subset
from transformers import PreTrainedTokenizer,AlbertTokenizer
initiate_aggregator_setting()
logging.info("device count {}".format(torch.cuda.device_count()))
logging.info(args)
# for i in range(torch.cuda.device_count()):
#     try:
#         device = torch.device('cuda:'+str(i))
#         torch.cuda.set_device(i)
#         logging.info(f'End up with cuda device {torch.rand(1).to(device=device)}')
#         break
#     except Exception as e:
#         assert i != torch.cuda.device_count()-1, 'Can not find a feasible GPU'
device = torch.device('cuda:0')
torch.cuda.set_device(0)
logging.info(f'End up with cuda device {torch.rand(1).to(device=device)}')

entire_train_data = None
sample_size_dic = {}


sampledClientSet = set()
distance_dict = {}
client_time = {}
one_client = {}
two_client = {}
three_client = {}
four_client = {}
five_client = {}
six_client = {}
seven_client = {}
epoch_time = 0

os.environ['MASTER_ADDR'] = args.ps_ip
os.environ['MASTER_PORT'] = args.ps_port
#os.environ['NCCL_DEBUG'] = 'INFO'

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

    def get_delta_w(self, nestedLr=0.01):
        delta_ws = []
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
                    d_p.add_(p.data,alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p,alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf,momentum)
                    else:
                        d_p = buf

                if nestedLr == 0.01:
                    delta_ws.append(group['lr'] * d_p)
                else:
                    delta_ws.append(nestedLr * d_p)

        return delta_ws
def collect_loss(delta_wss):
    count = 0
    sum_loss = 0
    for each in delta_wss:
        count+=1
        sum_loss += each / args.learning_rate
    avg_loss = sum_loss/count
    return sum_loss, avg_loss

"""
Initiate the clientSampler
它从队列中获取客户端的数据信息，如数据距离和大小，并注册到客户端采样器中。
如果提供了采样器的路径，则从文件加载采样器；否则，将创建一个新的采样器。
"""

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        #batch_size = target.size(0)

        #logging.info("====To get accuracy, top-k is {}, while shape is {}".format(maxk, output.shape))
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # print the target
        #logging.info(f"====Target:{target.cpu().numpy().flatten()}")
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k)

            #logging.info(f"====top: {k}, sum: {correct_k.item()}, predictions: {correct[:k].cpu().numpy().sum(0).flatten()}")

        return res
def tttest_model(net_g):
    global test_dataset,subset_test_dataset
    if args.task == "nlp":
        bs = 50
    else:
        bs = 250
    dataloader_kwargs = {
        'batch_size': bs,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': 5,
    }
    data_loader = torch.utils.data.DataLoader(
        dataset=subset_test_dataset,
        shuffle=False,
        **dataloader_kwargs
    )
    net_g.eval()
    test_loss = 0
    correct = 0
    a = 0
    topk=(1, 1)
    total = 0
    if args.task == "nlp":
        for idx, data in enumerate(data_loader):
            data, target = mask_tokens_here(data, tokenizer, args)
            data, target = data.cuda(device=0), target.cuda(device=0)
            outputs = net_g(input_ids=data, labels=target)
            loss = outputs.loss
            #test_loss += loss.data.item()
            #acc = accuracy(outputs[1].view(-1, 30000), target.view(-1), topk=(1, 5))
            #logging.info(f"outputs: {outputs}, target: {target}")
            logits = outputs.logits
            accu = compute_accuracy(logits, target)
            logging.info(f"accuracy: {accu}")
            #logging.info(f"outputs shape: {outputs[0].shape}, target shape: {target.shape}")
            #output = outputs[1].view(-1, 30000)
            #target = target.view(-1)
            #logging.info(f"output: {output}, target: {target}")
            #logging.info(f"output shape: {output.shape}, target shape: {target.shape}")
            #with torch.no_grad():
                #maxk = max(topk)
                #_, pred = output.topk(maxk, 1, True, True)
                #correct = pred.eq(target.view(1, -1).expand_as(pred))
                #acc = []
                #for k in topk:
                    #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                    #acc.append(correct_k)
            #correct += acc[0].item()
            correct += accu * target.size(0)    
            total += target.size(0)  # 增加总样本数量

        # 计算最终的准确率
        acc_final = 100.00 * correct / total

        return acc_final,test_loss
    else:
        for idx, (data, target) in enumerate(data_loader):
            if args.task == "speech":
                data = data.unsqueeze(1)  
            # 2. testing dataset 不变：10000
            if idx == 0:
                logging.info(f"data :{data[0],data[1]}")
            a+=len(data)
            data, target = data.cuda(device=0), target.cuda(device=0)
            log_probs = net_g(data)
            # logging.info(f"len: {a}")
            # logging.info(f"log_probs: {log_probs}")
            test_loss += F.cross_entropy(log_probs, target, reduction='sum',ignore_index=-1).item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            # logging.info(f"y_pred: {y_pred}")
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
    
        return accuracy, test_loss


def compute_accuracy(logits, target):
    # 获取模型的预测 (logits: [batch_size, seq_len, vocab_size])
    predictions = torch.argmax(logits, dim=-1)  # shape: [batch_size, seq_len]

    # 忽略 target 中值为 -100 的部分，只对非 -100 的部分计算准确率
    mask = target != -100  # 生成掩码，target 中为 -100 的部分将被忽略

    # 使用 mask 过滤 predictions 和 target，只保留非 -100 的部分
    filtered_predictions = predictions[mask]
    filtered_target = target[mask]

    # 可选：打印过滤后的 predictions 和 target 以进行调试
    #logging.info(f"Filtered predictions: {filtered_predictions}")
    #logging.info(f"Filtered target: {filtered_target}")

    # 计算正确的预测
    correct_predictions = (filtered_predictions == filtered_target)  # 只对非 -100 的位置比较

    # 统计正确预测的数量
    correct_count = correct_predictions.sum().item()

    # 统计总的有效词汇数量（即非 -100 的词汇数量）
    total_count = mask.sum().item()

    # 计算准确率
    accuracy = correct_count / total_count if total_count > 0 else 0

    # 打印准确率
    logging.info(f'Accuracy: {accuracy * 100:.2f}%')
    
    return accuracy

def registerTime(total_imgs, numOfClients,dict_users_unlabeled):
    if os.path.exists("/home/zikang/code/FLPerf/benchmark/dataset/data/device_info/client_device_capacity"):
        with open("/home/zikang/code/FLPerf/benchmark/dataset/data/device_info/client_device_capacity", 'rb') as fin:
            # {clientId: [computer, bandwidth]}
            global_client_profile = pickle.load(fin)
    clientId = 1
    num_client_profile = max(1, len(global_client_profile))
    for index in range(numOfClients):
        mapped_id = max(1, clientId%num_client_profile)
        systemProfile = global_client_profile[mapped_id] if mapped_id in global_client_profile else [1.0, 1.0]
        registerClientTime(clientId,len(dict_users_unlabeled[clientId-1])/(total_imgs/numOfClients), speed=systemProfile)
        clientId += 1
    sortClinetTime()
def registerClientTime(clientId,size,speed):
    global client_time
    model_size = 65536*2.5
    #beacause it not run all data, it only run upload_epoch times batch_size
    batch_size = 16
    upload_epoch = 5
    compute_speed = speed['computation']
    bandwidth = speed['communication']
    roundDurationLocal=3.0 * batch_size * upload_epoch*size/float(compute_speed)

    roundDurationComm=model_size/float(bandwidth)

    client_time[clientId] = roundDurationLocal+roundDurationComm




def sortClinetTime():
    global one_client,two_client,three_client,four_client,five_client,six_client,seven_client,client_time
    sorted_clients = [k for k, v in sorted(client_time.items(), key=lambda item: item[1])]
    if args.data_set == "cifar10":
        for each in range(50):
            one_client[sorted_clients[each]] = 0
        for each in range(20):
            two_client[sorted_clients[each+50]] = 0
        for each in range(10):
            three_client[sorted_clients[each+70]] = 0
        for each in range(8):
            four_client[sorted_clients[each+80]] = 0
        for each in range(6):
            five_client[sorted_clients[each+88]] = 0
        for each in range(4):
            six_client[sorted_clients[each+94]] = 0
        for each in range(2):
            seven_client[sorted_clients[each+98]] = 0
    elif args.data_set == "google_speech":
        for each in range(400):
            low_client[sorted_clients[each]] = 0
        for each in range(75):
            median_client[sorted_clients[each+400]] = 0
        for each in range(25):
            high_client[sorted_clients[each+475]] = 0
    elif args.data_set == "openImg":
        for each in range(400):
            low_client[sorted_clients[each]] = 0
        for each in range(75):
            median_client[sorted_clients[each+80]] = 0
        for each in range(25):
            high_client[sorted_clients[each+95]] = 0

    logging.info(f"sorted_clients: {sorted_clients}")
    # logging.info(f"low_client: {low_client}")
    # logging.info(f"median_client: {median_client}")
    # logging.info(f"high_client: {high_client}")
    logging.info(f"client_time: {client_time}")


def initiate_sampler_query(queue, numOfClients,dict_users_unlabeled):
    # Initiate the clientSampler
    global distance_dict,client_time,one_client,two_client,three_client,four_client,five_client,six_client,seven_client
    logging.info("start sampler")
    if args.sampler_path is None:
        client_sampler = clientSampler(args.sample_mode, args.score_mode, args=args, filter=args.filter_less, sample_seed=args.sample_seed)
    else:
        # load sampler
        with open(args.sampler_path, 'rb') as loader:
            client_sampler = pickle.load(loader)
    logging.info("finish the sampler")
    # load client profiles
    global_client_profile = {}
    if os.path.exists(args.client_path):
        with open(args.client_path, 'rb') as fin:
            # {clientId: [computer, bandwidth]}
            global_client_profile = pickle.load(fin)

    collectedClients = 0
    initial_time = time.time()
    clientId = 1
    passed = False
    num_client_profile = max(1, len(global_client_profile))

    # In this simulation, we run data split on each worker, which amplifies the # of datasets
    # Waiting for the data information from clients, or timeout
    while collectedClients < numOfClients or (time.time() - initial_time) > 5000:
        if not queue.empty():
            tmp_dict = queue.get()

            # we only need to go over once
            if not passed and args.sampler_path is None:
                
                rank_src = list(tmp_dict.keys())[0]
                distanceVec = tmp_dict[rank_src][0]
                sizeVec = tmp_dict[rank_src][1]

                for index, dis in enumerate(range(args.num_clients)):
                    # since the worker rankId starts from 1, we also configure the initial dataId as 1
                    mapped_id = max(1, clientId%num_client_profile)
                    systemProfile = []
                    if mapped_id in global_client_profile:
                        systemProfile.append(global_client_profile[mapped_id]['computation'])
                        systemProfile.append(global_client_profile[mapped_id]['communication'])
                    else:
                        systemProfile = [1.0,1.0]
                    # systemProfile = global_client_profile[mapped_id] if mapped_id in global_client_profile else [1.0, 1.0]
                    # for each client
                    # 使用 client_sampler.registerClient 注册客户端，包括源排名、客户端 ID、距离、大小和系统配置。
                    #client_sampler.registerClient(rank_src, clientId, dis, sizeVec[index], speed=systemProfile)
                    distance_dict[clientId] = 0
                    client_sampler.registerClient(rank_src, clientId, dis, len(dict_users_unlabeled[clientId-1]), speed=systemProfile)
                    #使用 client_sampler.registerDuration 注册客户端的训练参数，包括批量大小、上传周期和模型大小。
                    client_sampler.registerDuration(clientId,
                        batch_size=args.batch_size, upload_epoch=args.upload_epoch,
                        model_size=args.model_size)

                    clientId += 1

                passed = True

            collectedClients += 1
    if args.data_set == "cifar10":
        registerTime(50000,100,dict_users_unlabeled)
    elif args.data_set == "google_speech":
        registerTime(103171,500,dict_users_unlabeled)
    elif args.data_set == "openImg":
        registerTime(1229351,2000,dict_users_unlabeled)
    logging.info("====Info of all feasible clients {}".format(client_sampler.getDataInfo()))

    #return a client_sampler register all worker(GPU)
    # register for each worker in client sampler(将data分给每个worker)
    
    return client_sampler

def init_myprocesses(rank, size, queue, param_q, stop_signal, fn, backend,train_dataset,test_dataset):
    #initial the  process setting
    global sampledClientSet
    dict_users_labeled, dict_users_unlabeled = fed_iid(train_dataset, args.num_clients,args.label_rate,args)
    logging.info("server start connect process")
    dist.init_process_group(backend, init_method='env://',rank=rank, world_size=size)
    logging.info("finish allocate the data")
    # After collecting all data information, then decide the clientId to run
    workerRanks = [int(v) for v in str(args.learners).split('-')]
    # the number of gpu for clients
    
    clientSampler = initiate_sampler_query(queue, len(workerRanks),dict_users_unlabeled)
    

    # 对于每个工作进程(GPU)，选择一组要运行的客户端
    clientIdsToRun = []
    for wrank in workerRanks:
        nextClientIdToRun = clientSampler.nextClientIdToRun(hostId=wrank)
        clientSampler.clientOnHost([nextClientIdToRun], wrank)
        clientIdsToRun.append([nextClientIdToRun])
        sampledClientSet.add(nextClientIdToRun)

    # clientTensor = 【【1,2,3】，【4,5,6】，【7,8,9】】
    # broadcast to all worker process
    clientTensor = torch.tensor(clientIdsToRun, dtype=torch.int, device=device)
    dist.broadcast(tensor=clientTensor, src=0)
    
    # Start the PS service
    fn(queue, param_q, stop_signal, clientSampler,train_dataset,test_dataset,dict_users_labeled)

def prune_client_tasks(clientSampler, sampledClientsRealTemp, numToRealRun, global_virtual_clock):

    sampledClientsReal = []
    # 1. remove dummy clients that are not available to the end of training
    logging.info(f"sampledClientsReal: {sampledClientsRealTemp}")
    
    for virtualClient in sampledClientsRealTemp:
        roundDuration = clientSampler.getCompletionTime(virtualClient,
                                batch_size=args.batch_size, upload_epoch=args.upload_epoch,
                                model_size=args.model_size) * args.clock_factor

        if clientSampler.isClientActive(virtualClient, roundDuration + global_virtual_clock):
            sampledClientsReal.append(virtualClient)

    # 2. we decide to simulate the wall time and remove 1. stragglers 2. off-line
    completionTimes = []
    virtual_client_clock = {}
    
    for virtualClient in sampledClientsReal:
        roundDuration = clientSampler.getCompletionTime(virtualClient,
                                batch_size=args.batch_size, upload_epoch=args.upload_epoch,
                                model_size=args.model_size) * args.clock_factor
        completionTimes.append(roundDuration)
        virtual_client_clock[virtualClient] = roundDuration

    # 3. get the top-k completions
    sortedWorkersByCompletion = sorted(range(len(completionTimes)), key=lambda k:completionTimes[k])
    #total_worker:numToRealRun
    top_k_index = sortedWorkersByCompletion[:numToRealRun]

    clients_to_run = [sampledClientsReal[k] for k in top_k_index]

    dummy_clients = [sampledClientsReal[k] for k in sortedWorkersByCompletion[numToRealRun:]]
    round_duration = completionTimes[top_k_index[-1]]

    return clients_to_run, dummy_clients, virtual_client_clock, round_duration

def server_train(model_glob, count_epoch, w_glob, dataset_train, dict_users_labeled, device, args):
    dataloader_kwargs = {
        'batch_size': 5*args.batch_size,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': 5,
    }

    if count_epoch > 0:
        model_glob.load_state_dict(w_glob)
    
    base_model = copy.deepcopy(model_glob).to(device)
    if args.task == "nlp":
        optimizer =  MySGD(model_glob.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = torch.optim.SGD(model_glob.parameters(), lr=0.01, momentum=0.5)
    if args.task == "nlp":
        class_criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device=device)
    else:
        class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index= -1)
    model_glob.train()

    train_loader_labeled = torch.utils.data.DataLoader(
        dataset=DatasetSplit(dataset_train, dict_users_labeled),
        shuffle=True,
           **dataloader_kwargs
    )
    # 1. tarining data set是固定的 = 500
    
    for batch_idx, ((img, img_ema), label) in enumerate(train_loader_labeled):    
        if args.task == "speech":
            img = img.unsqueeze(1)
        if args.task == "nlp":
            img, label= mask_tokens_here(img, tokenizer, args) if args.mlm else (img, img)
        input_var = torch.autograd.Variable(img.cuda(device=0))
        target_var = torch.autograd.Variable(label.cuda(device=0))
        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(-1).sum()    
        if args.task == "nlp":
            #logging.info(f"img, label: {img,label}")
            outputs = model_glob(input_ids=img, labels=label)
            loss = outputs.loss
            logging.info(f"loss: {loss}")
        else:
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

def run(queue, param_q, stop_signal, clientSampler,dataset_train,test_dataset,dict_users_labeled):
    global logDir, sampledClientSet,client_time,one_client,two_client,three_client,four_client,five_client,six_client,seven_client,epoch_time
    
    now = datetime.datetime.now()
    current_time = now.strftime("%m_%d_%H_%M")
    output_path = "./outputs/{}_{}successive_{}rounds_{}/".format(str("Cifar"),str(7),str(2000),str(current_time))
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': 5,
    }

    if args.data_set == "cifar10":    
        model_glob = get_model('fedfixmatch', "Cifar").to(device)
    else:
        model_glob = get_model().to(device)

    if args.task != "nlp":
        warmup_round = 1
    else:
        warmup_round = 1
    acc_threshold = 0
    acc_for_model = 0
    path = os.path.join(output_path, "best_model_weight.pth")

    logging.info("====PS: get in run()")
    
    accuracy = []
    w_glob = []
    epoch_count = 0

    best_pretrain_weight = []

    pseudo_label_dataset = []
    logging.info("warm-up")


    for i in range(warmup_round):
        if args.task != "nlp":
            base_model = server_train(model_glob, epoch_count, w_glob, dataset_train, dict_users_labeled, device, args)
        
        else:
            start_time = time.time()
            base_model = server_train(model_glob, -1, w_glob, dataset_train, dict_users_labeled, device, args)
            end_time = time.time()
            logging.info(f"server train one time time: {end_time - start_time}")
        
        logging.info(f"test_dataset: {len(test_dataset)}")
        test_loader = torch.utils.data.DataLoader(
            dataset = test_dataset,
            shuffle=False,
            **dataloader_kwargs
        )
        model_glob.eval()
        if args.data_set == "openImg":
            acc, loss_train_test_labeled = 0,0
        else:
            start_time = time.time() 
            acc, loss_train_test_labeled = tttest_model(model_glob)
            logging.info('Warm Round {:3d}, Acc {:.2f}%'.format(i, acc))
            end_time = time.time()
            logging.info(f"testing one time time: {end_time - start_time}")
        del test_loader
        gc.collect()
        torch.cuda.empty_cache()

        logging.info('Warm Round {:3d}, Acc {:.2f}%'.format(i, acc))
        # with open(os.path.join(output_path, "pretrain_acc.txt"), 'a') as f:
        #     f.write('Round {:3d}, Acc {:.2f}% \n'.format(i, acc))
        if acc >= acc_threshold:
            print("weights of epoch {} on server have been saved".format(i))
            best_pretrain_weight = copy.deepcopy(model_glob.state_dict())
            acc_threshold = acc
        if i == warmup_round:
            model_glob.load_state_dict(best_pretrain_weight)
    #pre-training
    for _ in range(0):
        base_model = server_train(model_glob, -1, w_glob, dataset_train, dict_users_labeled, device, args)
    logging.info(f"after warm-up, send the model")
    for idx, param in enumerate(model_glob.parameters()):
        dist.broadcast(tensor=param.data, src=0)
    logging.info(f"after warm-up, send the base model")

    for name, param in enumerate(base_model.parameters()):
        dist.broadcast(tensor=param.data, src=0)

    workers = [int(v) for v in str(args.learners).split('-')]
    logging.info(f"workers: {workers}")

    # 3. initialize the record variables
    epoch_train_loss = 0
    
    data_size_epoch = 0   # len(train_data), one epoch

    global_virtual_clock = 0.
    round_duration = 0.

    staleness = 0
    learner_staleness = {l: 0 for l in workers}
    learner_local_step = {l: 0 for l in workers}
    learner_cache_step = {l: 0 for l in workers}
    pendingWorkers = {}
    virtualClientClock = {}
    exploredPendingWorkers = []
    avgUtilLastEpoch = 0.

    s_time = time.time()

    global_update = 0

    clientsLastEpoch = []

    last_sampled_clients = None

    # random component to generate noise
    median_reward = 1.

    epoch_count = warmup_round
    tep_model = copy.deepcopy(model_glob)
    total_model_to_update = 0
    max_time = 0
    all_cos = []
    repeat = 7
    tmp_save= {}
    tmp_save[7] = []
    tmp_save[6] = []
    tmp_save[5] = []
    tmp_save[4] = []
    tmp_save[3] = []
    tmp_save[2] = []
    repeatClient = []
    while True:
        if not queue.empty():
            try:
                handle_start = time.time()
                
                logging.info("epoch {} started".format(epoch_count))
                tmp_dict = queue.get()
                rank_src = list(tmp_dict.keys())[0]
                [w_glob_list,iteration_loss, trained_size, isWorkerEnd, clientIds, speed, testRes, virtualClock,ts_score_list,distance_list] = [tmp_dict[rank_src][i] for i in range(0, len(tmp_dict[rank_src]))]
                for each in ts_score_list:
                    all_cos.append(each)
                # addition3
                # if isWorkerEnd:
                #     logging.info("====Worker {} has completed all its data computation!".format(rank_src))
                #     learner_staleness.pop(rank_src)
                #     if (len(learner_staleness) == 0):
                #         stop_signal.put(1)
                #         break
                #     continue


                learner_local_step[rank_src] += 1

                clientsLastEpoch += clientIds
                ratioSample = 0
                handlerStart = time.time()
                
                #update the clients' utility
                
                for i, (clientId, distance) in enumerate(zip(clientIds, distance_list)):
                    
                    gradients = None
                    ranSamples = float(speed[i].split('_')[1])

                    data_size_epoch += trained_size[i]

                    #update distance
                    distance_dict[clientId] = distance
                    if repeat == 7:
                        repeatClient.append(clientId)
                        if clientId in one_client.keys():
                            one_client[clientId] = distance
                        elif clientId in two_client.keys():
                            two_client[clientId] = distance
                        elif clientId in three_client.keys():
                            three_client[clientId] = distance
                        elif clientId in four_client.keys():
                            four_client[clientId] = distance
                        elif clientId in five_client.keys():
                            five_client[clientId] = distance
                        elif clientId in six_client.keys():
                            six_client[clientId] = distance
                        else:
                            seven_client[clientId] = distance
                    else:
                        if clientId in one_client.keys():
                            one_client[clientId] = distance
                            repeatClient.append(clientId)
                        elif clientId in two_client.keys():
                            two_client[clientId] = distance
                            if repeat == 2:
                                if w_glob_list[i] != None:
                                    tmp_save[2].append(w_glob_list[i])
                                continue
                            repeatClient.append(clientId)
                        elif clientId in three_client.keys():
                            three_client[clientId] = distance
                            if repeat == 3:
                                if w_glob_list[i] != None:
                                    tmp_save[3].append(w_glob_list[i])
                                continue
                            repeatClient.append(clientId)
                        elif clientId in four_client.keys():
                            four_client[clientId] = distance
                            if repeat == 4:
                                if w_glob_list[i] != None:
                                    tmp_save[4].append(w_glob_list[i])
                                continue
                            repeatClient.append(clientId)
                        elif clientId in five_client.keys():
                            five_client[clientId] = distance
                            if repeat == 5:
                                if w_glob_list[i] != None:
                                    tmp_save[5].append(w_glob_list[i])
                                continue
                            repeatClient.append(clientId)
                        elif clientId in six_client.keys():
                            six_client[clientId] = distance
                            if repeat == 6:
                                if w_glob_list[i] != None:
                                    tmp_save[6].append(w_glob_list[i])
                                continue
                            repeatClient.append(clientId)
                        else:
                            seven_client[clientId] = distance
                            if w_glob_list[i] != None:
                                tmp_save[7].append(w_glob_list[i])
                            continue

                            
                    if w_glob_list[i] != None:
                        for idx, (name, param) in enumerate(tep_model.state_dict().items()):
                            if name in w_glob_list[i]: 
                                model_weight = torch.from_numpy(w_glob_list[i][name]).to(device=device)
                                param.add_(model_weight)  
                        total_model_to_update += 1

                if repeat == 7:
                    for group in tmp_save.keys():
                        for each in tmp_save[group]:
                            for idx, (name, param) in enumerate(tep_model.state_dict().items()):
                                if name in each: 
                                    model_weight = torch.from_numpy(each[name]).to(device=device)
                                    param.add_(model_weight)  
                            total_model_to_update += 1


        
                global_update += 1

                # get the current minimum local staleness_sum_epoch
                currentMinStep = min([learner_local_step[key] for key in learner_local_step.keys()])

                staleness += 1
                learner_staleness[rank_src] = staleness

                if learner_local_step[rank_src] >= args.stale_threshold + currentMinStep:
                    pendingWorkers[rank_src] = learner_local_step[rank_src]
                    # lock the worker
                    logging.info("Lock worker " + str(rank_src) + " with localStep " + str(pendingWorkers[rank_src]) +
                                            " , while globalStep is " + str(currentMinStep) + "\n")

                # if the local cache is too stale, then update it
                elif learner_cache_step[rank_src] < learner_local_step[rank_src] - args.stale_threshold:
                    pendingWorkers[rank_src] = learner_local_step[rank_src]

                # release all pending requests, if the staleness does not exceed the staleness threshold in SSP
                pendingWorkers[rank_src] = learner_local_step[rank_src]
                handle_dur = time.time() - handle_start
                logging.info(f"hhhhh: {pendingWorkers,learner_local_step}")
                workersToSend = []

                for pworker in pendingWorkers.keys():
                    # check its staleness
                    if pendingWorkers[pworker] <= args.stale_threshold + currentMinStep:
                        # start to send param, to avoid synchronization problem, first create a copy here?
                        workersToSend.append(pworker)
                    


                if len(workersToSend) > 0:
                    # logging.info(f"exploredPendingWorkers: {exploredPendingWorkers}")
                    # logging.info(f"clientIds: {clientIds}")
                    for clientId in exploredPendingWorkers:
                        clientSampler.registerScore(clientId, avgUtilLastEpoch,
                                                time_stamp=epoch_count, duration=virtualClientClock[clientIds],
                                                success=False
                                  )

                    workersToSend = sorted(workersToSend)
                    avgUtilLastEpoch = 0.

                    logging.info("====Epoch {} completes {} clients with loss {}, sampled rewards are: \n {} \n=========="
                                .format(epoch_count, len(clientsLastEpoch), epoch_train_loss, {x:clientSampler.getScore(0, x) for x in sorted(clientsLastEpoch)}))

                    epoch_train_loss = 0.
                    clientsLastEpoch = []
                    send_start = time.time()

                    # resampling the clients if necessary
                    if epoch_count % args.resampling_interval == 0 or epoch_count == warmup_round:
                        logging.info("====Start to sample for epoch {}, global virtualClock: {}, round_duration: {}"
                                        .format(epoch_count, global_virtual_clock, round_duration))


                        numToSample = int(args.total_worker * args.overcommit)

                        if args.fixed_clients and last_sampled_clients:
                            sampledClientsRealTemp = last_sampled_clients
                        else:
                            sampledClientsRealTemp = sorted(clientSampler.resampleClients(numToSample, cur_time=epoch_count))

                        last_sampled_clients = sampledClientsRealTemp

                        # remove dummy clients that we are not going to run
                        if repeat == 7:
                            clientsToRun = getTop(numToSample)
                        else:
                            clientsToRun = repeatClient
                        repeatClient = []
                        sampledClientSet = set(clientsToRun)
                        
                        logging.info("====Try to resample clients, final takes: \n {}"
                                    .format(clientsToRun, ))#virtualClientClock))

                        allocateClientToWorker = {}
                        allocateClientDict = {rank:0 for rank in workers}

                        # for those device lakes < # of clients, we use round-bin for load balance
                        for c in clientsToRun:
                            clientDataSize = clientSampler.getClientSize(c)
                            numOfBatches = int(math.ceil(clientDataSize/args.batch_size))

                            if numOfBatches > args.upload_epoch:
                                workerId = workers[(c-1)%len(workers)]
                            else:
                                # pick the one w/ the least load
                                workerId = sorted(allocateClientDict, key=allocateClientDict.get)[0]

                            if workerId not in allocateClientToWorker:
                                allocateClientToWorker[workerId] = []

                            allocateClientToWorker[workerId].append(c)
                            allocateClientDict[workerId] = allocateClientDict[workerId] + 1

                        for w in allocateClientToWorker.keys():
                            clientSampler.clientOnHost(allocateClientToWorker[w], w)

                    clientIdsToRun = [currentMinStep]
                    clientsList = []

                    endIdx = 0

                    for worker in workers:
                        learner_cache_step[worker] = currentMinStep
                        endIdx += clientSampler.getClientLenOnHost(worker)
                        clientIdsToRun.append(endIdx)
                        clientsList += clientSampler.getCurrentClientIds(worker)
                        # remove from the pending workers
                        del pendingWorkers[worker]

                    
                    logging.info("aggregate and send model------")
                    
                    logging.info(f"total_model_to_update: {total_model_to_update}")








                    if total_model_to_update> 0:
                        tep_model = fltrust.sub_model(tep_model,model_glob)
                                
                        tep_model = fltrust.scale_model(tep_model, 1.0/(total_model_to_update))
                    
                    if args.server_update and epoch_count %args.server_update_interval == 0:
                        logging.info(f"using server model as update{epoch_count}")
                        tep_model = copy.deepcopy(model_glob)


                    

                    w_glob = copy.deepcopy(tep_model.state_dict())
                    logging.info(f"{epoch_count,rank_src}, send w_glob")
                    for name, param in w_glob.items(): 
                        dist.broadcast(tensor=param, src=0) 
                    del tmp_dict
                    base_model = server_train(model_glob, epoch_count, w_glob, dataset_train, dict_users_labeled, device, args)
                    
                    
                    test_loader = torch.utils.data.DataLoader(
                        dataset=test_dataset,
                        shuffle=False,
                        **dataloader_kwargs
                    )
                    model_glob.eval()
                    if repeat == 7:
                        if epoch_count % args.eval_interval == 0:
                            start_time = time.time()
                            value = 10  
                            tensor = torch.tensor([value], dtype=torch.int).to(device = device)
                            dist.broadcast(tensor=tensor, src=0)
                            acc, loss_train_test_labeled = tttest_model(model_glob)
                            
                                #acc1,mAP1,confusion_matrix = testset_precision(model_glob, test_loader, cls_num)
                                # with open(os.path.join(output_path, "confusion_matrix.txt"), 'a') as f:
                                #     print("iteration is {}".format(epoch_count ), file = f)
                                #     print("\n\n\n", file =f )
                            del test_loader
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            logging.info('Round {:3d}, Acc {:.2f}%'.format(epoch_count-warmup_round, acc))
                            csv_dic = args.log_path + "/"+args.data_set+"_" +args.record_name+"_here.csv"
                            with open(csv_dic, mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([epoch_count-warmup_round, acc.item(),sum(virtualClock),max_time,epoch_time])
                            file.close()
                        else:
                            value = -10  
                            tensor = torch.tensor([value], dtype=torch.int).to(device = device)
                            dist.broadcast(tensor=tensor, src=0)
                        
                        
                        with open(args.load_path, 'wb') as f:
                            pickle.dump(model_glob, f)
                    else:
                        value = -10  
                        tensor = torch.tensor([value], dtype=torch.int).to(device = device)
                        dist.broadcast(tensor=tensor, src=0)
                    gc.collect()
                    torch.cuda.empty_cache()
                    logging.info(f"{epoch_count}, send the model")

                    for name, param in model_glob.state_dict().items():  
                        dist.broadcast(tensor=param, src=0)  

                    logging.info(f"{epoch_count}, send the base model")

                    for name, param in base_model.state_dict().items():  
                        dist.broadcast(tensor=param, src=0)  
                    


 
                    logging.info(f"clientIdsToRun: {clientIdsToRun}")
                    #send the client list
                    dist.broadcast(tensor=torch.tensor(clientIdsToRun, dtype=torch.int).to(device=device), src = 0)
                    logging.info(f"clientsList: {clientsList}")
                    dist.broadcast(tensor=torch.tensor(clientsList, dtype=torch.int).to(device=device),src = 0)
                    distanceList = []
                    for each in clientsList:
                        distanceList.append(distance_dict[each])
                    # last_model_parameters = [torch.clone(p.data) for p in model.parameters()]
                    dist.broadcast(tensor=torch.tensor(distanceList, dtype=torch.float).to(device=device), src=0)
                    # last_model_parameters = [torch.clone(p.data) for p in model.parameters()]

                    if global_update % args.display_step == 0:
                        logging.info("Handle Wight {} | Send {}".format(handle_dur, time.time() - send_start))

                    # update the virtual clock
                    global_virtual_clock += 0
                    received_updates = 0

                    sumDeltaWeights = []
                    clientWeightsCache = {}

                    if args.noise_factor > 0:
                        median_reward = clientSampler.get_median_reward()
                        logging.info('For epoch: {}, median_reward: {}, dev: {}'
                                        .format(epoch_count, median_reward, median_reward*args.noise_factor))
                    all_cos = [epoch_count-warmup_round] + all_cos
                    with open("./log/"+args.data_set+ "_"+args.record_name +"_cos.csv", mode='a', newline='') as file:

                        writer = csv.writer(file)
                        writer.writerow(all_cos)
                    gc.collect()
                    if repeat == 7:
                        epoch_count+=1
                    tep_model = copy.deepcopy(model_glob)
                    total_model_to_update = 0
                    all_cos = []
                    max_time = 0
                    repeat += 1
                    if repeat > 7:
                        repeat = 1
                        del tmp_save
                        tmp_save= {}
                        tmp_save[7] = []
                        tmp_save[6] = []
                        tmp_save[5] = []
                        tmp_save[4] = []
                        tmp_save[3] = []
                        tmp_save[2] = []
                    # logging.info(f"exploredPendingWorkers: {exploredPendingWorkers}")
                    # logging.info(f"clientIds: {clientIds}")
                    # logging.info(f"clientsToRun: {clientsToRun}")

            
                # The training stop
                if(epoch_count >= args.epochs+warmup_round+2):
                    stop_signal.put(1)
                    logging.info('Epoch is done: {}'.format(epoch_count-1))
                    break

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print("====Error: " + str(e) + '\n')
                logging.info("====Error: {}, {}, {}, {}".format(e, exc_type, fname, exc_tb.tb_lineno))
        e_time = time.time()
        if (e_time - s_time) >= float(args.timeout):
            stop_signal.put(1)
            print('Time up: {}, Stop Now!'.format(e_time - s_time))
            break

def getTop(numToSample):
    global one_client,two_client,three_client,four_client,five_client,six_client,seven_client, epoch_time, client_time
    if args.data_set == "cifar10":
        one_number = 25
        two_number = 10
        three_number = 5
        four_number = 4
        five_number = 3
        six_number = 2
        seven_number = 1
    elif args.data_set == "google_speech":
        low_number = 40
        median_number = 7
        high_number = 3
    elif args.data_set == "openImg":
        low_number = 80
        median_number = 14
        high_number = 6
    
    # Sorting from high to low by using reverse=True
    one_cls = sorted(one_client, key=one_client.get, reverse=True)[:one_number]
    two_cls = sorted(two_client, key=two_client.get, reverse=True)[:two_number]
    three_cls = sorted(three_client, key=three_client.get, reverse=True)[:three_number]
    four_cls = sorted(four_client, key=four_client.get, reverse=True)[:four_number]
    five_cls = sorted(five_client, key=five_client.get, reverse=True)[:five_number]
    six_cls = sorted(six_client, key=six_client.get, reverse=True)[:six_number]
    seven_cls = sorted(seven_client, key=seven_client.get, reverse=True)[:seven_number]
    
    return_list = one_cls + two_cls + three_cls + four_cls + five_cls + six_cls + seven_cls
    
    epoch_time = 0
    for each in return_list:
        if client_time[each] > epoch_time:
            epoch_time = client_time[each]
    
    return return_list



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# communication channel for client information
def initiate_channel():
    #initial queue for command exchange
    queue = Queue()
    #initial queue for param exchange
    param = Queue()
    #initial queue for passing stop signals
    stop_or_not = Queue()
    #These lines use the BaseManager.register method to register three methods that allow remote access to these queues
    BaseManager.register('get_queue', callable=lambda: queue)
    BaseManager.register('get_param', callable=lambda: param)
    BaseManager.register('get_stop_signal', callable=lambda: stop_or_not)
    #BaseManager instance to manage above queue
    #server: manager = BaseManager(address=(args.ps_ip, args.manager_port), authkey=b'queue')
    #        manager.start() -- create server
    #Client: manager = BaseManager(address=(args.ps_ip, args.manager_port), authkey=b'queue')
    #        manager.connect() -- connect the server
    #        queue = m.get_queue() -- through the registered method to get the queue
    logging.info(f"port: {args.ps_ip, args.manager_port}")
    manager = BaseManager(address=(args.ps_ip, args.manager_port), authkey=b'queue')

    return manager

if __name__ == "__main__":

    # Control the global random
    setup_seed(args.seed)
    global test_dataset,subset_test_dataset
    #initial the communication channel(initial server)
    manager = initiate_channel()
    manager.start()

    # get three queue
    q = manager.get_queue()  # queue for parameter_server signal process
    param_q = manager.get_param()  # init param
    stop_signal = manager.get_stop_signal()  # stop signal

    logging.info("====Start to initialize dataset")
    model, train_dataset, test_dataset = init_dataset()
    dataset_size = len(test_dataset)
    indices = list(range(dataset_size))
    if args.data_set == "openImg" or args.task == "nlp":
        split = dataset_size // 10
    else:
        split = dataset_size // 1  

    subset_indices = np.random.choice(indices, split, replace=False)
    subset_test_dataset = Subset(test_dataset, subset_indices)
    # initial the model and dataset, world_size: the number of gpu to call learner.py as clients
    # logging.info("---------------------data set {}".format(len(train_dataset)))
    # logging.info("---------------------learners {}".format(args.learners))
    world_size = len(str(args.learners).split('-')) + 1
    this_rank = args.this_rank

    # initial the server process:
    # run: run method to training process(until receive the stop signal)
    # args.backend: nccl（NVIDIA Collective Communications Library）：专为 NVIDIA GPU 设计，提供高性能的通信操作，适用于大规模的 GPU 分布式训练。
    init_myprocesses(this_rank, world_size,q, param_q, stop_signal, run, args.backend,train_dataset,test_dataset)

    #close the communication channel
    manager.shutdown()
    logging.info("==== close the communication channel ====")




