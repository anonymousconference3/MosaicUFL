







import copy, cmath, torch
import numpy as nd
# from mxnet import nd as mnd
from scipy import spatial
import logging



def add_model(dst_model, src_model):
    params1 = dst_model.state_dict().copy()
    params2 = src_model.state_dict().copy()
    with torch.no_grad():
        for name1 in params1:
            if name1 in params2:
                params1[name1] = params1[name1] + params2[name1]
    model = copy.deepcopy(dst_model)
    model.load_state_dict(params1, strict=False)
    return model

def grad_norm(model):
    arr, _ = get_net_arr(model)
    return norm(arr)

# def norm(arr):
#     return mnd.norm(mnd.array(arr)).asnumpy()[0]

def scale_model(model, scale):
    params = model.state_dict().copy()
    scale = torch.tensor(scale)
    with torch.no_grad():
        for name in params:
            params[name] = params[name].type_as(scale) * scale
    scaled_model = copy.deepcopy(model)
    scaled_model.load_state_dict(params, strict=False)
    return scaled_model

# def cosine_similarity(arr1, arr2):
# #     cosine_similarity = 1 - spatial.distance.cosine(arr1, arr2)
# #     return cosine_similarity
#     cs = mnd.dot(mnd.array(arr1), mnd.array(arr2)) / (mnd.norm(mnd.array(arr1)) + 1e-9) / (mnd.norm(mnd.array(arr2)) + 1e-9)
#     return cs.asnumpy()[0]
def cosine_similarity(global_model,local_model):
    params1 = torch.cat([x.view(-1) for x in global_model.parameters()])
    # params2 = torch.cat([x.view(-1) for x in local_model.parameters()])

    # l2_norm
    norm1 = torch.norm(params1, 2)
    norm2 = torch.norm(local_model, 2)

    cosine_similarity = torch.dot(params1, local_model) / (norm1 * norm2)
    logging.info("cos {}".format(cosine_similarity.item()))
    return cosine_similarity.item()

def get_net_arr(model, list = True):
    param_list = model
    if list == True:
        param_list = [param.cpu().data.numpy() for param in model.parameters()]   

    arr = nd.array([[]])
    slist = []
    for index, item in enumerate(param_list):
        slist.append(item.shape)
        item = item.reshape((-1, 1))
        if index == 0:
            arr = item
        else:
            arr = nd.concatenate((arr, item), axis=0)

    arr = nd.array(arr).squeeze()
    
    return arr, slist

def get_net_arr1(model):
    param_list = [param.cpu().data.numpy() for param in model.parameters()]

    arr = nd.array([[]])
    slist = []
    for index, item in enumerate(param_list):
        slist.append(item.shape)
        item = item.reshape((-1, 1))
        if index == 0:
            arr = item
        else:
            arr = nd.concatenate((arr, item), axis=0)

    arr = nd.array(arr).squeeze()
    
    return arr, slist

# def get_net_arr1(model):
#     param_list = [param.cpu().data.numpy() for param in model.parameters()]

#     arr = nd.array([[]])
#     slist = []
#     for index, item in enumerate(param_list):
#         slist.append(item.shape)
#         item = item.reshape((-1, 1))
#         if index == 0:
#             arr = item
#         else:
#             arr = nd.concatenate((arr, item), axis=0)

#     arr = nd.array(arr).squeeze()
    
#     return arr, slist

def grad_cosine_similarity(model1, model2):
    arr1, _ = get_net_arr1(model1)
    arr2, _ = get_net_arr1(model2)
    return cosine_similarity1(arr1, arr2)

def cosine_similarity1(arr1, arr2):
    arr1 = nd.array(arr1)
    arr2 = nd.array(arr2)
    dot_product = nd.dot(arr1, arr2)
    norm1 = nd.linalg.norm(arr1)
    norm2 = nd.linalg.norm(arr2)
    cosine_similarity = dot_product / (norm1 + 1e-9) / (norm2 + 1e-9)
    return cosine_similarity

def sub_model(model1, model2):
    params1 = model1.state_dict().copy()
    params2 = model2.state_dict().copy()
    with torch.no_grad():
        for name1 in params1:
            if name1 in params2:
                params1[name1] = params1[name1] - params2[name1]
    model = copy.deepcopy(model1)
    model.load_state_dict(params1, strict=False)
    return model


