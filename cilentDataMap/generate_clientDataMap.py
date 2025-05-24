import pickle
import numpy as np
from torchvision.datasets import CIFAR10


dataset = CIFAR10(root='./data', train=True, download=False)


num_clients = 100


dataset_size = len(dataset)


data_per_client = dataset_size // num_clients
remaining_data = dataset_size % num_clients

data_to_client_map = {}


index = 0
for client_no in range(num_clients):

    size = data_per_client + (1 if client_no < remaining_data else 0)
    for data_index in range(index, index + size):
        data_to_client_map[data_index] = client_no
    index += size


with open('data_to_client_map.pkl', 'wb') as f:
    pickle.dump(data_to_client_map, f)
