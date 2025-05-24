import pickle
from collections import OrderedDict

clientdatamap = OrderedDict() 
# Open the file in binary read mode
with open('openimage/clientDataMap', 'rb') as f:
    # Load the data from the file
    clientdatamap = pickle.load(f)

# Now you can use the clientdatamap dictionary

# Get a list of keys if needed
keys_list = list(clientdatamap.keys())

max = 0 

# Iterating through dictionary to find a specific value
for key, value in clientdatamap.items():  # Use .items() to get both keys and values
    if value == 1:
        print(key)  # This will print the key where the value is 13770

    if value > max:
        max = value

print(max)

    

# print(keys_list[0])

# print(clientdatamap[keys_list[0]])
