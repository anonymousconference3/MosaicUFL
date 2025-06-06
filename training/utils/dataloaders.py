# from torch.utils.data import Dataset
# import torch
# import os
# import pandas as pd

# # This loader works for Amazon/Yelp Review
# class TextSentimentDataset(Dataset):
#     def __init__(self, data_path, max_len, train=True, tokenizer=None):

#         file = 'training.csv' if train else 'testing.csv'

#         self.df = pd.read_csv(os.path.join(data_path, file), delimiter=',')
#         # A reset reindexes from 1 to len(df), the shuffled df frames are sparse.
#         self.df.reset_index(drop=True, inplace=True)
#         self.tokenizer = tokenizer
#         self.maxlen = max_len
#         self.client_mapping = {}
#         self.targets = []

#         # initiate the (sample, client) pairs
#         for row in self.df.itertuples():
#             (sample_id, client_id, score, text) = row
#             client_id = int(client_id) - 1
#             if client_id not in self.client_mapping:
#                 self.client_mapping[client_id] = []

#             self.targets.append(int(score))
#             self.client_mapping[client_id].append(sample_id)

#     def __len__(self):
#         return(len(self.df))

#     def __getitem__(self, index):
#         review = self.df.loc[index, 'Text']

#         # Classes start from 0.
#         label = int(self.df.loc[index, 'Score']) - 1

#         # Use BERT tokenizer since it needs to be able to match the tokens to the pre trained words.
#         tokens = self.tokenizer.tokenize(review)

#         # BERT inputs typically start with a '[CLS]' tag and end with a '[SEP]' tag. For
#         tokens = ['[CLS]'] + tokens + ['[SEP]']

#         if len(tokens) < self.maxlen:
#             # Add the ['PAD'] token
#             tokens = tokens + ['[PAD]' for item in range(self.maxlen-len(tokens))]
#         else:
#             # Truncate the tokens at maxLen - 1 and add a '[SEP]' tag.
#             tokens = tokens[:self.maxlen-1] + ['[SEP]']

#         # BERT tokenizer converts the string tokens to their respective IDs.
#         token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

#         # Converting to pytorch tensors.
#         tokens_ids_tensor = torch.tensor(token_ids)

#         # Masks place a 1 if token != PAD else a 0.
#         attn_mask = (tokens_ids_tensor != 0).long()

#         return (tokens_ids_tensor, attn_mask), label



from torch.utils.data import Dataset
import torch
import pandas as pd
import os

class TextSentimentDataset(Dataset):
    def __init__(self, data_path, max_len, train=True, tokenizer=None):
        # Load the correct file (training or testing)
        file = 'training.csv' if train else 'testing.csv'
        
        # Load the CSV file with 'label' and 'text' columns
        self.df = pd.read_csv(os.path.join(data_path, file), delimiter=',')
        self.df.reset_index(drop=True, inplace=True)  # Reindex the dataframe
        self.tokenizer = tokenizer
        self.maxlen = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Get the review text and label
        review = self.df.loc[index, 'text']
        label = int(self.df.loc[index, 'label'])  # Labels are 0-based already

        # Tokenize the review text
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.maxlen,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return (encoding['input_ids'].flatten(), encoding['attention_mask'].flatten()), torch.tensor(label, dtype=torch.long)
        