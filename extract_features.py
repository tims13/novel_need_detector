from operator import index
import numpy as np
import pandas as pd
import torch
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator
from tqdm import tqdm
from transformers import BertTokenizer
from model import BERT_feature
from config import *

# obtain test data - the need data detected
data_need_results = pd.read_csv(data_need_results_path, index_col=0)
data_need_detected = data_need_results[data_need_results['pred'] == 1]
data_need_detected = data_need_detected.reset_index(drop=True)
data_need_detected.to_csv(data_need_detected_path, index=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# load data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Model parameter
MAX_SEQ_LEN = 128
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
# Fields
text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.int)
fields_train = [('text', text_field), ('label', None), ('novel', None)]
fields_test = [('text', text_field), ('label', None), ('novel', None), ('pred', None)]
train = TabularDataset(
    path = data_novel_train_path,
    format = 'csv',
    skip_header = True,
    fields = fields_train
)
test = TabularDataset(
    path = data_need_detected_path,
    format = 'csv',
    skip_header = True,
    fields = fields_test
)

train_iter = Iterator(train, batch_size=1, device=device, sort=False, sort_within_batch=False, repeat=False, shuffle=False)
test_iter = Iterator(test, batch_size=1, device=device, sort=False, sort_within_batch=False, repeat=False, shuffle=False)

model = BERT_feature(feature_len).to(device)
print("Computing deep features...")

train_features = []
for x in tqdm(train_iter):
    text = x.text.type(torch.LongTensor)
    text = text.to(device)
    feature = model(text)
    train_features.append(feature.detach().cpu().numpy())
train_features = np.vstack(train_features)
print(train_features.shape)

test_features = []
for x in tqdm(test_iter):
    text = x.text.type(torch.LongTensor)
    text = text.to(device)
    feature = model(text)
    test_features.append(feature.detach().cpu().numpy())
test_features = np.vstack(test_features)
print(test_features.shape)

# save the results
np.savez(data_np_path, train=train_features, test=test_features)
print("The features are saved in "+ data_np_path)
