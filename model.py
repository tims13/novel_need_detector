import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label):
        loss, text_features = self.encoder(text, labels=label)[:2]
        return loss, text_features