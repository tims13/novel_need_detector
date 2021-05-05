import torch.nn as nn
from transformers import BertForSequenceClassification, BertModel, BertConfig

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label):
        loss, text_features = self.encoder(text, labels=label)[:2]
        return loss, text_features

class BERT_feature(nn.Module):
    def __init__(self, feature_len):
        super(BERT, self).__init__()
        options_name = "bert-base-uncased"
        config = BertConfig.from_pretrained(options_name)
        self.encoder = BertModel.from_pretrained(options_name, config=config)
        embedding_dim = self.encoder.config.hidden_size
        self.fc = nn.Linear(embedding_dim, feature_len)

    def forward(self, text):
        outputs = self.encoder(text)
        last_hidden_states = outputs[0]
        text_embeddings = last_hidden_states[:,0,:]
        text_features = self.fc(text_embeddings)
        # text_features = self.tanh(features)
        return text_features