import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class OneSentenceModel(nn.Module):
    def __init__(self, bert_model, n_classes, n_hidden=1024):
        super().__init__()
        self.bert = bert_model
        self.linear1 = nn.Linear(bert_model.encoder_layers[0].linear.linear1.in_features, n_hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden, n_classes)
    
    def forward(self, x):
        x = self.bert(x)
        x = x[:, 0, :]
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
    

class TwoSentencesModel(nn.Module):
    def __init__(self, bert_model, n_classes, n_hidden=1024):
        super().__init__()
        self.bert = bert_model
        self.linear1 = nn.Linear(2 * bert_model.encoder_layers[0].linear.linear1.in_features, n_hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden, n_classes)
    
    def forward(self, s):
        s1 = s[0]
        s2 = s[1]
        s1 = self.bert(s1)
        s1 = s1[:, 0, :]
        s2 = self.bert(s2)
        s2 = s2[:, 0, :]
        x = torch.cat((s1, s2), 1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x