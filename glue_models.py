import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class OneSentenceModel(nn.Module):
    def __init__(self, pretrained_model, n_classes, n_hidden=1024):
        super().__init__()
        self.bert = copy.deepcopy(pretrained_model)
        self.linear1 = nn.Linear(pretrained_model.encoder_layers[0].linear.linear1.in_features, n_hidden)
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
    def __init__(self, pretrained_model, n_classes, n_hidden=1024):
        super().__init__()
        self.bert = copy.deepcopy(pretrained_model)
        self.linear1 = nn.Linear(2 * pretrained_model.encoder_layers[0].linear.linear1.in_features, n_hidden)
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


class OneSentenceDataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_len=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        column_names = list(self.dataset.features.keys())
        sentence_key = column_names[0]
        label_key = column_names[1]
        
        sentence = self.dataset[sentence_key][index]
        label = self.dataset[label_key][index]

        sentence = self.tokenizer(sentence)['input_ids']
        padding = [self.tokenizer.vocab['[PAD]'] for _ in range(self.seq_len - len(sentence))]
        sentence = sentence + padding

        return torch.tensor(sentence), torch.tensor(label)
    
    def get_n_classes(self):
        column_names = list(self.dataset.features.keys())
        label_key = column_names[1]
        return len(set(self.dataset[label_key]))



class TwoSentencesDataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_len=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        column_names = list(self.dataset.features.keys())
        sentence1_key = column_names[0]
        sentence2_key = column_names[1]
        label_key = column_names[2]
        
        sentence1 = self.dataset[sentence1_key][index]
        sentence2 = self.dataset[sentence2_key][index]
        label = self.dataset[label_key][index]

        sentence1 = self.tokenizer(sentence1)['input_ids']
        padding = [self.tokenizer.vocab['[PAD]'] for _ in range(self.seq_len - len(sentence1))]
        sentence1 = sentence1 + padding

        sentence2 = self.tokenizer(sentence2)['input_ids']
        padding = [self.tokenizer.vocab['[PAD]'] for _ in range(self.seq_len - len(sentence2))]
        sentence2 = sentence2 + padding

        return (torch.tensor(sentence1), torch.tensor(sentence2)), torch.tensor(label)

    def get_n_classes(self):
        column_names = list(self.dataset.features.keys())
        label_key = column_names[2]
        return len(set(self.dataset[label_key]))