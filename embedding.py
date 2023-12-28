import torch.nn as nn
import math
import torch


class TokenEmbedding(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)    


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, seq_len):
        super().__init__()

        pe = torch.zeros(seq_len, d_model).float()
        pe.require_grad = False

        for pos in range(seq_len):   
            for i in range(0, d_model, 2):   
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        self.pe = pe.unsqueeze(0)   

    def forward(self, x):
        return self.pe


class EmbeddingLayer(torch.nn.Module):

    def __init__(self, vocab_size, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token = TokenEmbedding(d_model, vocab_size)
        self.position = PositionalEmbedding(d_model, seq_len)
        self.dropout = torch.nn.Dropout(dropout)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
       
    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence).to(self.device)
        return self.dropout(x)