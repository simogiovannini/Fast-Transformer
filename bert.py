import torch.nn as nn
from embedding import EmbeddingLayer
from feed_forward import FeedForwardLayer
from multi_head_attention import MultiHeadAttentionLayer


class EncoderLayer(nn.Module):

    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.mha = MultiHeadAttentionLayer(d_model, h, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear = FeedForwardLayer(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x_att = self.mha(x, mask)
        x = self.norm1(x + x_att)
        x_ff = self.linear(x)
        x = self.norm2(x + x_ff)
        return x


class BERTModel(nn.Module):

    def __init__(self, vocab_size, d_model, seq_len, n_layers, h, d_ff, dropout):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size, d_model, seq_len, dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.embedding(x)

        for encoder in self.encoder_layers:
            x = encoder.forward(x, mask)
        return x
    

class TokenPredictionHead(nn.Module):

    def __init__(self, vocab_size, d_model, seq_len, n_layers, h, d_ff, dropout):
        super().__init__()
        self.bert = BERTModel(vocab_size, d_model, seq_len, n_layers, h, d_ff, dropout)
        self.linear = nn.Linear(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(2)
    
    def forward(self, x):
        x = self.bert(x)
        x = self.linear(x)
        x = self.log_softmax(x)
        return x
    
