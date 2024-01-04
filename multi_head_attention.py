import torch
import torch.nn as nn
import math
from torch_kmeans import KMeans


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_model, h, dropout, attention_type='full', n_clusters = 25):
        super().__init__()
        self.d_model = d_model 
        self.h = h

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.attention_type = attention_type
        self.n_clusters = n_clusters

    @staticmethod
    def full_attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value)

    @staticmethod
    def clustered_attention(query, key, value, mask, dropout, n_clusters):
        d_k = query.shape[-1]
        R = torch.nn.functional.normalize(torch.randn(d_k, 32, device=query.device), dim=0)
        low_dim = query @ R
        hash = torch.where(low_dim > 0, 1, 0).double().to(query.device)
        
        clustering = KMeans(n_clusters=n_clusters, verbose=False, num_init=1, p_norm=1)
        centroids = torch.empty(query.shape[0], query.shape[1], n_clusters, query.shape[3]).to(query.device)
        labels = torch.empty(query.shape[0], query.shape[1], query.shape[2]).long().to(query.device)
        for i, s in enumerate(hash):
            result = clustering(s)
            curr_labels = result.labels
            labels[i] = curr_labels
            for j, head in enumerate(curr_labels):
                for id_cluster in range(n_clusters):
                    filter = (head == id_cluster).nonzero(as_tuple=True)
                    centroids[i][j][id_cluster] = torch.mean(query[i][j][filter], dim=0)

        attention_scores = (centroids @ key.transpose(-2, -1)) / math.sqrt(d_k)
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        attention_scores = attention_scores @ value
        broadcasted_attention_scores = torch.empty(query.shape[0], query.shape[1], query.shape[2], query.shape[3]).to(query.device)

        for i in range(broadcasted_attention_scores.shape[0]):
            for j in range(broadcasted_attention_scores.shape[1]):
                broadcasted_attention_scores[i][j] = attention_scores[i][j][labels[i][j]]
                
        return broadcasted_attention_scores

    def forward(self, x, mask):
        query = self.w_q(x)
        key = self.w_k(x)
        value = self.w_v(x)

        print('pre')
        print(query.shape)
        print(key.shape)
        print(value.shape)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        print('post')
        print(query.shape)
        print(key.shape)
        print(value.shape)

        print('mask')
        print(mask.shape)

        if self.attention_type == 'full':
            x = MultiHeadAttentionLayer.full_attention(query, key, value, mask, self.dropout)
        elif self.attention_type == 'clustered':
            x = MultiHeadAttentionLayer.clustered_attention(query, key, value, mask, self.dropout, self.n_clusters)
        elif self.attention_type == 'improved-clustered':
            x = MultiHeadAttentionLayer.improved_clustered_attention(query, key, value, mask, self.dropout)

        print('res')
        print(x.shape)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        print('final')
        print(x.shape)
        print('merged')
        print(self.w_o(x).shape)

        return self.w_o(x)