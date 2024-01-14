import torch
import torch.nn as nn
import math
from torch_kmeans import KMeans

from cluster_plotter import ClusterPlotter

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_model, h, dropout, attention_type='full', n_clusters=25):
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


    def full_attention(self, query, key, value, dropout):
        d_k = query.shape[-1]
        
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return attention_scores @ value
    

    def compute_clustered_attention_scores(self, query, key, dropout, n_clusters):
        d_k = query.shape[-1]
        R = torch.nn.functional.normalize(torch.randn(d_k, 16, device=query.device), dim=0)
        low_dim = query @ R
        hash = (low_dim > 0).float().to(query.device)
        
        kmeans = KMeans(n_clusters=n_clusters, num_init=1, p_norm=1, verbose=False)

        centroids = torch.zeros(query.shape[0], query.shape[1], n_clusters, query.shape[3], requires_grad=True).to(query.device)
        labels = torch.zeros(query.shape[0], query.shape[1], query.shape[2]).long().to(query.device)
        
        for i in range(len(hash)):
            for j in range(len(hash[i])):
                clust_res = kmeans(hash[i][j].unsqueeze(0)).labels.squeeze(0)
                labels[i][j] = clust_res
                for c in range(n_clusters):
                    filter = (clust_res == c).nonzero(as_tuple=True)
                    if len(filter[0]) > 0:
                        centroid = query[i][j][filter].mean(dim=0)
                    else:
                        centroid = torch.zeros(d_k).to(query.device)
                    centroids[i][j][c] = centroid

        if self.attention_type == 'clustered':
            ClusterPlotter.plot_clusters(labels[0][0], query[0][0])

        attention_scores = (centroids @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
                    
        return attention_scores, labels


    def clustered_attention(self, query, key, value, dropout, n_clusters):
        attention_scores, labels = self.compute_clustered_attention_scores(query, key, dropout, n_clusters)
        attention_scores = attention_scores @ value

        broadcasted_attention_scores = torch.zeros(query.shape[0], query.shape[1], query.shape[2], query.shape[3], requires_grad=True).to(query.device)

        for i in range(len(broadcasted_attention_scores)):
            for j in range(len(broadcasted_attention_scores[i])):
                for k in range(len(broadcasted_attention_scores[i][j])):
                    cluster_id = labels[i][j][k].item()
                    broadcasted_attention_scores[i][j][k] = attention_scores[i][j][cluster_id]
                    
        return broadcasted_attention_scores
    

    def improved_clustered_attention(self, query, key, value, dropout, n_clusters):
        d_k = query.shape[-1]
        num_top_keys = 32
        
        attention_scores, labels = self.compute_clustered_attention_scores(query, key, dropout, n_clusters)
        improved_attention_scores = torch.empty(attention_scores.shape[0], attention_scores.shape[1], attention_scores.shape[-1], attention_scores.shape[-1]).to(attention_scores.device)

        for i in range(len(improved_attention_scores)):
            for j in range(len(improved_attention_scores[i])):
                for k in range(len(improved_attention_scores[i][j])):
                    cluster_id = labels[i][j][k]
                    top_keys_res = torch.topk(attention_scores[i][j][cluster_id], num_top_keys)
                    cluster_mass = top_keys_res.values.sum()
                    top_keys = top_keys_res.indices
                    top_keys_matrix = torch.empty(top_keys.shape[0], key.shape[-1]).to(query.device)
                    for e, index in enumerate(top_keys):
                        idx = index.int().item()
                        top_keys_matrix[e] = key[i][j][idx]
                    
                    q = query[i][j][k]
                    query_key_attention = (q @ top_keys_matrix.T) / math.sqrt(d_k)
                    query_key_attention = query_key_attention.softmax(dim=0) * cluster_mass

                    improved_attention_scores[i][j][k] = attention_scores[i][j][cluster_id]
                    for idx, p in enumerate(top_keys):
                        p = p.int().item()
                        improved_attention_scores[i][j][k][p] = query_key_attention[idx]

        improved_attention_scores = improved_attention_scores @ value
        return improved_attention_scores


    def forward(self, x):
        query = self.w_q(x)
        key = self.w_k(x)
        value = self.w_v(x)  

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        if self.attention_type == 'full':
            x = self.full_attention(query, key, value, self.dropout)
        elif self.attention_type == 'clustered':
            x = self.clustered_attention(query, key, value, self.dropout, self.n_clusters)
        elif self.attention_type == 'improved_clustered':
            x = self.improved_clustered_attention(query, key, value, self.dropout, self.n_clusters)
        else:
            print('No valid attention type selected')

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)