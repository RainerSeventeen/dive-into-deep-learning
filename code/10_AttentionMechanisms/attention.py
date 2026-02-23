"""注意力分数函数"""

import math
import torch
from torch import nn
from d2l import torch as d2l

def masked_softmax(X, valid_lens):
    """掩膜 softmax, 指的是超出长度的序列不被计入, 计算出来值都是 0"""
    # 注意就是 softmax 的本质是固定其他维度, 某个轴上的概率和为 1
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1) # 在最后一个维度上做 softmax
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False) # k * n
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False) # q * n
        self.W_v = nn.Linear(num_hiddens, 1, bias=False) # n * 1
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # queries: (batch_size, Q, num_hidden) Q 是 query 的数量(不是size)
        # keys: (batch_size, KV, num_hiddens)
        # 广播扩展到: (batch_size, query_count, kv_count, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1) # (B, Q, KV, n)
        # 这里含义就是 q 和 k 之间两两配对 k * q
        features = torch.tanh(features)
        scores = self.W_v(features).squeeze(-1) # 得到 (B, Q, KV), 移除最后一个维度 1
        # 为了保证矩阵相同, nlp 里面经常会有填充语句到同长度
        # 这里的 valid_lens 就是为了移除对应的 padding
        self.attention_weights = masked_softmax(scores, valid_lens) # (B, Q, KV)

        # 对每一个 batch 
        # dropout 随机置 0 防止过拟合
        # (B, Q, KV) x (B, KV, value_size) 注意这里是 value size 的大小
        return torch.bmm(self.dropout(self.attention_weights), values)
    
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1] # 这里 query_size == key_size == d
        # 执行转置然后构建
        # queries: (B, Q, d), keys^T: (B, d, K)
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d) # (B, Q, K)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)