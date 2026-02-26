"""注意力函数

包含: masked softmax, additive attention, dot product attention, multi-head attention
"""

import math
import torch
from torch import nn
from d2l import torch as d2l


def masked_softmax(X, valid_lens):
    """对最后一维做 softmax，并把 padding 位置掩蔽掉。"""
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)

    shape = X.shape
    # valid_lens 可以是 (B,) 或 (B, Q)，这里统一成一维方便 sequence_mask
    if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)

    # 被掩蔽位置填充成很小的负数，softmax 后接近 0
    X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
    return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    """加性注意力"""

    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        # queries: (B, Q, q_size), keys/values: (B, K, k_size/v_size)
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 广播后得到每个 query 与每个 key 的组合特征: (B, Q, K, h)
        features = torch.tanh(queries.unsqueeze(2) + keys.unsqueeze(1))
        # 压到 1 个分数: (B, Q, K)
        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # (B, Q, K) x (B, K, v_size) -> (B, Q, v_size)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        # queries: (B, Q, d), keys: (B, K, d)
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        # 多个头拼接后，再做一次线性变换融合信息
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # 1) 先做线性投影，再切分成 num_heads 个头并并行计算
        queries = self.transpose_qkv(self.W_q(queries), self.num_heads)
        keys = self.transpose_qkv(self.W_k(keys), self.num_heads)
        values = self.transpose_qkv(self.W_v(values), self.num_heads)

        # 2) 每个头都需要对应的 valid_lens，所以按头数复制
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0
            )

        # 3) 每个头独立做缩放点积注意力
        output = self.attention(queries, keys, values, valid_lens)

        # 4) 把多个头的输出拼回原始隐藏维度，再做输出映射
        output_concat = self.transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

    @staticmethod
    def transpose_qkv(X, num_heads):
        """把 (B, T, H) 变成 (B*num_heads, T, H/num_heads) 以并行计算每个头。"""
        # H: num_hiddens, T: tokens, 序列长度, Q/KV, H 必须是 num_heads 整数倍
        # X: (B, T, H)
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        # -> (B, T, num_heads, H/num_heads)
        X = X.permute(0, 2, 1, 3)
        # -> (B, num_heads, T, H/num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])
        # -> (B*num_heads, T, H/num_heads)

    @staticmethod
    def transpose_output(X, num_heads):
        """transpose_qkv 的逆过程：把多头结果拼接回 (B, T, H)。"""
        # X: (B*num_heads, T, H/num_heads)
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        # -> (B, num_heads, T, H/num_heads)
        X = X.permute(0, 2, 1, 3)
        # -> (B, T, num_heads, H/num_heads)
        return X.reshape(X.shape[0], X.shape[1], -1)
        # -> (B, T, H)
