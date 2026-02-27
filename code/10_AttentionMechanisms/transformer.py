"""Transformer 架构"""

import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

from .attention import MultiHeadAttention

class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络, 本质实际是单隐藏层三维 MLP """
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, *kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        # pytorch 中的线性层 只关心最后一个维度, 最后一维大小应该等于 in_features
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
        
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    """残差连接 + 层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
        
    def forward(self, X, Y):
        """对输出做残差链接"""
        # X 是进入多头注意力的输入, Y 是多头注意力的输出
        return self.ln(self.dropout(Y) + X)

class EncoderBlock(nn.Module):
    """Transformer 编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(key_size, query_size,
            value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)
    
    def forward(self, X, valid_lens):
        Y = self.attention(X, X, X, valid_lens) # 点积所以全部一样
        Y = self.addnorm1(X, Y) # 对 多头输入 X 和输出 Y 执行
        Y = self.ffn(Y) # 位置前馈网络
        Y = self.addnorm2(Y) # 再来一次 Add & Norm
        return Y

class TransformerEncoder(nn.Module):
    """Transformer 编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                norm_shape, ffn_num_input, ffn_num_hiddens,
                                num_heads, dropout, use_bias))
    
    def froward(self, X, valid_lens, *args):
        # 这里乘以 sqrt(d)
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        # self.attention_weights = [None] * len(self.blks) # 这个是为了可视化做的
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            # self.attention_weights[i] = blk.attention.attention.attention_weights

        return X