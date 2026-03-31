"""Attention 相关代码

Scaled Dot-Product Attention, Multi-Head Attention, Position Encoding
Feed Forward Network, Residual Connection + LayerNorm (AddAndNorm)
Encoder Block, Decoder Block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """缩放点积注意力 (兼容 MHA)

        Args:
            Q : [batch_size, (h,), seq_len_q, d_k]
            K : [batch_size, (h,), seq_len_k, d_k]
            V : [batch_size, (h,), seq_len_k, d_v]
            mask : [batch_size, seq_len_q, seq_len_k]. Defaults to None.

        Returns:
            attn_weight: [batch_size, seq_len_q, seq_len_k]
            output: [batch_size, seq_len_q, d_v]
        """
        # torch.matmul() 只对最后两个维度做矩阵乘法, 前面的视作 batch, 能够兼容 MHA
        # K.transpose(-2, -1) 交换张量倒数第二维和倒数第一维
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, Lq, Lk]
        d_k = Q.size(-1)
        scores = scores / (d_k ** 0.5) # 缩放

        if mask is not None:
            # 对 mask 中为 0 的位置写成 -inf
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # 对最后一个维度做 softmax
        attn_weights = F.softmax(scores, dim=-1)    # [B, Lq, Lk]
        attn_weights = self.dropout(attn_weights)   # 一般只对权重做 dropout
        output = torch.matmul(attn_weights, V)      # [B, Lq, d_v]
        
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        
        # 可学习参数要放到 self 里面去
        self.W_q = nn.Linear(d_model, d_model) # 所有头映射矩阵的拼接版
        self.W_k = nn.Linear(d_model, d_model) # [d_model, h * d_k]
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # API 写法, 注意 API 不返回 attn_weight
        # self.attention = F.scaled_dot_product_attention()
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.out_dropout = nn.Dropout(dropout)


    def forward(self, Q, K, V, mask=None):
        """多头注意力
        
        Args:
            Q : [batch_size, seq_len_q, d_model]
            K : [batch_size, seq_len_k, d_model]
            V : [batch_size, seq_len_k, d_model]
            mask : [batch_size, seq_len_q, seq_len_k] 或者 [batch_size, seq_len_k] (padding mask)
            
        Returns:
            output: [batch_size, seq_len_q, d_model]
            attn_weights: [batch_size, nums_heads, seq_len_q, seq_len_k]
        """
        batch_size = Q.shape[0]
        

        # 线性映射到多个头
        Q = self.W_q(Q) # [bs, Lq, d_model]
        K = self.W_k(K) # [bs, Lk, d_model]
        V = self.W_v(V) # [bs, Lk, d_model]

        # 拆分多个头, [B, L, d_model] -> [B, h, L, d_k], 记得把头移动到 L 前面
        Q = Q.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # mask
        if mask is not None:
            # [B, Lq, Lk] -> [B, 1, Lq, Lk], 完整 mask
            # [B, Lk]  -> [B, 1, 1, Lk], padding mask
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)    # unsqueeze 是在位置 n 前插入一个 长度为 1 的维度
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            
        # 每一个 head 做缩放点积, 实际把 num_head 当 batch 一块处理掉了
        output, attn_weight = self.attention(Q, K, V, mask=mask)

        # 拼接 output 并计算
        output = output.transpose(1, 2) # 换回来 [B, L, h, d_k] 
        # contigugous 是把分散的内存拼成连续的, 因为经过了 transpose 会变底层位置
        # 这里 其实 reshape 允许不连续内存, 但是 view 会报错
        output = output.contiguous().reshape(batch_size, -1, self.d_model) # [B, Lq, d_model]
        output = self.W_o(output) # [B, Lq, d_model]
        output = self.out_dropout(output)

        return output, attn_weight

# class PositionEncoding(nn.Module):
# 不确定是否会考查, 先放着

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AddAndNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, sublayer_out):
        """Residual Connect + Layer Norm

        Args:
            x : [batch_size, seq_len, d_model]
            sublayer_out : [batch_size, seq_len, d_model]
        
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        x = x + self.dropout(sublayer_out)
        x = self.norm(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.add_norm1 = AddAndNorm(d_model=d_model, dropout=dropout)
        
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.add_norm2 = AddAndNorm(d_model=d_model, dropout=dropout)
    
    def forward(self, x, mask=None):
        # self attention
        attn_out, attn_weights = self.self_attn(x, x, x, mask=mask)
        x = self.add_norm1(x, attn_out)

        ffn_out = self.ffn(x)
        x = self.add_norm2(x, ffn_out)

        return x, attn_weights

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # masked self attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm1 = AddAndNorm(d_model, dropout)
        
        # cross attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm2 = AddAndNorm(d_model, dropout)
        
        # fnn
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.add_norm3 = AddAndNorm(d_model, dropout)

    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None):
        """自回归的 Decoder Block

        tgt_mask 给 self attention 使用, 一般是 causal mask (或加上 padding mask 的组合)
        memory_mask 给 cross attention 使用, 用于屏蔽 encoder 中 padding 部分
        
        Args:
            x : [batch_size, seq_len, d_model]
            encoder_output : [batch_size, seq_len, d_model]
            tgt_mask: [batch_size, tgt_len, tgt_len] or [batch_size, tgt_len]
            memory_mask: [batch_size, tgt_len, src_len] or [batch_size, src_len]

        Returns:
            output: [batch_size, tgt_len, d_model]
            self_attn_weights: [batch_size, num_heads, tgt_len, tgt_len]
            cross_attn_weights: [batch_size, num_heads, tgt_len, src_len]
        """
        # masked self attention
        self_attn_out, self_attn_weights = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.add_norm1(x, self_attn_out)

        # cross attention
        cross_attn_out, cross_attn_weights = self.cross_attn(
            x, encoder_output, encoder_output, mask=memory_mask)
        x = self.add_norm2(x, cross_attn_out)

        # ffn
        ffn_out = self.ffn(x)
        x = self.add_norm3(x, ffn_out)

        return x, self_attn_weights, cross_attn_weights
