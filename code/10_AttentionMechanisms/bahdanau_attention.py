"""Bahdanau 注意力"""

import math
import torch
from torch import nn
from d2l import torch as d2l

class AttentionDecoder(d2l.Decoder):
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    """不用纠结这个的具体实现, 先跳过"""