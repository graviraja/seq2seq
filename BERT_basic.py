'''This code contains the implementation of Basic BERT.

'''

import re
import math
import numpy as np
from random import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# BERT Parameters
maxlen = 512
batch_size = 4
max_pred = 20   # max tokens of prediction
n_layers = 12
n_heads = 8
d_model = 768
d_ff = 768 * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2
dropout = 0.1

text = (
    'Hello, how are you? I am Romeo.\n'
    'Hello, Romeo My name is Juliet. Nice to meet you.\n'
    'Nice meet you too. How are you today?\n'
    'Great. My baseball team won the competition.\n'
    'Oh Congratulations, Juliet\n'
    'Thanks you Romeo'
)

sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')   # filter '.', ',', '?', '!'
word_list = list(set(" ".join(sentences).split()))
word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
for i, w in enumerate(word_list):
    word_dict[w] = i + 4
number_dict = {i: w for i, w in enumerate(word_dict)}
vocab_size = len(word_dict)

token_list = list()
for sentence in sentences:
    arr = [word_dict[s] for s in sentence.split()]
    token_list.append(arr)


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(maxlen, d_model)
        self.seg_embedding = nn.Embedding(n_segments, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).repeat(x.size(0), 1)
        embedding = self.tok_embedding(x) + self.pos_embedding(x) + self.seg_embedding(x)

        return self.dropout(self.norm(embedding))


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class PositionwiseFeedforward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class BERT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass
