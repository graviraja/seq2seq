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

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(d_model, d_model)

        self.scale = torch.sqrt(torch.FloatTensor([d_model // n_heads]))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        # query => [batch_size, seq_len, d_model]
        # key => [batch_size, seq_len, d_model]
        # value => [batch_size, seq_len, d_model]

        batch_size = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(batch_size, -1, n_heads, d_model // n_heads).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, n_heads, d_model // n_heads).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, n_heads, d_model // n_heads).permute(0, 2, 1, 3)
        # Q, K, V => [batch_size, n_heads, seq_len, d_model//n_heads]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy => [batch_size, n_heads, seq_len, seq_len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = self.dropout(self.softmax(energy))
        # attention => [batch_size, n_heads, seq_len, seq_len]

        x = torch.matmul(attention, V)
        # x => [batch_size, n_heads, seq_len, d_model//n_heads]
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, d_model)
        # x => [batch_size, seq_len, d_model]

        x = self.fc(x)

        return x


class PositionwiseFeedforward(nn.Module):
    def __init__(self):
        super().__init__()

        # self.fc1 = nn.Linear(d_model, d_ff)
        # self.fc2 = nn.Linear(d_ff, d_model)

        self.fc1 = nn.Conv1d(d_model, d_ff, 1)
        self.fc2 = nn.Conv1d(d_ff, d_model, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x => [batch_size, seq_len, d_model]

        x = x.permute(0, 2, 1)
        # x => [batch_size, d_model, seq_len]

        x = self.dropout(gelu(self.fc1(x)))
        # x => [batch_size, d_ff, seq_len]

        x = self.fc2(x)
        # x => [batch_size, d_model, seq_len]

        x = x.permute(0, 2, 1)
        # x => [batch_size, seq_len, d_model]

        return x


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_self_attn = MultiHeadAttention()
        self.encoder_feed_fwd = PositionwiseFeedforward()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, input_mask):
        # input => [batch_size, seq_len, d_model]

        encoder_outputs = self.layer_norm(input + self.dropout(self.encoder_self_attn(input, input, input, input_mask)))
        encoder_outputs = self.layer_norm(encoder_outputs + self.dropout(self.encoder_feed_fwd(encoder_outputs)))
        return encoder_outputs


class BERT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


def make_batch():
    batch = []
    positive = negative = 0
    while positive != batch_size / 2 or negative != batch_size / 2:
        # randomly pick an index for a and b
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences))

        # convert to tokens
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]

        # create the input by merging a and b
        input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['SEP']] + tokens_b + [word_dict['SEP']]

        # create the segment ids
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # MASK LM
        # 15% o the input sentence tokens
        n_pred = min(max_pred, max(1, int(round(len(input_ids) * 0.15))))

        cand_maked_pos = [i for i, token in enumerate(input_ids)]
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            # 80% of time MASK
            if random() < 0.8:
                input_ids[pos] = word_dict['[MASK]']
            # 10% of time replace with random
            elif random() < 0.5:
                index = randint(0, vocab_size - 1)
                input_ids[pos] = word_dict[number_dict[index]]

        # padding zeros
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])
            negative += 1

    return batch
