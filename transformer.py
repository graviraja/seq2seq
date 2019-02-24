'''This code contains the implementation of the paper Attention is all you need.

Paper: https://arxiv.org/pdf/1706.03762.pdf
Reference code: https://github.com/bentrevett/pytorch-seq2seq
'''
import os
import math
import time
import spacy
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)


class SelfAttention(nn.Module):
    '''This class implements the Multi-Head attention.

    Args:
        hid_dim: A integer indicating the hidden dimension.
        n_heads: A integer indicating the number of self attention heads.
        dropout: A float indicating the amount of dropout.
        device: A device to use.
    '''
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0
        # in paper, hid_dim = 512, n_heads = 8

        # query, key, value weight matrices
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        # linear layer to applied after concating the attention head outputs.
        self.fc = nn.Linear(hid_dim, hid_dim)

        # scale factor to be applied in calculation of self attention.
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        # query => [batch_size, sent_len, hidden_dim]
        # key => [batch_size, sent_len, hidden_dim]
        # value => [batch_size, sent_len, hidden_dim]

        batch_size = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # Q, K, V => [batch_size, sent_len, hidden_dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        # Q, K, V => [batch_size, n_heads, sent_len, hid_dim//n_heads]

        # z = softmax[(Q.K)/sqrt(q_dim)].V
        # Q => [batch_size, n_heads, sent_len, hid_dim//n_heads]
        # K => [batch_size, n_heads, hid_dim//n_heads, sent_len]
        # Q.K => [batch_size, n_heads, sent_len, sent_len]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy => [batch_size, n_heads, sent_len, sent_len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = self.do(F.softmax(energy, dim=-1))
        # attention => [batch_size, n_heads, sent_len, sent_len]

        x = torch.matmul(attention, V)
        # x => [batch_size, n_heads, sent_len, hid_dim // n_heads]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x => [batch_size, sent_len, n_heads, hid_dim // n_heads]

        # combine all heads
        x = x.view(batch_size, -1, self.hid_dim)

        x = self.fc(x)
        # x => [batch_size, sent_len, hid_dim]
        return x


class PositionwiseFeedforward(nn.Module):
    '''This class implements the Position Wise Feed forward Layer.

    This will be applied after the multi-head attention layer.

    Args:
        hid_dim: A integer indicating the hidden dimension of model.
        pf_dim: A integer indicating the position wise feed forward layer hidden dimension.
        dropout: A float indicating the amount of dropout.
    '''
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x => [batch_size, sent_len, hidden_dim]

        x = self.dropout(F.relu(self.fc_1(x)))
        # x => [batch_size, sent_len, pf_dim]

        x = self.fc_2(x)
        # x=> [batch_size, sent_len, hid_dim]
        return x


class EncoderLayer(nn.Module):
    '''This is the single encoding layer module.

    '''
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.ln = nn.LayerNorm(hid_dim)
        self.do = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src => [batch_size, sent_len, hid_dim]
        # src_mask => [batch_size, sent_len]

        # apply the self attention layer for the src, then add the src(residual), and then apply layer normalization
        src = self.ln(src + self.do(self.sa(src, src, src, src_mask)))

        # apply the self positionwise_feedforward layer for the src, then add the src(residual), and then apply layer normalization
        src = self.ln(src + self.do(self.pf(src)))
        return src


class Encoder(nn.Module):
    '''This is the complete Encoder Module.

    It stacks multiple Encoderlayers on top of each other.

    Args:
        input_dim: A integer indicating the input vocab size.
        hid_dim: A integer indicating the hidden dimension of the model.
        n_layers: A integer indicating the number of encoder layers in the encoder.
        n_heads: A integer indicating the number of self attention heads.
        pf_dim: A integer indicating the hidden dimension of positionwise feedforward layer.
        encoder_layer: EncoderLayer class.
        self_attention: SelfAttention Layer class.
        positionwise_feedforward: PositionwiseFeedforward Layer class.
        dropout: A float indicating the amount of dropout.
        device: A device to use.
    '''
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, encoder_layer, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.encoder_layer = encoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(1000, hid_dim)

        self.layers = nn.ModuleList([encoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt([torch.FloatTensor([hid_dim])]).to(device)

    def forward(self, src, src_mask):
        # src => [batch_size, sent_len]
        # src_mask => [batch_size, sent_len]

        pos = torch.arange(0, src.shape[1]).unsqueeze(0)
        # pos => [1, sent_len]
        pos = pos.repeat(src.shape[0], 1).to(self.device)
        # pos => [batch_size, sent_len]

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        # src => [batch_size, sent_len, hid_dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        return src


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.ln = nn.LayerNorm(hid_dim)
        self.do = nn.Dropout(dropout)

    def forward(self):
        pass
