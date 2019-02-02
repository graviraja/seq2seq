''' This code contains the implementation of paper,
Neural Machine Translation by Jointly Learning to Align and Translate.

'''
import os
import math
import random
import spacy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

from torchtext.datasets import Multi30k, TranslationDataset
from torchtext.data import Field, BucketIterator

# set the random seed to have deterministic results
SEED = 1

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# spacy used for tokenization
spacy_en = spacy.load('en')
spacy_de = spacy.load('de')


def tokenize_de(text):
    # tokenizes the german text into a list of strings(tokens) and reverse it
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]     # list[::-1] used to reverse the list


def tokenize_en(text):
    # tokenizes the english text into a list of strings(tokens)
    return [tok.text for tok in spacy_en.tokenizer(text)]


# torchtext's Field handle how the data should be processed. For more refer: https://github.com/pytorch/text

# use the tokenize_de, tokenize_en for tokenization of german and english sentences.
# German is the src, English is the trg
# append the <sos> (start of sentence), <eos> (end of sentence) tokens to all sentences.
SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)


# we will be using Multi30k dataset. This is a dataset with ~30K parallel English, German, French sentences.

# exts specifies which languages to use as source and target. source goes first
# fields define which data processing to apply for source and target
train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
print('Loaded data...')

# build the vocab
# consider words which are having atleast min_freq.
# words having less than min_freq will be replaced by <unk> token
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)
print('Vocab builded...')

# define batch size
BATCH_SIZE = 32

# use the gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create data iterators for the data
# padding all the sentences to same length, replacing words by its index,
# bucketing (minimizes the amount of padding by grouping similar length sentences)
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size=BATCH_SIZE, device=device)


class Encoder(nn.Module):
    ''' This class contains the implementation of Encoder Module.

    This implements a bidrectional gru model.

    Args:
        input_dim: A integer indicating the size of input.
        embedding_dim: A integer indicating the embedding size.
        enc_hidden_dim: A integer indicating the hidden dimension of encoder.
        dec_hidden_dim: A integer indicating the hidden dimension of decoder.
        dropout: A float indicating the amount of dropout.
    '''
    def __init__(self, input_dim, embedding_dim, enc_hidden_dim, dec_hidden_dim, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = embedding_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, enc_hidden_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src is of shape [sequence_len, batch_size]

        embedded = self.embedding(src)
        # embedded is of shape [sequence_len, batch_size, embedding_dim]
        embedded = self.dropout(embedded)

        outputs, hidden = self.rnn(embedded)
        # outputs is of shape [sequence_len, batch_size, hidden_size * num_directions]
        # hidden is of shape [num_layers * num_directions, batch_size, hidden_size]

        # hidden[-2, :, :] is the last of forwards RNN
        # hidden[-1, :, :] is the last of backwards RNN

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        # outputs shape is [sequence_len, batch_size, enc_hidden_dim * 2]
        # hidden shape is [batch_size, dec_hidden_dim]

        return outputs, hidden


class Attention(nn.Module):
    ''' This class implements the attention mechanism.

    Args:
        enc_hidden_dim: A integer indicating the encoder hidden dimension.
        dec_hidden_dim: A integer indicating the decoder hidden dimension.
    '''
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()

        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim

        # we will concate the encoder outputs and previous state vector of decoder
        self.attn = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Parameter(torch.rand(dec_hidden_dim))

    def forward(self, hidden, encoder_outputs):
        # hidden is of shape [batch_size, hidden_dim]
        # outputs is of shape [sequence_len, batch_size, hidden_dim * num_directions]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat the decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden is of shape [batch_size, sequence_len, dec_hidden_dim]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs is of shape [batch_size, sequence_len, enc_hidden_dim * 2]

        energy = torch.tanh(self.attn(torch.cat((encoder_outputs, hidden), dim=2)))
        # energy is of shape [batch_size, sequence_len, dec_hidden_dim]

        energy = energy.permute(0, 2, 1)
        # energy is of shape [batch_size, dec_hidden_dim, sequence_len]

        # v is of shape [dec_hidden_dim]
        v = self.v.repeat(batch_size, 1)
        # v is of shape [batch_size, dec_hidden_dim]
        v = v.unsqueeze(1)
        # v is of shape [batch_size, 1, dec_hidden_dim]

        attention = torch.bmm(v, energy).squeeze(1)
        # attention is of shape [batch_size, sequence_len]

        return F.softmax(attention, dim=1)
