''' This is a simple sequence to sequence implementation in pytorch.
We are implementing the machine translation task for German -> English.

'''

import os
import math
import random
import spacy

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import TranslationDataset, Multi30k
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

# create data iterators for the data
# padding all the sentences to same length, replacing words by its index,
# bucketing (minimizes the amount of padding by grouping similar length sentences)
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size=BATCH_SIZE)


class Encoder(nn.Module):
    ''' Sequence to sequence networks consists of Encoder and Decoder modules.
    This class contains the implementation of Encoder module.

    Args:
        input_dim: A integer indicating the size of input dimension.
        emb_dim: A integer indicating the size of embeddings.
        hidden_dim: A integer indicating the hidden dimension of RNN layers.
        n_layers: A integer indicating the number of layers.
        dropout: A float indicating dropout.
    '''
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)  # default is time major
        self.dropout = dropout

    def forward(self, src):
        # src is of shape [sentence_length, batch_size], it is time major

        # embedded is of shape [sentence_length, batch_size, embedding_size]
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)

        # inputs to the rnn is input, (h, c); if hidden, cell states are not passed means default initializes to zero.
        # input is of shape [sequence_length, batch_size, input_size]
        # hidden is of shape [num_layers * num_directions, batch_size, hidden_size]
        # cell is of shape [num_layers * num_directions, batch_size, hidden_size]
        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs are always from the top hidden layer, if bidirectional outputs are concatenated.
        # outputs shape [sequence_length, batch_size, hidden_dim * num_directions]
        return hidden, cell


class Decoder(nn.Module):
    ''' This class contains the implementation of Decoder Module.

    Args:
        output_dim: A integer indicating the size of output dimension.
        embedding_dim: A integer indicating the embedding size.
        hidden_dim: A integer indicating the hidden size of rnn.
        n_layers: A integer indicating the number of layers in rnn.
        dropout: A float indicating the dropout.
    '''
    def __init__(self, embedding_dim, output_dim, hidden_dim, n_layers, dropout):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input is of shape [batch_size]
        # hidden is of shape [n_layer * num_directions, batch_size, hidden_size]
        # cell is of shape [n_layer * num_directions, batch_size, hidden_size]

        input = input.unsqueeze(0)
        # input shape is [1, batch_size]. reshape is needed rnn expects a rank 3 tensors as input.
        # so reshaping to [1, batch_size] means a batch of batch_size each containing 1 index.

        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        # embedded is of shape [1, batch_size, embedding_dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # generally output shape is [sequence_len, batch_size, hidden_dim * num_directions]
        # generally hidden shape is [num_layers * num_directions, batch_size, hidden_dim]
        # generally cell shape is [num_layers * num_directions, batch_size, hidden_dim]

        # sequence_len and num_directions will always be 1 in the decoder.
        # output shape is [1, batch_size, hidden_dim]
        # hidden shape is [num_layers, batch_size, hidden_dim]
        # cell shape is [num_layers, batch_size, hidden_dim]

        predicted = self.linear(output.squeeze(0))  # linear expects as rank 2 tensor as input
        # predicted shape is [batch_size, output_dim]

        return predicted, hidden, cell
