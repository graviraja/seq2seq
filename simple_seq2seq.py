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
