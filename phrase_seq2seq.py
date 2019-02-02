''' This contains the implementation of paper
Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation in pytorch.

'''
import os
import math
import spacy
import random

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

# use the gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create data iterators for the data
# padding all the sentences to same length, replacing words by its index,
# bucketing (minimizes the amount of padding by grouping similar length sentences)
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size=BATCH_SIZE, device=device)


class Encoder(nn.Module):
    ''' This class contains the Encoder Module implementation.
    This uses GRU unit, with a single layer.

    Args:
        input_dim: A integer indicating the size of inputs.
        embedding_dim: A integer indicating the embedding size.
        hidden_dim: A integer indicating the size of hidden dimension.
        dropout: A float indicating the dropout amount.
    '''
    def __init__(self, input_dim, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src is of shape [sequence_len, batch_size]

        embedded = self.embedding(src)
        # embedded is of shape [sequence_len, batch_size, embedding_dim]
        embedded = self.dropout(embedded)

        outputs, hidden = self.rnn(embedded)
        # outputs shape is [sequence_len, batch_size, num_directions * hidden_size]
        # hidden shape is [num_layers * num_directions, batch_size, hidden_size]

        return hidden


class Decoder(nn.Module):
    ''' This class contains the Decoder Module implementation.

    Args:
        output_dim: A integer indicating the size of outputs.
        embedding_dim: A integer indicating the embedding size.
        hidden_dim: A integer indicating the size of hidden dimension.
        dropout: A float indicating the amount of dropout.
    '''
    def __init__(self, output_dim, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, embedding_dim)

        # gru receives input_token and context_vector at each time step as inputs
        self.rnn = nn.GRU(embedding_dim + hidden_dim, hidden_dim)

        # linear layer receives input_token, context_vector, hidden_state as inputs
        self.linear = nn.Linear(embedding_dim + hidden_dim + hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
        # input is of shape [batch_size]
        # hidden is of shape [num_layers * num_directions, batch_size, hidden_dim]
        # context is of shape [num_layers * num_directions, batch_size, hidden_dim]

        input = input.unsqueeze(0)
        # input shape is [1, batch_size]. reshape is needed rnn expects a rank 3 tensors as input.
        # so reshaping to [1, batch_size] means a batch of batch_size each containing 1 index.
        embedded = self.embedding(input)
        # embedded is of shape [1, batch_size, embedding_dim]
        embedded = self.dropout(embedded)

        # num_layers and num_directions will always be 1 in the decoder.
        # context shape is [1, batch_size, hidden_dim]
        # hidden shape is [1, batch_size, hidden_dim]

        embedded_context = torch.cat((embedded, context), dim=2)
        # embedded_context is of shape [1, batch_size, embedding_dim + hidden_dim]

        output, hidden = self.rnn(embedded_context, hidden)
        # output is of shape [sequence_len, batch_size, num_directions_hidden_dim]
        # hidden is of shape [num_layers * num_directions, batch_size, hidden_dim]
        # sequence_len, num_layers and num_directions will always be 1 in the decoder, therefore:
        # output shape is [1, batch_size, hidden_dim]
        # hidden shape is [1, batch_size, hidden_dim]

        output = torch.cat((embedded.squeeze(0), context.squeeze(0), output.squeeze(0)), dim=1)
        # output is of shape [batch_size, embedding_dim + hidden_dim + hidden_dim]

        prediction = self.linear(output)
        # prediction of shape [batch_size, output_dim]

        return prediction, hidden


class Seq2Seq(nn.Module):
    ''' This class contains the implementaion of seq2seq module.
    It uses encoder to produce the context vectors.
    It uses decoder to make the predictions.

    Args:
        encoder: A Encoder class instance.
        decoder: A Decoder class instance.
        device: To indicate which device to use.
    '''
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src is of shape [sequence_len, batch_size]
        # trg is of shape [sequence_len, batch_size]
        # if teacher_forcing_ratio is 0.5 we use ground-truth inputs 50% of time and 50% time we use decoder outputs.

        max_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # to store the outputs of the decoder
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of encoder is the context vector.
        context = self.encoder(src)

        # context is also used as the initial hidden state to decoder.
        hidden = context

        # first input to the decoder is the <sos> token
        input = trg[0, :]

        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden, context)
            outputs[t] = output
            use_teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if use_teacher_force else top1)

        # outputs is of shape [sequence_len, batch_size, output_dim]
        return outputs


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256   # encoder embedding size
DEC_EMB_DIM = 256   # decoder embedding size (can be different from encoder embedding size)
HID_DIM = 512       # hidden dimension (must be same for encoder & decoder)
ENC_DROPOUT = 0.5   # encoder dropout
DEC_DROPOUT = 0.5   # decoder dropout (can be different from encoder droput)

# encoder
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
# decoder
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)

# model
model = Seq2Seq(enc, dec, device).to(device)

# optimizer
optimizer = optim.Adam(model.parameters())

# loss function calculates the average loss per token
# passing the <pad> token to ignore_idx argument, we will ignore loss whenever the target token is <pad>
pad_idx = TRG.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


def train(model, iterator, optimizer, criterion, clip):
    ''' Training loop for the model to train.

    Args:
        model: A Seq2Seq model instance.
        iterator: A DataIterator to read the data.
        optimizer: Optimizer for the model.
        criterion: loss criterion.
        clip: gradient clip value.

    Returns:
        epoch_loss: Average loss of the epoch.
    '''
    #  some layers have different behavior during train/and evaluation (like BatchNorm, Dropout) so setting it matters.
    model.train()
    # loss
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        # trg is of shape [sequence_len, batch_size]
        # output of shape [sequence_len, batch_size, output_dim]
        output = model(src, trg)

        # loss function works only 2d logits, 1d targets
        # so flatten the trg, output tensors. Ignore the <sos> token
        # trg shape shape should be [(sequence_len - 1) * batch_size]
        # output shape should be [(sequence_len - 1) * batch_size, output_dim]
        loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))

        # backward pass
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # weight update
        optimizer.step()

        epoch_loss += loss.item()

    # return the average loss
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    ''' Evaluation loop for the model to evaluate.

    Args:
        model: A Seq2Seq model instance.
        iterator: A DataIterator to read the data.
        criterion: loss criterion.

    Returns:
        epoch_loss: Average loss of the epoch.
    '''
    #  some layers have different behavior during train/and evaluation (like BatchNorm, Dropout) so setting it matters.
    model.eval()
    # loss
    epoch_loss = 0

    # we don't need to update the model parameters. only forward pass.
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)     # turn off the teacher forcing

            # loss function works only 2d logits, 1d targets
            # so flatten the trg, output tensors. Ignore the <sos> token
            # trg shape shape should be [(sequence_len - 1) * batch_size]
            # output shape should be [(sequence_len - 1) * batch_size, output_dim]
            loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))

            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

N_EPOCHS = 10           # number of epochs
CLIP = 10               # gradient clip value
SAVE_DIR = 'models'     # directory name to save the models.
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'phrase_model.pt')

best_valid_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

for epoch in range(N_EPOCHS):

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')

# load the parameters(state_dict) that gave the best validation loss and run the model to test.
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
