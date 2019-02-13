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
import torch.nn.functional as F

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
    By taking the encoder outputs and decoder hidden state.

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

        # batch matrix multiplication
        # v          => [batch_size, 1, dec_hidden_dim]
        # energy     => [batch_size, dec_hidden_dim, sequence_len]
        # v * energy => [batch_size, 1, sequence_len]
        attention = torch.bmm(v, energy).squeeze(1)
        # attention is of shape [batch_size, sequence_len]

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    ''' This class implements the Decoder module.
    By using the attention vector produced by the Attention class.

    Args:
        output_dim: A integer indicating the output dimension.
        embedding_dim: A integer indicating the embedding size.
        enc_hidden_dim: A integer indicating the hidden dimension of encoder.
        dec_hidden_dim: A integer indicating the hidden dimension of decoder.
        dropout: A float indicating the amount of dropout.
        attention: A Attention class instance for calculating the attention vector.
    '''
    def __init__(self, output_dim, embedding_dim, enc_hidden_dim, dec_hidden_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.GRU((enc_hidden_dim * 2) + embedding_dim, dec_hidden_dim)
        self.out = nn.Linear((enc_hidden_dim * 2) + (dec_hidden_dim + embedding_dim), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input is of shape [batch_size]
        # hidden is of shape [batch_size, hidden_dim]
        # encoder_outputs is of shape [sequence_len, batch_size, hidden_dim * num_directions]

        input = input.unsqueeze(0)
        # input shape is [1, batch_size]. reshape is needed rnn expects a rank 3 tensors as input.
        # so reshaping to [1, batch_size] means a batch of batch_size each containing 1 index.

        embedded = self.embedding(input)
        # embedded is of shape [1, batch_size, embedding_dim]
        embedded = self.dropout(embedded)

        a = self.attention(hidden, encoder_outputs)
        # a is of shape [batch_size, sequence_len]
        a = a.unsqueeze(1)
        # a shape is [batch_size, 1, sequence_len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs shape is [batch_size, sequence_len, enc_hidden_dim * 2]

        weighted = torch.bmm(a, encoder_outputs)
        # weighted shape is [batch_size, 1, enc_hidden_dim * 2]
        weighted = weighted.permute(1, 0, 2)
        # weighted shape is [1, batch_size, enc_hidden_dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input shape is [1, batch_size, (enc_hidden_dim * 2) + embedding_dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output = [sent len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # sent len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden

        embedded = embedded.squeeze(0)  # [batch_size, embedding_dim]
        output = output.squeeze(0)      # [batch_size, dec_hidden_dim]
        weighted = weighted.squeeze(0)  # [batch_size, enc_hidden_dim * 2]

        output = self.out(torch.cat((output, weighted, embedded), dim=1))
        # output shape is [batch_size, output_dim]

        return output, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    ''' This class contains the implementation of Sequence to sequence model.
    By using the Encoder, Decoder class instances.

    Args:
        encoder: A Encoder class instance.
        decoder: A Decoder class instance.
        device: device type to use.
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

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # to store the outputs of decoder
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs is of shape [sequence_len, batch_size, hidden_dim * num_directions]
        # hidden is of shape [num_layers * num_directions, batch_size, hidden_dim]
        # after processing through encoder,
        # encoder_outputs is of shape [sequence_len, batch_size, hidden_dim * 2]
        # hidden is of shape [batch_size, decoder_hidden_dim], since actual hidden states are passed through a linear layer.
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is always <sos> token
        input = trg[0, :]

        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            use_teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if use_teacher_force else top1)

        return outputs


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)

optimizer = optim.Adam(model.parameters())
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
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'attention_model.pt')

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
