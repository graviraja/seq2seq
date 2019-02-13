''' This code contains the implementation of paper,
Neural Machine Translation by Jointly Learning to Align and Translate, using packing padded sequences.

This code is taken reference from: https://github.com/bentrevett/pytorch-seq2seq
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
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size=BATCH_SIZE, sort_key=lambda x: len(x.src), sort_within_batch=True, device=device)


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

    def forward(self, src, src_len):
        # src is of shape [sequence_len, batch_size]
        # src_len is [len_of_each_sentence_in_batch]

        embedded = self.embedding(src)
        # embedded is of shape [sequence_len, batch_size, embedding_dim]

        pack_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)

        packed_outputs, hidden = self.rnn(pack_embedded)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        # outputs shape is [sequence_len, batch_size, hidden_dim * num_dir]
        # hidden shape is [num_layers * num_dir, batch_size, hidden_dim]

        # hidden is stacked => [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from last layer

        # hidden [-2, :, :] => last of the forward rnn
        # hidden [-1, :, :] => last of the backward rnn

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs shape is [sequence_len, batch_size, enc_hidden_dim * 2]
        # hidden shape is [batch_size, dec_hidden_dim]
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()

        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim

        self.attn = nn.Linear(encoder_hidden_dim * 2 + decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Parameter(torch.rand(decoder_hidden_dim))

    def forward(self, decoder_hidden_state, encoder_outputs, mask):
        # decoder_hidden_state shape is [batch_size, decoder_hidden_dim]
        # since decoder_hidden_state is calculated per time step
        # encoder_outputs shape is [sequence_len, batch_size, encoder_hidden_dim * 2]
        # since encoder is bidirectional

        batch_size = encoder_outputs.shape[1]
        sequence_len = encoder_outputs.shape[0]

        # repeat the decoder_hidden_state sequence_len times
        decoder_hidden_state = decoder_hidden_state.unsqueeze(1).repeat(1, sequence_len, 1)
        # decoder_hidden_state shape is [batch_size, sequence_len, decoder_hidden_dim]
        # decoder_hidden_state is in batch major

        # convert the encoder_outputs to batch major form
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs is of shape [batch_size, sequence_len, encoder_hidden_dim * 2]
        # now we can apply linear layer for encoder_outputs and decoder_hidden_dim by concating along dim=2
        # we call this energy
        energy = torch.tanh(self.attn(torch.cat((encoder_outputs, decoder_hidden_state), dim=2)))
        # energy shape is [batch_size, sequence_len, decoder_hidden_dim]
        # reshape the energy into [batch_size, decoder_hidden_dim, sequence_len]
        # which then becomes suitable for matrix multiplication with v to get the a
        v = self.v.repeat(batch_size, 1)
        # v shape is [batch_size, decoder_hidden_dim]

        # reshape the v so it is suitable to multiply the energy
        v = v.unsqueeze(1)
        # v shape is [batch_size, 1, decoder_hidden_dim]

        # batch matrix multiplication
        # v          => [batch_size, 1, dec_hidden_dim]
        # energy     => [batch_size, dec_hidden_dim, sequence_len]
        # v * energy => [batch_size, 1, sequence_len]
        attention = torch.bmm(v, energy).squeeze(1)

        attention = attention.masked_fill(mask == 0, -1e10)
        # attention shape is [batch_size, sequence_len]
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = nn.Linear((enc_hid_dim * 2) + emb_dim + dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        # input shape is [batch_size]
        # hidden shape is [batch_size, decoder_hidden_dim]
        # encoder_outputs shape is [sequence_len, batch_size, enc_hidden_dim * 2]
        # mask is [batch_size, sequence_len]

        input = input.unsqueeze(0)
        # input shape is [1, batch_size]

        embedded = self.dropout(self.embedding(input))
        # embedded shape is [1, batch_size, emb_dim]

        a = self.attention(hidden, encoder_outputs, mask)
        # a shape is [batch_size, sequence_len]

        a = a.unsqueeze(1)
        # a shape is [batch_size, 1, sequence_len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs shape is [batch_size, sequence_len, enc_hidden_dim * 2]

        # batch matrix multiplication
        # a           => [batch_size, 1, seq_len]
        # enc_out     => [batch_size, seq_len, enc_hid_dim * 2]
        # a * enc_out => [batch_size, 1, enc_hid_dim * 2]
        weighted = torch.bmm(a, encoder_outputs)
        # weighted shape is [batch_size, 1, enc_hidden_dim * 2]

        weighted = weighted.permute(1, 0, 2)
        # weighted shape is [1, batch_size, enc_hidden_dim * 2], time major
        # reshaping is needed, so it can be concatenated with embedded to pass to rnn
        rnn_input = torch.cat((weighted, embedded), dim=2)
        # rnn_input shape is [1, batch_size, (enc_hidden_dim * 2 + emb_dim)]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output shape is [1, batch_size, dec_hidden_dim]
        # hidden shape is [1, batch_size, dec_hidden_dim]

        # we need to pass rnn_output, weighted_vector, rnn_input_token as input to linear layer
        # reshape all the three vectors to [batch_size, <respective_dim>] and concat them along dim=1
        output = output.squeeze(0)
        embedded = embedded.squeeze(0)
        weighted = weighted.squeeze(0)

        output = self.out(torch.cat((output, embedded, weighted), dim=1))
        # output shape is [batch_size, output_dim]

        return output, hidden.squeeze(0), a.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, sos_idx, eos_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

    def create_mask(self, src):
        mask = (src != self.pad_idx).permute(1, 0)
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        # src is of shape [src_sequence_len, batch_size]
        # src_len is of shape [batch_size]
        # trg is of shape [trg_sequence_len, batch_size]

        batch_size = src.shape[1]

        if trg is None:
            inference = True
            assert teacher_forcing_ratio == 0, "Must be zero during inference"
            trg = torch.zeros((100, batch_size), dtype=torch.long).fill_(self.sos_idx).to(self.device)
            # trg of shape [100, batch_size], max_len in target sequences is 100
        else:
            inference = False

        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # to store the decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # to store the attention
        attentions = torch.zeros(max_len, batch_size, src.shape[0]).to(self.device)

        # encoding part
        encoder_outputs, hidden = self.encoder(src, src_len)
        # encoder_outputs shape is [src_sequence_len, batch_size, enc_hidden_dim * 2]
        # hidden shape is [batch_size, dec_hidden_dim]

        # initial input to the decoder is always <sos>
        input = trg[0, :]

        mask = self.create_mask(src)
        # mask shape is [batch_size, src_sequence_len]

        for t in range(1, max_len):
            output, hidden, attention = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[t] = output
            attentions[t] = attention
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
            if inference and input.item() == self.eos_idx:
                return outputs[:t], attentions[:t]
        return outputs, attentions


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
PAD_IDX = SRC.vocab.stoi['<pad>']
SOS_IDX = TRG.vocab.stoi['<sos>']
EOS_IDX = TRG.vocab.stoi['<eos>']

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, PAD_IDX, SOS_IDX, EOS_IDX, device).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(model))

optimizer = optim.Adam(model.parameters())
pad_idx = TRG.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):

        src, src_len = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output, _ = model(src, src_len, trg)

        loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src, src_len = batch.src
            trg = batch.trg

            output, _ = model(src, src_len, trg, 0)     # turn off teacher forcing

            loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

N_EPOCHS = 10
CLIP = 10
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'tut4_model.pt')

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
