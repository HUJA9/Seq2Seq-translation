import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import masked_cross_entropy
import prepare
import importlib


# Configure models
attn_model = 'dot'
# hidden_size = 500
# n_layers = 2
# dropout = 0.1
# batch_size = 100
# batch_size = 50
USE_CUDA=prepare.USE_CUDA

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout)

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=False)
        # print("Input_length:",input_lengths)
        # print("packed:", packed, " hidden:", hidden)
        output, hidden = self.gru(packed, hidden)
        # output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)  # unpack (back to padded)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output)
        # output = output[:,:,:self.hidden_size]+output[:,:,self.hidden_size]
        return output, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        #         print('[attn] seq len', seq_len)
        #         print('[attn] encoder_outputs', encoder_outputs.size()) # S x B x N
        #         print('[attn] hidden', hidden.size()) # S=1 x B x N

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(batch_size, seq_len))  # B x S
        #         print('[attn] attn_energies', attn_energies.size())

        '''
        if USE_CUDA:
            attn_energies = attn_energies.cuda()
        '''

        # For each batch of encoder outputs
        for b in range(batch_size):
            # Calculate energy for each encoder output
            for i in range(seq_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        #         print('[attn] attn_energies', attn_energies.size())
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden[0].dot(encoder_output[0])  # 2D->1D here
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden[0].dot(energy[0])  # 2D->1D here
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            # print("ENERGY：",energy.size())
            # print("V:",self.v,self.v.size())
            energy = self.v[0].dot(energy[0])
            return energy



'''
rnn_output = Variable(torch.zeros(1, 2, 10))
encoder_outputs = Variable(torch.zeros(3, 2, 10))
attn = Attn('concat', 10)
attn(rnn_output, encoder_outputs)

attn_weights = torch.zeros(2, 1, 3)
print('attn_weights', attn_weights.size())
encoder_outputs = torch.zeros(3, 2, 10)
print('encoder_outputs', encoder_outputs.size())
#    B x N x M
#  , B x M x P
# -> B x N x P
context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
context = context.transpose(0, 1)
print('context', context.size())
'''

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time (in order to do teacher forcing)

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        # print("batch_size:",batch_size,"hidden_size:",hidden_size)
        #         print('[decoder] input_seq', input_seq.size()) # batch_size x 1
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size)  # S=1 x B x N; self.hidden_size
        #         print('[decoder] word_embedded', embedded.size())

        # Get current hidden state from input word and last hidden state
        #         print('[decoder] last_hidden', last_hidden.size())
        rnn_output, hidden = self.gru(embedded, last_hidden)
        #         print('[decoder] rnn_output', rnn_output.size())

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        #         print('[decoder] attn_weights', attn_weights.size())
        #         print('[decoder] encoder_outputs', encoder_outputs.size())
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N
        #         print('[decoder] context', context.size())

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        #         print('[decoder] rnn_output', rnn_output.size())
        #         print('[decoder] context', context.size())
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6)
        #         output = F.log_softmax(self.out(concat_output))
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights


'''
# ## Testing the models
#
# To make sure the encoder and decoder are working (and working together) we'll do a quick test.
#
# First by creating and padding a batch of sequences:

# In[394]:
# Input as batch of sequences of word indexes
batch_size = 2
random_batch=prepare.random_batch
input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size)
print('input_batches', input_batches.size())
print('target_batches', target_batches.size())


# Create models with a small size (in case you need to manually inspect):

# In[395]:

# Create models
hidden_size = 8
n_layers = 2

input_lang=prepare.input_lang
output_lang=prepare.output_lang
MAX_LENGTH=prepare.MAX_LENGTH
encoder_test = EncoderRNN(input_lang.n_words, hidden_size, n_layers)
decoder_test = LuongAttnDecoderRNN('general', hidden_size, output_lang.n_words, n_layers)

if USE_CUDA:
    encoder_test.cuda()
    decoder_test.cuda()


# Then running the entire batch of input sequences through the encoder to get per-batch encoder outputs:

# In[396]:

# Test encoder
encoder_outputs, encoder_hidden = encoder_test(input_batches, input_lengths, None)
print('encoder_outputs', encoder_outputs.size()) # max_len x B x hidden_size
print('encoder_hidden', encoder_hidden.size()) # n_layers x B x hidden_size


# Then starting with a SOS token, run word tokens through the decoder to get each next word token. Instead of doing this with the whole sequence, it is done one at a time, to support using it's own predictions to make the next prediction. This will be one time step at a time, but batched per time step. In order to get this to work for short padded sequences, the batch size is going to get smaller each time.

# In[397]:

decoder_attns = torch.zeros(batch_size, MAX_LENGTH, MAX_LENGTH)
decoder_hidden = encoder_hidden
decoder_context = Variable(torch.zeros(1, decoder_test.hidden_size))
criterion = nn.NLLLoss()

max_length = max(target_lengths)
all_decoder_outputs = Variable(torch.zeros(max_length, batch_size, decoder_test.output_size))

if USE_CUDA:
    decoder_context = decoder_context.cuda()
    all_decoder_outputs = all_decoder_outputs.cuda()

loss = 0


# import masked_cross_entropy
# importlib.reload(masked_cross_entropy)
masked_cross_entropy=masked_cross_entropy.masked_cross_entropy

# Run through decoder one time step at a time
for t in range(max_length - 1):
    decoder_input = target_batches[t]
    target_batch = target_batches[t + 1]

    decoder_output, decoder_context, decoder_hidden, decoder_attn = decoder_test(
        decoder_input, decoder_context, decoder_hidden, encoder_outputs
    )
    print('decoder output = %s, decoder_hidden = %s, decoder_attn = %s' % (
        decoder_output.size(), decoder_hidden.size(), decoder_attn.size()
    ))
    all_decoder_outputs[t] = decoder_output

# print('all decoder outputs', all_decoder_outputs.size())
# print('target batches', target_batches.size())
# print('all_decoder_outputs', all_decoder_outputs.transpose(0, 1).contiguous().view(-1, decoder_test.output_size))
print('target lengths', target_lengths)
loss = masked_cross_entropy(
    all_decoder_outputs.transpose(0, 1).contiguous(),
    target_batches.transpose(0, 1).contiguous(),
    target_lengths
)
# loss = criterion(all_decoder_outputs, target_batches)
# print('loss', loss.size())
print('loss', loss.data)
'''

