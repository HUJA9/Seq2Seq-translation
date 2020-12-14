# coding: utf-8
import matplotlib
import random
import time
import datetime
import math
import socket
hostname = socket.gethostname()

import model
import prepare
import masked_cross_entropy

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
matplotlib.use('Agg')
import numpy as np

import io
import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
# import visdom
# vis = visdom.Visdom()

USE_CUDA = False

SOS_token = 0
EOS_token = 1

# Configure models
attn_model = 'dot'
hidden_size = 500
n_layers = 2
dropout = 0.1
batch_size = 100
MAX_LENGTH = prepare.MAX_LENGTH

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_epochs = 50000
epoch = 0
plot_every = 20
print_every = 100
evaluate_every = 1000

# Initialize models
EncoderRNN=model.EncoderRNN
LuongAttnDecoderRNN=model.LuongAttnDecoderRNN
optim=model.optim
input_lang=prepare.input_lang
output_lang=prepare.output_lang

print(">>>Initial")
encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers, dropout=dropout)
decoder = LuongAttnDecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout=dropout)

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
masked_cross_entropy=masked_cross_entropy.masked_cross_entropy

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every

# # Training
# 
# ## Defining a training iteration
# 
# To train we first run the input sentence through the encoder word by word, and keep track of every output and the latest hidden state. Next the decoder is given the last hidden state of the decoder as its first hidden state, and the `<SOS>` token as its first input. From there we iterate to predict a next token from the decoder.
# 
# ### Teacher Forcing vs. Scheduled Sampling
# 
# "Teacher Forcing", or maximum likelihood sampling, means using the real target outputs as each next input when training. The alternative is using the decoder's own guess as the next input. Using teacher forcing may cause the network to converge faster, but [when the trained network is exploited, it may exhibit instability](http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf).
# 
# You can observe outputs of teacher-forced networks that read with coherent grammar but wander far from the correct translation - you could think of it as having learned how to listen to the teacher's instructions, without learning how to venture out on its own.
# 
# The solution to the teacher-forcing "problem" is known as [Scheduled Sampling](https://arxiv.org/abs/1506.03099), which simply alternates between using the target values and predicted values when training. We will randomly choose to use teacher forcing with an if statement while training - sometimes we'll feed use real target as the input (ignoring the decoder's output), sometimes we'll use the decoder's output.

# In[398]:

# [SOS_token] * 5


# In[399]:

def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, max_length=MAX_LENGTH):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # Run words through encoder
    # print(">>>Train_input_batches:", input_batches.size())
    # print(">>>Train_input_lengths:", len(input_lengths))
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_token] * batch_size])).transpose(0, 1)
#     print('decoder_input', decoder_input.size())
    decoder_context = encoder_outputs[-1]
    decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder

    max_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_length, batch_size, decoder.output_size))

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio

    # TODO: Get targets working
    if True:
        # Run through decoder one time step at a time
        for t in range(max_length):
#             target_batch = target_batches[t]

            # Trim down batches of other inputs
#             decoder_hidden = decoder_hidden[:, :len(target_batch)]
#             encoder_outputs = encoder_outputs[:, :len(target_batch)]

            decoder_output, decoder_context, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_context, decoder_hidden, encoder_outputs
            )
#             print(decoder_output.size(), decoder_hidden.size(), decoder_attn.size())

#             loss += criterion(decoder_output, target_batch)
            all_decoder_outputs[t] = decoder_output

            decoder_input = target_batches[t]
            # TODO decoder_input = target_variable[di] # Next target is next input

    # Teacher forcing: Use the ground-truth target as the next input
    elif use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output[0], target_variable[di])
            decoder_input = target_variable[di] # Next target is next input

    # Without teacher forcing: use network's own prediction as the next input
    else:
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output[0], target_variable[di])

            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]])) # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
            if ni == EOS_token: break

    # Loss calculation and backpropagation
#     print('all_decoder_outputs', all_decoder_outputs.size())
#     print('target_batches', target_batches.size())
    # print("Target_length:",target_lengths)
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # seq x batch -> batch x seq
        target_batches.transpose(0, 1).contiguous(),  # seq x batch -> batch x seq
        target_lengths
    )

    loss.backward()

    # Clip gradient norm
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data, ec, dc

# ## Running training

# Plus helper functions to print time elapsed and estimated time remaining, given the current time and progress.

# In[404]:

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


indexes_from_sentence=prepare.indexes_from_sentence
def evaluate(input_seq, max_length=MAX_LENGTH):
    # print("Input_seq_len:",len(input_seq))
    # input_lengths = [len(input_seq[i]) for i in range(0,len(input_seq))]
    # input_lengths = [len(input_seq[0])]
    input_seqs = [indexes_from_sentence(input_lang, input_seq)]
    input_lengths = [len(input_seqs[0])]
    input_batches = Variable(torch.LongTensor(input_seqs)).transpose(0, 1)
    if USE_CUDA:
        input_batches = input_batches.cuda()

    # Run through encoder
    print("Evaluate......")
    # print("Input_batches:",input_batches.size())
    # print("Input_length:",len(input_lengths))
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden

    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    # Store output words and attention states
    decoded_words = []
    decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_context, decoder_hidden, encoder_outputs
        )
        decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0].item()
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        # Next input is chosen word
        # THIS MIGHT BE THE LAST PART OF BATCHING (or is it already going?)
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if USE_CUDA: decoder_input = decoder_input.cuda()

    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]


def evaluate_randomly():
    pairs = prepare.pairs
    pair = random.choice(pairs)

    output_words, attentions = evaluate(pair[1])
    output_sentence = ' '.join(output_words)
    show_attention(pair[1], output_words, attentions)

    print('>', pair[1])
    print('=', pair[2])
    print('<', output_sentence)
    print('')

def show_plot_visdom():
    buf = io.BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    attn_win = 'attention (%s)' % hostname
    # vis.image(torchvision.transforms.ToTensor()(Image.open(buf)), win=attn_win, opts={'title': attn_win})

def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    show_plot_visdom()
    plt.show()
    plt.close()

def evaluate_and_show_attention(input_sentence):
    output_words, attentions = evaluate(input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)
    win = 'evaluted (%s)' % hostname
    # text = '<p>&gt; %s</p><p>= %s</p><p>&lt; %s</p>' % (input_sentence, target_sentence, output_sentence)
    # vis.text(text, win=win, opts={'title': win})


# Begin!
ecs = []
dcs = []
eca = 0
dca = 0
random_batch = prepare.random_batch
criterion = nn.NLLLoss()
while epoch < n_epochs:
    epoch += 1

    print("---------Epoch %d---------"%(epoch))

    # Get training data for this cycle
    input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size)
    # print("input_length:",input_lengths)
    # print("target_length:",target_lengths)
    # print("batch_size:",batch_size)

    # Run the train function
    loss, ec, dc = train(
        input_batches, input_lengths, target_batches, target_lengths,
        encoder, decoder,
        encoder_optimizer, decoder_optimizer, criterion
    )

    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss
    eca += ec
    dca += dc

    if epoch == 1:
        evaluate_randomly()
        continue

    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' \
                        % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
        print(print_summary)
        evaluate_randomly()

    if epoch % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0

        # TODO: Running average helper
        ecs.append(eca / plot_every)
        dcs.append(dca / plot_every)
        ecs_win = 'encoder grad (%s)' % hostname
        dcs_win = 'decoder grad (%s)' % hostname
        # vis.line(np.array(ecs), win=ecs_win, opts={'title': ecs_win})
        # vis.line(np.array(dcs), win=dcs_win, opts={'title': dcs_win})
        eca = 0
        dca = 0

