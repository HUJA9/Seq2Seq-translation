# -*- coding=utf-8 -*-
import unicodedata
import string
import re
import random
import time
import datetime
import math
import socket

hostname = socket.gethostname()

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence  # , masked_cross_entropy
from masked_cross_entropy import *

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

USE_CUDA = False
SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count=3):
        keep = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep.append(k)

        print('total', len(self.word2index))
        print('keep', len(keep))
        print('keep %', len(keep) / len(self.word2index))

        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

        for word in keep:
            self.index_word(word)


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    #     filename = '../data/%s-%s.txt' % (lang1, lang2)
    filename = 'data/%s-%s.txt' % (lang1, lang2)
    print("filename:", filename)
    lines = open(filename, encoding='utf-8').read().strip().split('\n')  # encoding utf-8

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


MIN_LENGTH = 5
MAX_LENGTH = 20

good_prefixes = (
    "i ", "he  ", "she ", "you ", "they ", "we "
)


def filter_pair(p):
    return len(p[1].split(' ')) <= MAX_LENGTH and len(p[2].split(' ')) <= MAX_LENGTH and len(
        p[1].split(' ')) >= MIN_LENGTH and len(p[2].split(' ')) >= MIN_LENGTH  # and \


#         p[1].startswith(good_prefixes)

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1_name, lang2_name, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1_name, lang2_name, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Indexing words...")
    for pair in pairs:
        input_lang.index_words(pair[1])
        output_lang.index_words(pair[2])

    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)
input_lang.trim()
output_lang.trim()

keep_pairs = []

for pair in pairs:
    keep_input = True
    keep_output = True

    for word in pair[1].split(' '):
        if word not in input_lang.word2index:
            keep_input = False
            break

    for word in pair[2].split(' '):
        if word not in output_lang.word2index:
            keep_output = False
            break

    if keep_input and keep_output:
        keep_pairs.append(pair)

print(len(pairs))
print(len(keep_pairs))
print(len(keep_pairs) / len(pairs))
pairs = keep_pairs


# Return a list of indexes, one for each word in the sentence
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def pad_seq(seq, max_length):
    seq += [0 for i in range(max_length - len(seq))]
    return seq


def random_batch(batch_size=3):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        # print("Pair_in_random_batch:",pair)
        input_seqs.append(indexes_from_sentence(input_lang, pair[1]))  # pair[0]->pair[1]
        target_seqs.append(indexes_from_sentence(output_lang, pair[2]))  # pair[1]->pair[2]

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[1]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch x seq) tensors, transpose into (seq x batch)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    return input_var, input_lengths, target_var, target_lengths


random_batch()
