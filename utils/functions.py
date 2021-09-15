from __future__ import print_function
from __future__ import absolute_import
import sys
import numpy as np

import time
t = time.localtime()
result = time.strftime("%m-%d-%Y %H:%M:%S", t)

import logging
logging.basicConfig(
            filename=result,
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            return '0'
        else:
            new_word += char
    return new_word

def read_ner_instance(input_file, ner_alphabet, sa_alphabet, word_alphabet, number_normalized, max_sent_length):
    NER_text, NER_idx, SA_text, SA_idx = [], [], [], []
    words, ners, sas = [], [], []
    word_idx, ner_idx, sa_idx = [], [], []

    if sa_alphabet is None: #Source domain 
        for line in open(input_file, 'r').readlines():
            if len(line) > 2:
                pairs = line.strip().split(' ')
                word = normalize_word(pairs[0])
                words.append(word)
                word_idx.append(word_alphabet.get_index(word))
                ners.append(pairs[1])
                ner_idx.append(ner_alphabet.get_index(pairs[1]))
            else:
                if len(words) > 0 and len(words) < max_sent_length:                    
                    NER_text.append([words, ners])
                    NER_idx.append([word_idx, ner_idx])
                words, ners = [], []
                word_idx, ner_idx = [], []
    else:
        for line in open(input_file, 'r').readlines():
            if len(line) > 2:
                pairs = line.strip().split(' ')
                word = normalize_word(pairs[0])
                words.append(word)
                word_idx.append(word_alphabet.get_index(word))
                ners.append(pairs[1])
                ner_idx.append(ner_alphabet.get_index(pairs[1]))
                sas.append(pairs[2])
                sa_idx.append(sa_alphabet.get_index(pairs[2]))
            else:
                if len(words) > 0 and len(words) < max_sent_length:
                    NER_text.append([words, ners])
                    NER_idx.append([word_idx, ner_idx])
                    SA_text.append([words, sas])
                    SA_idx.append([word_idx, sa_idx])
                words, ners, sas = [], [], []
                word_idx, ner_idx, sa_idx = [], [], []

    return NER_text, NER_idx, SA_text, SA_idx

def read_lm_instance(input_file, word_alphabet, number_normalized, max_sent_length):
    instance_text, instance_idx = [],[]    
    words, ners = [],[]
    word_idx, ner_idx = [], []
    for line in open(input_file, 'r').readlines():
        pairs = line.strip().split()
        if len(pairs) < 1:
            continue
        for word in pairs:
            word = normalize_word(word)
            label = '\0'
            words.append(word)
            word_idx.append(word_alphabet.get_index(word))
            ners.append(label)
            ner_idx.append(1)
        if len(words) > 0 and len(words) < max_sent_length:
            instance_text.append([words, ners])
            instance_idx.append([word_idx, ner_idx])
        words, ners = [], []
        word_idx, ner_idx = [], []
    return instance_text, instance_idx

def build_pretrain_embedding(embedding_path, word_alphabet, embed_dim=100, norm=True):
    embed_dict = dict()
    if embedding_path is not None:
        embed_dict, embed_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embed_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embed_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embed_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embed_dict[word])
            else:
                pretrain_emb[index, :] = embed_dict[word]
            perfect_match += 1
        elif word.lower() in embed_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embed_dict[word.lower()])
            else:
                pretrain_emb[index, :] = embed_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embed_dim])
            not_match += 1
    return pretrain_emb, embed_dim

def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens)),'%s %s %s'% (embedd_dim + 1, len(tokens), line)
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim
