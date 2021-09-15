from __future__ import print_function

import time
import logging
t = time.localtime()
result = time.strftime("%m-%d-%Y %H:%M:%S", t)
logging.basicConfig(filename=str(result),
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import argparse
import random
import math
import copy
import numpy as np

import torch
from torch.autograd import Variable
from torch import optim

from utils import conlleval
from model.seqmodel import SeqModel
from utils.data import Data

try:
    import cPickle as pickle
except ImportError:
    import pickle

logger.info('Oo'*30)

DWA = True

seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)
logger.info('random seed: %s' % seed_num)

def count_PRF(n_perfect, n_miss, n_over, mode, Print = False):
    total = n_perfect + n_miss
    true_pos = n_perfect
    false_neg = n_miss
    false_pos = n_over
    p = true_pos / (true_pos + false_pos) if (true_pos + false_pos) != 0 else 0
    r = true_pos / total if total != 0 else 0
    f = (2 * p * r) / (p + r) if (p + r) != 0 else 0
    logger.info(mode + " p {0:.4f}, r {1:.4f}, f1 {2:.4f}".format(p, r, f))
    return f

def count_Result(pred_word, true_word):
    perfect, partial, over = [], [], []
    n_perfect, n_miss, n_over = 0, 0, 0
    for pred_ens in pred_word:
        if pred_ens in true_word:
            n_perfect += 1
            perfect.append(pred_ens)
            true_word.remove(pred_ens)
    for en in perfect:
        pred_word.remove(en)

    perfect, partial, over = [], [], []    
    for true_ens in true_word:
        if true_ens in pred_word:
            n_perfect += 1
            perfect.append(true_ens)
            pred_word.remove(true_ens)
    for en in perfect:
        true_word.remove(en)

    if len(pred_word) != 0:
        n_over += len(pred_word)
    if len(true_word) != 0:
        n_miss += len(true_word)
    return n_perfect, n_miss, n_over

def CompleteMatch(pred_results, gold_results):
    pos, neg, neu = {}, {}, {}
    pos_pred_word, neg_pred_word, neu_pred_word = [], [], []
    isPos, isNeg, isNeu = False, False, False
    for index in range(len(pred_results)):
        if pred_results[index] == 'Pos' and not isNeg and not isNeu:
            pos[index] = pred_results[index]
            isPos = True
        
        elif pred_results[index] == 'Neg' and not isPos and not isNeu:
            neg[index] = pred_results[index]
            isNeg = True
        
        elif pred_results[index] == 'Neu' and not isPos and not isNeg:
            neu[index] = pred_results[index]
            isNeu = True
        
        elif pred_results[index] == 'None':
            if isPos and not isNeg and not isNeu:
                pos_pred_word.append(pos)
                pos = {}
                isPos = False
            if isNeg and not isPos and not isNeu:
                neg_pred_word.append(neg)
                neg = {}
                isNeg = False
            if isNeu and not isPos and not isNeg:
                neu_pred_word.append(neu)
                neu = {}
                isNeu = False
        
        if index == len(pred_results[index])-1:
            if isPos and not isNeg and not isNeu:
                pos_pred_word.append(pos)
                pos = {}
                isPos = False
            if isNeg and not isPos and not isNeu:
                neg_pred_word.append(neg)
                neg = {}
                isNeg = False
            if isNeu and not isPos and not isNeg:
                neu_pred_word.append(neu)
                neu = {}
                isNeu = False
    #true_sen
    pos, neg, neu = {}, {}, {}
    pos_true_word, neg_true_word, neu_true_word = [], [], []
    isPos, isNeg, isNeu = False, False, False
    for index in range(len(gold_results)):    
        if gold_results[index] == 'Pos':    
            pos[index] = gold_results[index]
            isPos = True
        
        elif gold_results[index] == 'Neg':
            neg[index] = gold_results[index]
            isNeg = True
        
        elif gold_results[index] == 'Neu':
            neu[index] = gold_results[index]
            isNeu = True
        
        elif gold_results[index] == 'None':
            if isPos and not isNeg and not isNeu:
                pos_true_word.append(pos)
                pos = {}
                isPos = False
            if isNeg and not isPos and not isNeu:
                neg_true_word.append(neg)
                neg = {}
                isNeg = False
            if isNeu and not isPos and not isNeg:
                neu_true_word.append(neu)
                neu = {}
                isNeu = False
        
        if index == len(gold_results[index])-1:
            if isPos and not isNeg and not isNeu:
                pos_true_word.append(pos)
                pos = {}
                isPos = False
            if isNeg and not isPos and not isNeu:
                neg_true_word.append(neg)
                neg = {}
                isNeg = False
            if isNeu and not isPos and not isNeg:
                neu_true_word.append(neu)
                neu = {}
                isNeu = False
    return pos_pred_word, neg_pred_word, neu_pred_word, pos_true_word, neg_true_word, neu_true_word

def SentimentEvaluate(mode, data_instance, label_alphabet, data, model, idx, dataset):
    model.eval()
    pred_results, gold_results = [], []
    for batch_id in range(len(data_instance) // data.HP_batch_size + 1):
        instance = data_instance[batch_id * data.HP_batch_size: (batch_id + 1) * data.HP_batch_size \
            if (batch_id + 1) * data.HP_batch_size < len(data_instance) else len(data_instance)]

        if not instance:
            continue
        instance_batch_data = batchify_with_label(instance, data.HP_gpu, True)
        tag_seq = model(mode, instance_batch_data[0], instance_batch_data[1], instance_batch_data[8])

        pred_label, gold_label = recover_label(tag_seq, instance_batch_data[6], instance_batch_data[8],
                                               label_alphabet, instance_batch_data[2])
        pred_results += pred_label
        gold_results += gold_label
    
    assert len(pred_results) == len(gold_results)

    pos_pred_word, neg_pred_word, neu_pred_word, \
    pos_true_word, neg_true_word, neu_true_word = CompleteMatch(pred_results, gold_results)
    n_perfect, n_miss, n_over = count_Result(pos_pred_word, pos_true_word)
    pos_F = count_PRF(n_perfect, n_miss, n_over, 'pos', True)
    n_perfect, n_miss, n_over = count_Result(neg_pred_word, neg_true_word)
    neg_F = count_PRF(n_perfect, n_miss, n_over, 'neg', True)
    n_perfect, n_miss, n_over = count_Result(neu_pred_word, neu_true_word)
    neu_F = count_PRF(n_perfect, n_miss, n_over, 'neu', True)
    AVG_F = (pos_F + neg_F + neu_F) / 3
    return AVG_F

def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label, gold_label = [], []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert (len(pred) == len(gold))
        pred_label.extend(pred)
        gold_label.extend(gold)
    
    return pred_label, gold_label

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def SingerEvaluate(mode, data_instance, label_alphabet, data, model, idx, dataset):
    model.eval()
    pred_results = []
    gold_results = []
    for batch_id in range(len(data_instance) // data.HP_batch_size + 1):
        instance = data_instance[batch_id * data.HP_batch_size: (batch_id + 1) * data.HP_batch_size \
            if (batch_id + 1) * data.HP_batch_size < len(data_instance) else len(data_instance)]

        if not instance:
            continue
        instance_batch_data = batchify_with_label(instance, data.HP_gpu, True)
        tag_seq = model(mode, instance_batch_data[0], instance_batch_data[1], instance_batch_data[8])

        pred_label, gold_label = recover_label(tag_seq, instance_batch_data[6], instance_batch_data[8],
                                               label_alphabet, instance_batch_data[2])
        pred_results += pred_label
        gold_results += gold_label

    assert len(pred_results) == len(gold_results)
    n_perfect, n_miss, n_over, f = 0, 0, 0, 0
    predSTART,trueSTART = False,False
    pred_word, true_word = [], []
    p, t = [], []
    
    for index in range(len(pred_results)):
        #pred_sen
        if pred_results[index] == 'I-S':
            predSTART = True
            p.append(index)
        
        if pred_results[index] == 'O-S' and predSTART:
            p.append(index)
        
        if pred_results[index] == 'I-O' or pred_results[index] == 'O-O':
            if predSTART:
                pred_word.append(p)
                p = []
                predSTART = False
        
        if index == len(pred_results) - 1 and predSTART:
            pred_word.append(p)
            p = []
            predSTART = False

        #true_sen
        if gold_results[index] == 'I-S':
            trueSTART = True
            t.append(index)
        
        if gold_results[index] == 'O-S' and trueSTART:
            t.append(index)
        
        if gold_results[index] == 'I-O' or gold_results[index] == 'O-O':
            if trueSTART:
                true_word.append(t)
                t = []
                trueSTART = False
        
        if index == len(gold_results)-1 and trueSTART:
            true_word.append(t)
            t = []
            trueSTART = False

    n_perfect, n_miss, n_over = count_Result(pred_word, true_word)
    f = count_PRF(n_perfect, n_miss, n_over, 'Singer', True)

    return f

def ApplianceEvaluate(mode, data_instance, label_alphabet, data, model, idx, dataset):
    model.eval()
    pred_results = []
    gold_results = []
    for batch_id in range(len(data_instance) // data.HP_batch_size + 1):
        instance = data_instance[batch_id * data.HP_batch_size: (batch_id + 1) * data.HP_batch_size \
            if (batch_id + 1) * data.HP_batch_size < len(data_instance) else len(data_instance)]

        if not instance:
            continue
        instance_batch_data = batchify_with_label(instance, data.HP_gpu, True)
        tag_seq = model(mode, instance_batch_data[0], instance_batch_data[1], instance_batch_data[8])

        pred_label, gold_label = recover_label(tag_seq, instance_batch_data[6], instance_batch_data[8],
                                               label_alphabet, instance_batch_data[2])
        pred_results += pred_label
        gold_results += gold_label

    assert len(pred_results) == len(gold_results)
    
    n_perfect, n_miss, n_over = 0, 0, 0
    predSTART,trueSTART = False,False
    pred_word, true_word = [], []
    p, t = [], []
    
    for index in range(len(pred_results)):
        #pred_sen
        if pred_results[index] == 'I-PN':
            predSTART = True
            p.append(index)
        
        if pred_results[index] == 'O-PN' and predSTART:
            p.append(index)
        
        if pred_results[index] == 'I-O' or pred_results[index] == 'O-O':
            if predSTART:
                pred_word.append(p)
                p = []
                predSTART = False
        
        if index == len(pred_results) - 1 and predSTART:
            pred_word.append(p)
            p = []
            predSTART = False

        #true_sen
        if gold_results[index] == 'I-PN':
            trueSTART = True
            t.append(index)
        
        if gold_results[index] == 'O-PN' and trueSTART:
            t.append(index)
        
        if gold_results[index] == 'I-O' or gold_results[index] == 'O-O':
            if trueSTART:
                true_word.append(t)
                t = []
                trueSTART = False
        
        if index == len(gold_results)-1 and trueSTART:
            true_word.append(t)
            t = []
            trueSTART = False

    n_perfect, n_miss, n_over = count_Result(pred_word, true_word)
    f_PN = count_PRF(n_perfect, n_miss, n_over, 'PN', True)

    n_perfect, n_miss, n_over = 0, 0, 0
    predSTART,trueSTART = False,False
    pred_word, true_word = [], []
    p, t = [], []
    
    for index in range(len(pred_results)):
        #pred_sen
        if pred_results[index] == 'I-PB':
            predSTART = True
            p.append(index)
        
        if pred_results[index] == 'O-PB' and predSTART:
            p.append(index)
        
        if pred_results[index] == 'I-O' or pred_results[index] == 'O-O':
            if predSTART:
                pred_word.append(p)
                p = []
                predSTART = False
        
        if index == len(pred_results) - 1 and predSTART:
            pred_word.append(p)
            p = []
            predSTART = False

        #true_sen
        if gold_results[index] == 'I-PB':
            trueSTART = True
            t.append(index)
        
        if gold_results[index] == 'O-PB' and trueSTART:
            t.append(index)
        
        if gold_results[index] == 'I-O' or gold_results[index] == 'O-O':
            if trueSTART:
                true_word.append(t)
                t = []
                trueSTART = False
        
        if index == len(gold_results)-1 and trueSTART:
            true_word.append(t)
            t = []
            trueSTART = False

    n_perfect, n_miss, n_over = count_Result(pred_word, true_word)
    f_PB = count_PRF(n_perfect, n_miss, n_over, 'PB', True)
    return f_PN, f_PB

def batchify_with_label(input_batch_list,  gpu, volatile_flag=False):
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    labels = [sent[1] for sent in input_batch_list]

    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()
    
    mask = Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).byte()
    word_seq_tensor = Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    label_seq_tensor = Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    lm_forward_seq_tensor = Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    lm_backward_seq_tensor = Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    
    for idx, (seq, label, seq_len) in enumerate(zip(words, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
        if seq_len > 1:
            lm_forward_seq_tensor[idx, 0: seq_len - 1] = word_seq_tensor[idx, 1: seq_len]
            lm_forward_seq_tensor[idx, seq_len - 1] = torch.LongTensor([1])  # unk word
            lm_backward_seq_tensor[idx, 1: seq_len] = word_seq_tensor[idx, 0: seq_len - 1]
            lm_backward_seq_tensor[idx, 0] = torch.LongTensor([1])  # unk word
        else:
            lm_forward_seq_tensor[idx, 0] = torch.LongTensor([1])  # unk word
            lm_backward_seq_tensor[idx, 0] = torch.LongTensor([1])  # unk word
        label_seq_tensor[idx, :seq_len] = torch.LongTensor(label)
        mask[idx, :seq_len] = torch.Tensor([1] * seq_len)

    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    lm_forward_seq_tensor = lm_forward_seq_tensor[word_perm_idx]
    lm_backward_seq_tensor = lm_backward_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]

    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()

        lm_forward_seq_tensor = lm_forward_seq_tensor.cuda()
        lm_backward_seq_tensor = lm_backward_seq_tensor.cuda()
        mask = mask.cuda()
    lm_seq_tensor = [lm_forward_seq_tensor, lm_backward_seq_tensor]

    return word_seq_tensor, word_seq_lengths, word_seq_recover, \
           None, None, None,\
           label_seq_tensor, lm_seq_tensor, mask

def train(train_data):
    model = SeqModel(train_data)
    logger.info(model)
   
    optimizer = optim.SGD(model.parameters(), lr = train_data.HP_lr, 
                          weight_decay = train_data.HP_l2, momentum = train_data.HP_momentum)

    NER_PN, NER_PB, SA_Singer = [], [], []
    best_epoch_PN, best_epoch_PB, best_epoch_Singer_SA = 0, 0, 0
    best_test_PN, best_test_PB, best_test_Singer_SA = 0, 0, 0

    NER_Singer, SA_Pro = [], []
    best_epoch_Singer, best_epoch_Product = 0, 0
    best_test_Singer, best_test_Product = 0, 0

    losses, lossA, lossB = [], [], []
    
    lambda_weight = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1),\
                    random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]\
                    if DWA else [0.8, 1, 0.5, 0.5, 1, 1]

    for idx in range(train_data.HP_iteration):
        epoch_start = time.time()
        logger.info("Epoch: %s/%s" % (idx, train_data.HP_iteration))
        print("Epoch: %s/%s" % (idx, train_data.HP_iteration))

        optimizer = lr_decay(optimizer, idx, train_data.HP_lr_decay, train_data.HP_lr)

        random.shuffle(train_data.ner_1_train_idx)
        random.shuffle(train_data.ner_2_train_idx)
        
        random.shuffle(train_data.lm_1_idx)
        random.shuffle(train_data.lm_2_idx)

        random.shuffle(train_data.sa_1_train_idx)
        random.shuffle(train_data.sa_2_train_idx)

        model.train()
        model.zero_grad()

        ner_1_loss, ner_2_loss, sa_1_loss, sa_2_loss = 0, 0, 0, 0
        lm_1_perplexity,lm_2_perplexity = 0, 0
        
        ner_1_batch_size = train_data.HP_batch_size
        batch_nums = len(train_data.ner_1_train_idx) // ner_1_batch_size
        ner_2_batch_size = len(train_data.ner_2_train_idx) // batch_nums

        lm_1_batch_size = len(train_data.lm_1_idx) // batch_nums
        lm_2_batch_size = len(train_data.lm_2_idx) // batch_nums

        if DWA:
            T = 0.5
            if  idx > 1:
                w_0 = lossB[0] / lossA[0]
                w_1 = lossB[1] / lossA[1]
                w_2 = lossB[2] / lossA[2]
                w_3 = lossB[3] / lossA[3]
                w_4 = lossB[4] / lossA[4]
                w_5 = lossB[5] / lossA[5]

                SUM = np.exp(w_0 / T) + np.exp(w_2 / T) + np.exp(w_4 / T) + np.exp(w_1 / T) + np.exp(w_3 / T) + np.exp(w_5 / T)

                lambda_weight[0] = 6*np.exp(w_0 / T) / SUM
                lambda_weight[1] = 6*np.exp(w_1 / T) / SUM
                lambda_weight[2] = 6*np.exp(w_2 / T) / SUM
                lambda_weight[3] = 6*np.exp(w_3 / T) / SUM
                lambda_weight[4] = 6*np.exp(w_4 / T) / SUM
                lambda_weight[5] = 6*np.exp(w_5 / T) / SUM

        for batch_id in range(batch_nums):
            ner_1_data = train_data.ner_1_train_idx[batch_id * ner_1_batch_size: (batch_id + 1) * ner_1_batch_size \
            if(batch_id + 1) * ner_1_batch_size < len(train_data.ner_1_train_idx) else len(train_data.ner_1_train_idx)]

            ner_2_data = train_data.ner_2_train_idx[batch_id * ner_2_batch_size: (batch_id + 1) * ner_2_batch_size \
            if(batch_id + 1) * ner_2_batch_size < len(train_data.ner_2_train_idx) else len(train_data.ner_2_train_idx)]

            lm_1_data = train_data.lm_1_idx[batch_id * lm_1_batch_size: (batch_id + 1) * lm_1_batch_size \
            if(batch_id + 1) * lm_1_batch_size < len(train_data.lm_1_idx) else len(train_data.lm_1_idx)]
            
            lm_2_data = train_data.lm_2_idx[batch_id * lm_2_batch_size: (batch_id + 1) * lm_2_batch_size \
            if(batch_id + 1) * lm_2_batch_size < len(train_data.lm_2_idx) else len(train_data.lm_2_idx)]
            
            sa_1_data = train_data.sa_1_train_idx[batch_id * ner_1_batch_size: (batch_id + 1) * ner_1_batch_size \
            if(batch_id + 1) * ner_1_batch_size < len(train_data.sa_1_train_idx) else len(train_data.sa_1_train_idx)]
            
            sa_2_data = train_data.sa_2_train_idx[batch_id * ner_2_batch_size: (batch_id + 1) * ner_2_batch_size \
            if(batch_id + 1) * ner_2_batch_size < len(train_data.sa_2_train_idx) else len(train_data.sa_2_train_idx)]
            
            ner_1_batch_data = batchify_with_label(ner_1_data, train_data.HP_gpu)
            ner_2_batch_data = batchify_with_label(ner_2_data, train_data.HP_gpu)

            lm_1_batch_data = batchify_with_label(lm_1_data, train_data.HP_gpu)
            lm_2_batch_data = batchify_with_label(lm_2_data, train_data.HP_gpu)
            
            sa_1_batch_data = batchify_with_label(sa_1_data, train_data.HP_gpu)
            sa_2_batch_data = batchify_with_label(sa_2_data, train_data.HP_gpu)

            model_loss = 0
            
            if idx / 2 == 0:
                lossA = []
                loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq = \
                    model.loss('ner1', ner_1_batch_data[0], ner_1_batch_data[1], ner_1_batch_data[6], ner_1_batch_data[7], ner_1_batch_data[8])
                lossA.append(loss)
                loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq = \
                    model.loss('ner2', ner_2_batch_data[0], ner_2_batch_data[1],  ner_2_batch_data[6], ner_2_batch_data[7], ner_2_batch_data[8])
                lossA.append(loss)
                
                loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq = \
                    model.loss('lm1', lm_1_batch_data[0], lm_1_batch_data[1],  lm_1_batch_data[6], lm_1_batch_data[7], lm_1_batch_data[8])
                lossA.append(loss)
                loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq = \
                    model.loss('lm2', lm_2_batch_data[0], lm_2_batch_data[1],  lm_2_batch_data[6], lm_2_batch_data[7], lm_2_batch_data[8])
                lossA.append(loss)

                loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq = \
                     model.loss('sa1', sa_1_batch_data[0], sa_1_batch_data[1],  sa_1_batch_data[6], sa_1_batch_data[7], sa_1_batch_data[8])
                lossA.append(loss)
                loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq = \
                     model.loss('sa2', sa_2_batch_data[0], sa_2_batch_data[1],  sa_2_batch_data[6], sa_2_batch_data[7], sa_2_batch_data[8])
                lossA.append(loss)

                for i in range(len(lossA)):
                    model_loss += lossA[i] * float(lambda_weight[i])

            else:
                lossB = []
                loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq = \
                    model.loss('ner1', ner_1_batch_data[0], ner_1_batch_data[1], ner_1_batch_data[6], ner_1_batch_data[7], ner_1_batch_data[8])
                lossB.append(loss)
                loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq = \
                    model.loss('ner2', ner_2_batch_data[0], ner_2_batch_data[1],  ner_2_batch_data[6], ner_2_batch_data[7], ner_2_batch_data[8])
                lossB.append(loss)
                
                loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq = \
                    model.loss('lm1', lm_1_batch_data[0], lm_1_batch_data[1],  lm_1_batch_data[6], lm_1_batch_data[7], lm_1_batch_data[8])
                lossB.append(loss)
                loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq = \
                    model.loss('lm2', lm_2_batch_data[0], lm_2_batch_data[1],  lm_2_batch_data[6], lm_2_batch_data[7], lm_2_batch_data[8])
                lossB.append(loss)

                loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq = \
                     model.loss('sa1', sa_1_batch_data[0], sa_1_batch_data[1],  sa_1_batch_data[6], sa_1_batch_data[7], sa_1_batch_data[8])
                lossB.append(loss)
                loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq = \
                     model.loss('sa2', sa_2_batch_data[0], sa_2_batch_data[1],  sa_2_batch_data[6], sa_2_batch_data[7], sa_2_batch_data[8])
                lossB.append(loss)

                for i in range(len(lossB)):
                    model_loss += lossB[i] * float(lambda_weight[i])

            model_loss.backward()
            optimizer.step()
            model.zero_grad()
        
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
                
        logger.info('lambda_weight: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f'%(lambda_weight[0],lambda_weight[1],lambda_weight[2],lambda_weight[3],lambda_weight[4],lambda_weight[5]))

        # source
        f_PN, f_PB = ApplianceEvaluate('ner1', train_data.ner_1_test_idx, train_data.label_alphabet_ner_1, train_data, model, idx, 'test')
        NER_PN.append(f_PN)
        NER_PB.append(f_PB)

        f_Product = SentimentEvaluate('sa1', train_data.sa_1_train_idx, train_data.label_alphabet_sa_1, train_data, model, idx, 'test')
        SA_Pro.append(f_Product)

        # target
        f_Singer = SingerEvaluate('ner2', train_data.ner_2_test_idx, train_data.label_alphabet_ner_2, train_data, model, idx, 'test')
        NER_Singer.append(f_Singer)

        f_Singer_SA = SentimentEvaluate('sa2', train_data.sa_2_train_idx, train_data.label_alphabet_sa_2, train_data, model, idx, 'test')
        SA_Singer.append(f_Singer_SA)

        # NER
        if f_PN > best_test_PN:
            best_epoch_PN = idx
            best_test_PN = f_PN
        if f_PB > best_test_PB:
            best_epoch_PB = idx
            best_test_PB = f_PB

        if f_Singer > best_test_Singer:
            best_epoch_Singer = idx
            best_test_Singer = f_Singer

        # SA
        if f_Product > best_test_Product:
            best_epoch_Product = idx
            best_test_Product = f_Product

        if f_Singer_SA > best_test_Singer_SA:
            best_epoch_Singer_SA = idx
            best_test_Singer_SA = f_Singer_SA

    logger.info("the best PN NER test score is in epoch %s, test:%.4f" % (best_epoch_PN, NER_PN[best_epoch_PN]))
    logger.info("the best PB NER test score is in epoch %s, test:%.4f" % (best_epoch_PB, NER_PB[best_epoch_PB]))
    logger.info("the best Singer NER test score is in epoch %s, test:%.4f" % (best_epoch_Singer, NER_Singer[best_epoch_Singer]))

    # SA
    logger.info("the best Product SA test score is in epoch %s, test:%.4f" % (best_epoch_Product, SA_Pro[best_epoch_Product]))
    logger.info("the best Singer SA test score is in epoch %s, test:%.4f" % (best_epoch_Singer_SA, SA_Singer[best_epoch_Singer_SA]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cross ner via cross language model')
    parser.add_argument('--config', help='configuration File')
    args = parser.parse_args()

    supervised_data = Data()
    supervised_data.HP_gpu = torch.cuda.is_available()
    print('use GPU: %s' % supervised_data.HP_gpu)

    supervised_data.read_config(args.config)
    status = supervised_data.status.lower()

    supervised_data.build_language_model_alphabet(supervised_data.supervised_lm_1_train,
                                                  supervised_data.supervised_lm_2_train)
    
    supervised_data.build_alphabet(supervised_data.supervised_ner_1_train,
                                   supervised_data.supervised_ner_2_train, 'train')

    supervised_data.build_alphabet(supervised_data.supervised_ner_1_test,
                                   supervised_data.supervised_ner_2_test, 'test')

    supervised_data.build_task_domain_alphabet()
    supervised_data.fix_alphabet()
    
    supervised_data.generate_instance()

    supervised_data.build_pretrain_emb()

    supervised_data.show_data_summary()
    
    train(supervised_data)