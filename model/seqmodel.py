# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Xiaobo Liang and Chen Jia
# @Last Modified time: 2019-05-20

from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wordsequence import WordSequence
from .crf import CRF
from .sampled_softmax_loss_formal import SampledSoftmaxLoss


class SeqModel(nn.Module):
    def __init__(self, data):
        super(SeqModel, self).__init__()
        self.bilstm_flag = data.HP_bilstm
        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss

        label_size_1 = data.label_alphabet_ner_1_size
        label_size_2 = data.label_alphabet_ner_2_size
        self.ner_1_crf = CRF(label_size_1, self.gpu)
        self.ner_2_crf = CRF(label_size_2, self.gpu)

        data.label_alphabet_ner_1_size += 2
        data.label_alphabet_ner_2_size += 2

        self.word_hidden = WordSequence(data)
        self.lstm_hidden = data.HP_hidden_dim // 2 if self.bilstm_flag else data.HP_hidden_dim
        
        self.hidden2tag_ner_1 = nn.Linear(data.HP_hidden_dim, data.label_alphabet_ner_1_size)
        self.hidden2tag_ner_2 = nn.Linear(data.HP_hidden_dim, data.label_alphabet_ner_2_size)
        self.hidden2tag_sa_1 = nn.Linear(data.HP_hidden_dim, data.label_alphabet_sa_1_size)
        self.hidden2tag_sa_2 = nn.Linear(data.HP_hidden_dim, data.label_alphabet_sa_2_size)

        self.lm_1_sampled_loss = SampledSoftmaxLoss(num_words=data.word_alphabet_size, embedding_dim=self.lstm_hidden,
                                                    num_samples=50, sparse=False, gpu=self.gpu)
        self.lm_2_sampled_loss = SampledSoftmaxLoss(num_words=data.word_alphabet_size, embedding_dim=self.lstm_hidden,
                                                    num_samples=50, sparse=False, gpu=self.gpu)
        self.lm_2_sampled_loss = self.lm_1_sampled_loss


        if self.gpu:
            self.hidden2tag_ner_1 = self.hidden2tag_ner_1.cuda()
            self.hidden2tag_ner_2 = self.hidden2tag_ner_2.cuda()
            self.hidden2tag_sa_1 = self.hidden2tag_sa_1.cuda()            
            self.hidden2tag_sa_2 = self.hidden2tag_sa_2.cuda()

            self.lm_1_sampled_loss = self.lm_1_sampled_loss.cuda()
            self.lm_2_sampled_loss = self.lm_2_sampled_loss.cuda()

    def loss(self, mode, word_inputs, word_seq_lengths, batch_label, lm_seq_tensor, mask):
        if self.bilstm_flag:
            outs_forward, outs_backward, outs = self.word_hidden(mode, word_inputs, word_seq_lengths)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)

        loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq = None, None, None, None, None
        
        if 'lm' in mode:
            loss, perplexity, tag_seq_forward, tag_seq_backward = \
                self.language_model_loss(mode, outs_forward, outs_backward, batch_size, seq_len, lm_seq_tensor[0], lm_seq_tensor[1], mask)
        else: 
            loss, tag_seq = self.neg_log_likelihood_loss(mode, outs, batch_size, seq_len, batch_label, mask)

        return loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq

    def neg_log_likelihood_loss(self, mode, outs, batch_size, seq_len, batch_label, mask):
        if mode == 'ner1':
            outs = self.hidden2tag_ner_1(outs)
            crf = self.ner_1_crf
            total_loss = crf.neg_log_likelihood_loss(outs, mask, batch_label)
            scores, tag_seq = crf.viterbi_decode(outs, mask)
        
        elif mode == 'ner2':
            outs = self.hidden2tag_ner_2(outs)
            crf = self.ner_2_crf
            total_loss = crf.neg_log_likelihood_loss(outs, mask, batch_label)
            scores, tag_seq = crf.viterbi_decode(outs, mask)
        
        elif mode == 'sa1':
            outs = self.hidden2tag_sa_1(outs)
            loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
            outs = outs.view(batch_size * seq_len, -1)
            score = F.log_softmax(outs, 1)
            total_loss = loss_function(score, batch_label.view(batch_size * seq_len))
            _, tag_seq = torch.max(score, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)

        elif mode == 'sa2':
            outs = self.hidden2tag_sa_2(outs)
            loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
            outs = outs.view(batch_size * seq_len, -1)
            score = F.log_softmax(outs, 1)
            total_loss = loss_function(score, batch_label.view(batch_size * seq_len))
            _, tag_seq = torch.max(score, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)

        if self.average_batch:
            total_loss = total_loss / batch_size if batch_size != 0 else 0

        return total_loss, tag_seq

    def language_model_loss(self, mode, outs_forward, outs_backward, batch_size, seq_len, lm_forward_seq_tensor,
                            lm_backward_seq_tensor, mask):

        if self.bilstm_flag:
            softmax_lm = self.lm_1_sampled_loss if mode == 'lm1' else self.lm_2_sampled_loss

            losses = []
            for idx, embedding, targets in ((0, outs_forward, lm_forward_seq_tensor),
                                            (1, outs_backward, lm_backward_seq_tensor)):
                non_masked_targets = targets.masked_select(mask) - 1
                non_masked_embedding = embedding.masked_select(mask.unsqueeze(-1)).view(-1, self.lstm_hidden)
                losses.append(softmax_lm(non_masked_embedding, non_masked_targets))
            total_loss = 0.5 * (losses[0] + losses[1])

            tag_seq_forward, tag_seq_backward = None, None
            length_mask = torch.sum(mask.float(), dim=1).float()
            num = length_mask.sum(0).data[0]
        
            perplexity = total_loss / num if num != 0 else 0.0

            if self.average_batch:
                total_loss = total_loss / batch_size if batch_size != 0 else 0

            return total_loss, perplexity, tag_seq_forward, tag_seq_backward

    def forward(self, mode, word_inputs, word_seq_lengths, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)

        _, _, outs = self.word_hidden(mode, word_inputs, word_seq_lengths)

        if mode == 'ner1':
            outs = self.hidden2tag_ner_1(outs)
            crf = self.ner_1_crf
            scores, tag_seq = crf.viterbi_decode(outs, mask)

        elif mode == 'ner2':
            outs = self.hidden2tag_ner_2(outs)
            crf = self.ner_2_crf
            scores, tag_seq = crf.viterbi_decode(outs, mask)
        
        elif mode == 'sa1':
            outs = self.hidden2tag_sa_1(outs)
            outs = outs.view(batch_size * seq_len, -1)
            _, tag_seq = torch.max(outs, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            tag_seq = mask.long() * tag_seq
        
        elif mode == 'sa2':
            outs = self.hidden2tag_sa_2(outs)
            outs = outs.view(batch_size * seq_len, -1)
            _, tag_seq = torch.max(outs, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            tag_seq = mask.long() * tag_seq

        return tag_seq