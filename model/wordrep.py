from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn

class WordRep(nn.Module):
    def __init__(self, data):
        super(WordRep, self).__init__()
        self.gpu = data.HP_gpu
        self.embedding_dim = data.word_emb_dim
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        self.drop = nn.Dropout(data.HP_dropout)
        if self.gpu:
            self.drop = self.drop.cuda()
            self.word_embedding = self.word_embedding.cuda()

    def forward(self, mode, word_inputs, word_seq_lengths):
        word_embeds = self.word_embedding(word_inputs)
        word_list = [word_embeds]
        word_represent = torch.cat(word_list, 2)
        word_represent = self.drop(word_represent)
        
        return word_represent