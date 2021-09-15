from __future__ import print_function
from __future__ import absolute_import
from .alphabet import Alphabet
from .functions import *

import time
t = time.localtime()
result = time.strftime("%m-%d-%Y %H:%M:%S", t)

import logging
logging.basicConfig(
            filename=result,
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle

class Data:
    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 128
        self.MAX_WORD_LENGTH = -1

        self.number_normalized = True
        self.norm_word_emb = False

        self.task_alphabet = Alphabet('task')
        self.domain_alphabet = Alphabet('domain')

        self.seg = True
        self.Ren = False
        self.HIGHWAY = False
        self.MESSAGE = False
        self.W1, self.W2, self.W3, self.W4 = 0, 0, 0, 0

        # supervised learning
        self.supervised_ner_1_train, self.supervised_ner_1_test = None, None
        self.supervised_ner_2_train, self.supervised_ner_2_test = None, None
        self.supervised_lm_1_train,  self.supervised_lm_2_train = None, None

        self.word_emb_dir = None

        self.ner_1_train_text, self.ner_1_test_text = [], []
        self.ner_2_train_text, self.ner_2_test_text = [], []

        self.sa_1_train_text, self.sa_1_test_text = [], []
        self.sa_2_train_text, self.sa_2_test_text = [], []

        self.lm_1_text, self.lm_2_text = [], []

        self.ner_1_train_idx, self.ner_1_test_idx = [], []
        self.ner_2_train_idx, self.ner_2_test_idx = [], []

        self.sa_1_train_idx, self.sa_1_test_idx = [], []
        self.sa_2_train_idx, self.sa_2_test_idx = [], []

        self.lm_1_idx, self.lm_2_idx = [], []

        self.word_alphabet = Alphabet('word')

        self.label_alphabet_ner_1 = Alphabet('label', True)
        self.label_alphabet_ner_2 = Alphabet('label', True)
        self.label_alphabet_sa_1 = Alphabet('label', True)
        self.label_alphabet_sa_2 = Alphabet('label', True)

        self.pretrain_word_embedding = None

        self.word_alphabet_size = 0

        self.label_alphabet_ner_1_size,  self.label_alphabet_ner_2_size  = 0, 0
        self.label_alphabet_sa_1_size, self.label_alphabet_sa_2_size = 0, 0

        self.task_alphabet_size, self.domain_alphabet_size = 0, 0

        self.word_emb_dim = 100

        self.task_emb_dim, self.domain_emb_dim = 8, 8

        self.task_number = 3
        self.domain_number = 2

        self.ner_task_name = 'ner'
        self.sa_task_name = 'sa'
        self.lm_task_name = 'lm'

        self.domain_1_name = 'domain_1'
        self.domain_2_name = 'domain_2'

        # Networks
        self.use_ner_crf = True

        # Training
        self.average_batch_loss = False
        self.optimizer = "SGD"
        self.status = "train"
        self.mode = 'supervised'

        self.HP_cnn_layer = 4
        self.HP_iteration = 1000
        self.HP_batch_size = 30
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True
        self.HP_LM = False

        self.HP_gpu = False
        self.HP_lr = 0.015
        self.HP_lr_cpg = 0.005
        self.HP_lr_decay = 0.05
        self.HP_clip = None
        self.HP_momentum = 0
        self.HP_l2 = 1e-8

    def show_data_summary(self):
        logger.info("DATA SUMMARY START:")
        logger.info(" I/O:")
        logger.info("     MAX SENTENCE LENGTH: %s" % self.MAX_SENTENCE_LENGTH)
        logger.info("     MAX   WORD   LENGTH: %s" % self.MAX_WORD_LENGTH)
        logger.info("     Number   normalized: %s" % self.number_normalized)
        logger.info("     Word embedding  dir: %s" % self.word_emb_dir)
        
        logger.info("     train_source: ")
        logger.info("     Train  file directory: %s" % (self.supervised_ner_1_train))
        logger.info("     Test   file directory: %s" % (self.supervised_ner_1_test))
        logger.info("     LM   file directory: %s" % (self.supervised_lm_1_train))
        
        logger.info("     train_target: ")
        logger.info("     Train  file directory: %s" % (self.supervised_ner_2_train))
        logger.info("     Test   file directory: %s" % (self.supervised_ner_2_test))
        logger.info("     LM   file directory: %s"   % (self.supervised_lm_2_train))

        logger.info("     Train instance number: %s, Train instance number: %s" % (len(self.ner_1_train_text), len(self.ner_2_train_text)))
        logger.info("     Test  instance number: %s, Test  instance number: %s" % (len(self.ner_1_test_text), len(self.ner_2_test_text)))
        logger.info("     LM  instance number: %s, LM  instance number: %s" % (len(self.lm_1_text), len(self.lm_2_text)))
        
        logger.info(" " + "++" * 20)
        logger.info(" Training:")
        logger.info("     Optimizer: %s" % self.optimizer)
        logger.info("     Iteration: %s" % self.HP_iteration)
        logger.info("     BatchSize: %s" % self.HP_batch_size)
        logger.info('     Average  batch   loss: %s' % self.average_batch_loss)

        logger.info(" " + "++" * 20)
        logger.info(" Hyperparameters:")
        logger.info("     Hyper              lr: %s" % self.HP_lr)
        logger.info("     Hyper        lr_decay: %s" % self.HP_lr_decay)
        logger.info("     Hyper         HP_clip: %s" % self.HP_clip)
        logger.info("     Hyper        momentum: %s" % self.HP_momentum)
        logger.info("     Hyper              l2: %s" % self.HP_l2)
        logger.info("     Hyper      hidden_dim: %s" % self.HP_hidden_dim)
        logger.info("     Hyper         dropout: %s" % self.HP_dropout)
        logger.info("     Hyper      lstm_layer: %s" % self.HP_lstm_layer)
        logger.info("     Hyper          bilstm: %s" % self.HP_bilstm)
        logger.info("     Hyper             GPU: %s" % self.HP_gpu)

        logger.info(" " + "++" * 20)
        logger.info("task_emb_dim   : %s" % self.task_emb_dim)
        logger.info("domain_emb_dim : %s" % self.domain_emb_dim)
        logger.info("HIGHWAY: %s" % self.HIGHWAY)
        logger.info("MESSAGE: %s" % self.MESSAGE)
        logger.info("loss weight: [%s, %s, %s, %s]"%(self.W1, self.W2, self.W3, self.W4))

        logger.info("++" * 50)
        sys.stdout.flush()

    def build_language_model_alphabet(self, lm_1_file=None, lm_2_file=None):
        for line in open(lm_1_file).readlines():
            pairs = line.strip()
            for word in pairs:
                word = normalize_word(word)  
                self.word_alphabet.add(word)
                
        for line in open(lm_2_file).readlines():
            pairs = line.strip()
            for word in pairs:
                word = normalize_word(word)  
                self.word_alphabet.add(word)
        self.word_alphabet_size = self.word_alphabet.size()

    def build_alphabet(self, ner_1_file, ner_2_file, mode):
        for line in open(ner_1_file, 'r').readlines():
            if len(line) > 2:
                pairs = line.strip().split(' ')
                self.word_alphabet.add(normalize_word(pairs[0]))
                self.label_alphabet_ner_1.add(pairs[1])
                self.label_alphabet_sa_1.add(pairs[2])
        for line in open(ner_2_file, 'r').readlines():
            if len(line) > 2:
                pairs = line.strip().split(' ')
                self.word_alphabet.add(normalize_word(pairs[0]))    
                self.label_alphabet_ner_2.add(pairs[1])
                self.label_alphabet_sa_2.add(pairs[2])

        self.word_alphabet_size = self.word_alphabet.size()
        self.label_alphabet_ner_1_size = self.label_alphabet_ner_1.size()
        self.label_alphabet_ner_2_size = self.label_alphabet_ner_2.size()
        self.label_alphabet_sa_1_size = self.label_alphabet_sa_1.size()
        self.label_alphabet_sa_2_size = self.label_alphabet_sa_2.size()

    def filter_word_count(self):
        new_d1_vocab = Alphabet("filter_word")
        for word, index in self.word_alphabet.iteritems():
            if self.word_alphabet.get_count(word) >= 1:
                new_d1_vocab.add(word)
        self.word_alphabet = new_d1_vocab
        self.word_alphabet_size = new_d1_vocab.size()
        print("new vocab size {}".format(self.word_alphabet_size))

    def build_task_domain_alphabet(self):
        self.task_alphabet.add("ner")
        self.task_alphabet.add("sa")
        self.task_alphabet.add("lm")
        self.domain_alphabet.add("domain_1")
        self.domain_alphabet.add("domain_2")
        self.task_alphabet_size = self.task_alphabet.size()
        self.domain_alphabet_size = self.domain_alphabet.size()

    def fix_alphabet(self):
        self.word_alphabet.close()

        self.label_alphabet_ner_1.close()
        self.label_alphabet_ner_2.close()
        self.label_alphabet_sa_1.close()
        self.label_alphabet_sa_2.close()

        self.task_alphabet.close()
        self.domain_alphabet.close()

    def build_pretrain_emb(self):
        if self.word_emb_dir:
            self.pretrain_word_embedding, self.word_emb_dim =\
                build_pretrain_embedding(self.word_emb_dir, self.word_alphabet, self.word_emb_dim, self.norm_word_emb)

    def generate_instance(self):
        self.fix_alphabet()

        self.ner_1_train_text, self.ner_1_train_idx, self.sa_1_train_text, self.sa_1_train_idx = \
            read_ner_instance(self.supervised_ner_1_train, self.label_alphabet_ner_1, self.label_alphabet_sa_1, 
                              self.word_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        
        self.ner_1_test_text, self.ner_1_test_idx, self.sa_1_test_text, self.sa_1_test_idx = \
            read_ner_instance(self.supervised_ner_1_test, self.label_alphabet_ner_1, self.label_alphabet_sa_1, 
                              self.word_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        
        self.ner_2_train_text, self.ner_2_train_idx, self.sa_2_train_text, self.sa_2_train_idx = \
            read_ner_instance(self.supervised_ner_2_train, self.label_alphabet_ner_2, self.label_alphabet_sa_2, 
                              self.word_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)

        self.ner_2_test_text, self.ner_2_test_idx, self.sa_2_test_text, self.sa_2_test_idx = \
            read_ner_instance(self.supervised_ner_2_test, self.label_alphabet_ner_2, self.label_alphabet_sa_2, 
                              self.word_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)

        self.lm_1_text, self.lm_1_idx = \
            read_lm_instance(self.supervised_lm_1_train, 
                             self.word_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        self.lm_2_text, self.lm_2_idx = \
            read_lm_instance(self.supervised_lm_2_train, 
                             self.word_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)   
        
    def load(self, data_file):
        f = open(data_file, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self, save_file):
        f = open(save_file, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def read_config(self, config_file):
        config = config_file_to_dict(config_file)

        the_item = 'supervised_ner_1_train'
        if the_item in config:
            self.supervised_ner_1_train = config[the_item]

        the_item = 'supervised_ner_1_test'
        if the_item in config:
            self.supervised_ner_1_test = config[the_item]

        the_item = 'supervised_ner_2_train'
        if the_item in config:
            self.supervised_ner_2_train = config[the_item]

        the_item = 'supervised_ner_2_test'
        if the_item in config:
            self.supervised_ner_2_test = config[the_item]

        the_item = 'supervised_lm_1_train'
        if the_item in config:
            self.supervised_lm_1_train = config[the_item]

        the_item = 'supervised_lm_2_train'
        if the_item in config:
            self.supervised_lm_2_train = config[the_item]

        the_item = 'word_embed_dir'
        if the_item in config:
            self.word_emb_dir = config[the_item]

        the_item = 'MAX_SENTENCE_LENGTH'
        if the_item in config:
            self.MAX_SENTENCE_LENGTH = int(config[the_item])

        the_item = 'MAX_WORD_LENGTH'
        if the_item in config:
            self.MAX_WORD_LENGTH = int(config[the_item])

        the_item = 'norm_word_emb'
        if the_item in config:
            self.norm_word_emb = str2bool(config[the_item])

        the_item = 'number_normalized'
        if the_item in config:
            self.number_normalized = str2bool(config[the_item])

        the_item = 'seg'
        if the_item in config:
            self.seg = str2bool(config[the_item])

        the_item = 'Ren'
        if the_item in config:
            self.Ren = str2bool(config[the_item])

        the_item = 'HIGHWAY'
        if the_item in config:
            self.HIGHWAY = str2bool(config[the_item])

        the_item = 'MESSAGE'
        if the_item in config:
            self.MESSAGE = str2bool(config[the_item])

        the_item = 'W1'
        if the_item in config:
            self.W1 = float(config[the_item])

        the_item = 'W2'
        if the_item in config:
            self.W2 = float(config[the_item])

        the_item = 'W3'
        if the_item in config:
            self.W3 = float(config[the_item])

        the_item = 'W4'
        if the_item in config:
            self.W4 = float(config[the_item])

        the_item = 'word_emb_dim'
        if the_item in config:
            self.word_emb_dim = int(config[the_item])

        the_item = 'task_emb_dim'
        if the_item in config:
            self.task_emb_dim = int(config[the_item])

        the_item = 'domain_emb_dim'
        if the_item in config:
            self.domain_emb_dim = int(config[the_item])

        the_item = 'use_ner_crf'
        if the_item in config:
            self.use_ner_crf = str2bool(config[the_item])

        the_item = 'optimizer'
        if the_item in config:
            self.optimizer = config[the_item]

        the_item = 'ave_batch_loss'
        if the_item in config:
            self.average_batch_loss = str2bool(config[the_item])

        the_item = 'status'
        if the_item in config:
            self.status = config[the_item]

        the_item = 'mode'
        if the_item in config:
            self.mode = config[the_item]

        the_item = 'iteration'
        if the_item in config:
            self.HP_iteration = int(config[the_item])

        the_item = 'batch_size'
        if the_item in config:
            self.HP_batch_size = int(config[the_item])

        the_item = 'hidden_dim'
        if the_item in config:
            self.HP_hidden_dim = int(config[the_item])

        the_item = 'dropout'
        if the_item in config:
            self.HP_dropout = float(config[the_item])

        the_item = 'lstm_layer'
        if the_item in config:
            self.HP_lstm_layer = int(config[the_item])

        the_item = 'bilstm'
        if the_item in config:
            self.HP_bilstm = str2bool(config[the_item])

        the_item = 'gpu'
        if the_item in config:
            self.HP_gpu = str2bool(config[the_item])

        the_item = 'learning_rate'
        if the_item in config:
            self.HP_lr = float(config[the_item])

        the_item = 'learning_rate_cpg'
        if the_item in config:
            self.HP_lr_cpg = float(config[the_item])

        the_item = 'lr_decay'
        if the_item in config:
            self.HP_lr_decay = float(config[the_item])

        the_item = 'clip'
        if the_item in config:
            self.HP_clip = float(config[the_item])

        the_item = 'momentum'
        if the_item in config:
            self.HP_momentum = float(config[the_item])

        the_item = 'l2'
        if the_item in config:
            self.HP_l2 = float(config[the_item])

def config_file_to_dict(input_file):
    config = {}
    fins = open(input_file, 'r').readlines()
    for line in fins:
        if len(line) > 0 and line[0] == "#":
            continue
        if "=" in line:
            pair = line.strip().split('#', 1)[0].split('=', 1)
            item = pair[0]
            if item in config:
                print("Warning: duplicated config item found: %s, updated." % (pair[0]))
            config[item] = pair[-1]
    return config

def str2bool(string):
    if string == "True" or string == "true" or string == "TRUE":
        return True
    else:
        return False