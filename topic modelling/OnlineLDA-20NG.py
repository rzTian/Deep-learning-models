#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 00:36:46 2020

@author: runzhitian
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.nn import Parameter
import torch.nn.functional as F
from pprint import pprint, pformat
import pickle
import argparse
import os
import math

from scipy.stats import norm, entropy
from sklearn.decomposition import LatentDirichletAllocation





def to_onehot(data, min_length):
    return np.bincount(data, minlength=min_length)

def make_data():
    global data_tr, data_te, vocab, vocab_size
    dataset_tr = 'data/20news_clean/train.txt.npy'
    data_tr = np.load(dataset_tr, allow_pickle=True, encoding="latin1")
    dataset_te = 'data/20news_clean/test.txt.npy'
    data_te = np.load(dataset_te, allow_pickle=True, encoding="latin1")
    class StrToBytes:
        
        def __init__(self, fileobj):
            self.fileobj = fileobj  
        def read(self, size):  
            return self.fileobj.read(size).encode()  
        def readline(self, size=-1):  
            return self.fileobj.readline(size).encode()

    vocab = 'data/20news_clean/vocab.pkl'
    with open(vocab, 'r') as data_file:  
        vocab = pickle.load(StrToBytes(data_file))

    vocab_size=len(vocab)
    #--------------convert to one-hot representation------------------
    print ('Converting data to one-hot representation')
    data_tr = np.array([to_onehot(doc.astype('int'),vocab_size) for doc in data_tr if np.sum(doc)!=0])
    data_te = np.array([to_onehot(doc.astype('int'),vocab_size) for doc in data_te if np.sum(doc)!=0])
    #--------------print the data dimentions--------------------------
    print ('Data Loaded')
    print ('Dim Training Data',data_tr.shape)
    print ('Dim Test Data',data_te.shape)
  
    
def print_top_words(beta, feature_names, n_top_words=10):
    print ('---------------Printing the Topics------------------')
    for i in range(len(beta)):
        line = " ".join([feature_names[j] 
                            for j in beta[i].argsort()[:-n_top_words - 1:-1]])
#        topics = identify_topic_in_line(line)
#        print('|'.join(topics))
#        print('     {}'.format(line))
        print('{}'.format(line))
    print ('---------------End of Topics------------------')   
    


#------------------------------------------------
np.random.seed(1)
num_topics = 50
#alpha_prior = 0.1



make_data()

lda = LatentDirichletAllocation(n_components=num_topics, 
#                                doc_topic_prior=alpha_prior,
#                                topic_word_prior=0.1,
                                learning_method='online',
                                max_iter=100,
                                batch_size=200)
print('Training....')
lda.fit(data_tr)


print("perplexity is:", lda.perplexity(data_te))
emb = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]    #[n_components, n_features]

print_top_words(emb, list(zip(*sorted(vocab.items(), key=lambda x:x[1])))[0])

