#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:57:35 2020

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
import matplotlib.pyplot as plt
from torch.nn import init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
np.random.seed(1)
ngpu = 1

num_topics = 50
num_epoch = 70
BATCH_SIZE = 200
opt_type = 'Adam'
Use_MOMENT = True
momentum = 0.99
LR =  0.002    #1e-3 

prior_alpha = torch.Tensor(1, num_topics).fill_(0.98).to(device)
SAVE_FILE = 'lda-vae-20newsgroup'



class ProdLDA(nn.Module):

    def __init__(self, num_input, prior=prior_alpha):
        super(ProdLDA, self).__init__()
        
        self.num_input = num_input
        self.prior_alpha = prior
        
        # encoder
        self.en1_fc     = nn.Linear(self.num_input, 500)             # 1995 -> 100
        self.en2_fc     = nn.Linear(500, 500)             # 100  -> 100
        self.en2_drop   = nn.Dropout(0.2)
        self.mean_fc    = nn.Linear(500, num_topics)             # 100  -> 50
        self.mean_bn    = nn.BatchNorm1d(num_topics)                      # bn for mean
        
        self.encoder = nn.Sequential(
            
            nn.Linear(self.num_input, 500),   #input=parameter+noise
            nn.Softplus(),
            
            nn.Linear(500, 50),   #input=parameter+noise
            nn.BatchNorm1d(50),
            nn.Softplus(),
            
            
            )
        # z
        self.p_drop     = nn.Dropout(0.2)
        # decoder
        self.decoder    = nn.Linear(num_topics, self.num_input)             # 50   -> 1995
        self.decoder_bn = nn.BatchNorm1d(self.num_input)                      # bn for decoder
        self.decoder.weight.data.uniform_(0, 1)
    
    
    def InverseCDF(self, alpha):
        
        noise_hat = (1-1e-10)*torch.rand_like(alpha)+1e-10
        
        gamma_alpha = torch.mvlgamma(alpha, p=1).exp()
    
        assert not torch.isinf(alpha).any()
        assert not torch.isinf(gamma_alpha).any()
        
        sample = (noise_hat*alpha*gamma_alpha).pow(1/alpha)
        assert not torch.isnan(sample).any()
        
        
        
        
        return sample/sample.sum(1, keepdim=True)
        
        
        
    
    def forward(self, inputs, compute_loss=True, avg_loss=True):
        
        
        en1 = F.softplus(self.en1_fc(inputs))                            # en1_fc   output
        en2 = F.softplus(self.en2_fc(en1))                              # encoder2 output
        en2 = self.en2_drop(en2)
        alpha   = F.softplus(self.mean_bn  (self.mean_fc  (en2)))
        alpha = torch.min(30*torch.ones_like(alpha), alpha)  #when alpha is larger than 30, the value of gamma(alpha) become extremly high
        
        p = self.InverseCDF(alpha)

        
        assert not torch.isinf(p).any()
        assert not torch.isnan(p).any()
        
        p = self.p_drop(p)
        
       
        # do reconstruction
        recon = F.softmax(self.decoder_bn(self.decoder(p)), dim=1)             # reconstructed distribution over vocabulary
#        recon = F.softmax(self.decoder(p), dim=1)
        
        
        
        
        if compute_loss:
            return recon, self.loss(inputs=inputs, recon=recon, alpha=alpha, avg=avg_loss)
        else:
            return recon

    def loss(self, inputs, recon, alpha, avg=True):
        # NL
        NL  = -(inputs * (recon+1e-10).log()).sum(1)

        #KLD between two multi-gamma distributions 
        prior_alpha = self.prior_alpha.expand_as(alpha)
        
        KLD = torch.mvlgamma(prior_alpha, p=1).sum(1) - torch.mvlgamma(alpha, p=1).sum(1) + ((alpha-prior_alpha)*torch.digamma(alpha)).sum(1) 
        
        # loss
        loss = (NL + KLD)
        # in traiming mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean(), KLD.mean()
        else:
#            return loss.sum(), KLD.sum()
            return loss, KLD
    

#-----------loading dataset----------#

def to_onehot(data, min_length):
    return np.bincount(data, minlength=min_length)

def make_data():
    global data_tr, data_te, tensor_tr, tensor_te, vocab, vocab_size
    dataset_tr = 'data/AGNews/train.txt.npy'
    data_tr = np.load(dataset_tr, allow_pickle=True, encoding="latin1")
    dataset_te = 'data/AGNews/test.txt.npy'
    data_te = np.load(dataset_te, allow_pickle=True, encoding="latin1")
    class StrToBytes:
        
        def __init__(self, fileobj):
            self.fileobj = fileobj  
        def read(self, size):  
            return self.fileobj.read(size).encode()  
        def readline(self, size=-1):  
            return self.fileobj.readline(size).encode()

    vocab = 'data/AGNews/vocab.pkl'
    with open(vocab, 'rb') as data_file:  

        vocab = pickle.load(data_file)

    vocab_size=len(vocab)
    #--------------convert to one-hot representation------------------
    print ('Converting data to one-hot representation')
    data_tr = np.array([to_onehot(doc.astype('int'),vocab_size) for doc in data_tr if np.sum(doc)!=0])
    data_te = np.array([to_onehot(doc.astype('int'),vocab_size) for doc in data_te if np.sum(doc)!=0])
    #--------------print the data dimentions--------------------------
    print ('Data Loaded')
    print ('Dim Training Data',data_tr.shape)
    print ('Dim Test Data',data_te.shape)
    #--------------make tensor datasets-------------------------------
    tensor_tr = torch.from_numpy(data_tr).float()
    tensor_te = torch.from_numpy(data_te).float()
    tensor_tr = tensor_tr.to(device)
    tensor_te = tensor_te.to(device)
    


#----------------------------------------------------#
        

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        
        
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.2)      #




def make_model():
    global model
    num_input = vocab_size
    model = ProdLDA(num_input=num_input).to(device)
#    model.apply(weights_init)
    
#统计模型有多少参数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())


def make_optimizer():
    global optimizer
    if opt_type == 'Adam'and Use_MOMENT:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), LR, betas=(momentum, 0.999))
#        
    elif opt_type == 'Adam'and not Use_MOMENT:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), LR)
        
    elif opt_type == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), LR)
#        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum)
    else:
        assert False, 'Unknown optimizer {}'.format(opt_type)

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

       
def train():
    train_loss = []
    
    for epoch in range(num_epoch):
        start_time = time.time()
        all_indices = torch.randperm(tensor_tr.size(0)).split(BATCH_SIZE)
        loss_epoch = 0.0
        kld_epoch = 0.0
        model.train()                   # switch to training mode
        for batch_indices in all_indices:
            batch_indices = batch_indices.to(device)
            inputs = tensor_tr[batch_indices]
            recon, (loss, kld_loss) = model(inputs, compute_loss=True, avg_loss=True)
            
            # optimize
            optimizer.zero_grad()       # clear previous gradients
            loss.backward()             # backprop
            optimizer.step()            # update parameters
            # report
            loss_epoch += loss.item()    # add loss to loss_epoch
            kld_epoch += kld_loss.item()
            
#        if epoch % 5 == 0:
#            print('Epoch {}, loss={}'.format(epoch, loss_epoch / len(all_indices)))
        
        
        print('====> Epoch: {} Average loss: {:.4f}, KL Div: {:.4f}'.format(
          epoch, loss_epoch / len(all_indices), kld_epoch / len(all_indices)))
        
        train_loss.append(loss_epoch / len(all_indices))   
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
              
        torch.save(model.state_dict(), SAVE_FILE)
        
        
        
        
    return train_loss




def print_perp(model):
#    cost=[]
    model.eval()                        # switch to testing mode
    inputs = tensor_te
    recon, (loss, KLD) = model(inputs, compute_loss=True, avg_loss=False)
    loss = loss.data
    
    
#    perplexity = torch.exp(loss.sum()/tensor_te.sum())
#    print('The approximated perplexity is: ', perplexity)
    
    counts = tensor_te.sum(1)
    avg = (loss / counts).mean()
    print('The approximated perplexity is: ', torch.exp(avg))




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


#------------------------------------------------------------------#            
make_data()
num_input = vocab_size
make_model()
print(f'The model has {count_parameters(model)[0]:,} trainable parameters')
print(f'The total number of parameter is {count_parameters(model)[1]:,}') 

make_optimizer()
#model.load_state_dict(torch.load(SAVE_FILE))

train_loss = train()
print('Training completed! Saving model.')
torch.save(model.state_dict(), SAVE_FILE)
train_loss = np.array(train_loss)


print_perp(model)
emb = model.decoder.weight.data.cpu().numpy().T    # emb-->(num_topics, 1995)
print_top_words(emb, list(zip(*sorted(vocab.items(), key=lambda x:x[1])))[0])
