#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 09:48:24 2020

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)





ngpu = 1
num_epoch = 500


BATCH_SIZE = 64
opt_type = 'Adam'
LR = 5e-5
num_topics = 50

mean = torch.Tensor(1, num_topics).fill_(0.0)
variance = torch.Tensor(1, num_topics).fill_(1.)


class ProdLDA(nn.Module):

    def __init__(self, num_input, num_topics, mean, variance):
        super(ProdLDA, self).__init__()
        
        self.num_input = num_input
        self.num_topic = num_topics
        self.mean = mean
        self.variance = variance
        # encoder
        self.en1_fc     = nn.Linear(self.num_input, 500)             # 1995 -> 100
        
        self.mean_fc    = nn.Linear(500, self.num_topic)             # 100  -> 50
        self.mean_bn    = nn.BatchNorm1d(self.num_topic)                      # bn for mean
        self.logvar_fc  = nn.Linear(500, self.num_topic)             # 100  -> 50
        self.logvar_bn  = nn.BatchNorm1d(self.num_topic)                      # bn for logvar
        
        # decoder
        self.decoder    = nn.Linear(self.num_topic, self.num_input)             # 50   -> 1995
        self.decoder_bn = nn.BatchNorm1d(self.num_input)                      # bn for decoder
 
        prior_mean   = self.mean
        prior_var    = self.variance
#        prior_mean, prior_var = Laplace(alpha_prior)
        prior_logvar = prior_var.log()
        self.register_buffer('prior_mean',    prior_mean)
        self.register_buffer('prior_var',     prior_var)
        self.register_buffer('prior_logvar',  prior_logvar)
        
        self.decoder.weight.data.uniform_(0, 1)
    

    def forward(self, inputs, compute_loss=False, avg_loss=True):
        # compute posterior
        en = torch.tanh(self.en1_fc(inputs))                            # en1_fc   output
        
        posterior_mean   = self.mean_fc  (en)          # posterior mean
        posterior_logvar = self.logvar_fc(en)          # posterior log variance
        posterior_var    = posterior_logvar.exp()
        # take sample
        eps = Variable(inputs.data.new().resize_as_(posterior_mean.data).normal_()) # noise
        z = posterior_mean + posterior_var.sqrt() * eps                 # reparameterization

        # do reconstruction
        recon = F.softmax(self.decoder(z), dim=1)             # reconstructed distribution over vocabulary
        
        if compute_loss:
            return recon, self.loss(inputs, recon, posterior_mean, posterior_logvar, posterior_var, avg_loss)
        else:
            return recon

    def loss(self, inputs, recon, posterior_mean, posterior_logvar, posterior_var, avg=True):
        # NL
        NL  = -(inputs * (recon+1e-10).log()).sum(1)
        
        prior_mean   = Variable(self.prior_mean).expand_as(posterior_mean)
        prior_var    = Variable(self.prior_var).expand_as(posterior_mean)
        prior_logvar = Variable(self.prior_logvar).expand_as(posterior_mean)
        var_division    = posterior_var  / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        # put KLD together
        KLD = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - self.num_topic )
        # loss
        loss = (NL + KLD)
        # in traiming mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean(), KLD.mean()
        else:
            return loss, KLD






def to_onehot(data, min_length):
    return np.bincount(data, minlength=min_length)

def make_data():
    global data_tr, data_te, tensor_tr, tensor_te, vocab, vocab_size
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
    #--------------make tensor datasets-------------------------------
    tensor_tr = torch.from_numpy(data_tr).float()
    tensor_te = torch.from_numpy(data_te).float()
    tensor_tr = tensor_tr.to(device)
    tensor_te = tensor_te.to(device)

#def weights_init(m):
#    classname = m.__class__.__name__
#    if classname.find('Linear') != -1:
#        m.weight.data.normal_(0.0, 0.2)      #use a larger variance to initilize the model
#        m.bias.data.fill_(0)
#    elif classname.find('BatchNorm') != -1:
#        m.weight.data.normal_(1.0, 0.2)      #
#        m.bias.data.fill_(0)    
        

def make_model():
    global model
    num_input = data_tr.shape[1]
    model = ProdLDA(num_input, num_topics, mean, variance).to(device)
#    model.apply(weights_init)
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())


def make_optimizer():
    global optimizer
    if opt_type == 'Adam':
        
        optimizer = torch.optim.Adam(model.parameters(), LR)
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
            recon, (loss, kld_loss) = model(inputs, compute_loss=True)
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
              
        torch.save(model.state_dict(), 'vae-lda-gaussian.pt')
        
    return train_loss

def print_perp(model):
#    cost=[]
    model.eval()                        # switch to testing mode
    inputs = tensor_te
    recon, (loss, _) = model(inputs, compute_loss=True, avg_loss=False)
    print(loss)
    loss = loss.data

    
    counts = tensor_te.sum(1)
    avg = (loss / counts).mean()
    print('The approximated perplexity is: ', torch.exp(avg))    
    
    

associations = {
    'jesus': ['prophet', 'jesus', 'matthew', 'christ', 'worship', 'church'],
    'comp ': ['floppy', 'windows', 'microsoft', 'monitor', 'workstation', 'macintosh', 
              'printer', 'programmer', 'colormap', 'scsi', 'jpeg', 'compression'],
    'car  ': ['wheel', 'tire'],
    'polit': ['amendment', 'libert', 'regulation', 'president'],
    'crime': ['violent', 'homicide', 'rape'],
    'midea': ['lebanese', 'israel', 'lebanon', 'palest'],
    'sport': ['coach', 'hitter', 'pitch'],
    'gears': ['helmet', 'bike'],
    'nasa ': ['orbit', 'spacecraft'],
}

def identify_topic_in_line(line):
    topics = []
    for topic, keywords in associations.items():
        for word in keywords:
            if word in line:
                topics.append(topic)
                break
    return topics

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
    
make_model()
print(f'The model has {count_parameters(model)[0]:,} trainable parameters')
print(f'The total number of parameter is {count_parameters(model)[1]:,}') 

make_optimizer()
model.load_state_dict(torch.load('vae-lda-gaussian.pt'))

train_loss = train()
print('Training completed! Saving model.')
torch.save(model.state_dict(), 'vae-lda-gaussian.pt')
train_loss = np.array(train_loss)

print_perp(model)
emb = model.decoder.weight.data.cpu().numpy().T    # emb-->(num_topics, 1995)
print_top_words(emb, list(zip(*sorted(vocab.items(), key=lambda x:x[1])))[0])
