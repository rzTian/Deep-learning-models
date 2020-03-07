#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:32:20 2019

@author: runzhitian
"""


import torch
from torchtext import data
import torch.nn.functional as F #
import torch.optim as optim 
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
from torchtext import datasets
from torchtext.vocab import GloVe
from torchtext.data import Iterator, BucketIterator
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1) 

num_epochs = 60
embed_size = 100
num_hiddens = 100      
num_layers = 1 
bidirectional = False
batch_size = 64
labels = 2
lr = 0.02
ngpu = 1
SentenceLength=500
dropout = 0.5

TEXT = data.Field(include_lengths = True, fix_length=SentenceLength)
LABEL = data.LabelField(dtype = torch.float)

train, test = datasets.IMDB.splits(TEXT, LABEL) 
TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=100),min_freq=10, unk_init = torch.Tensor.normal_) #min_freq删除任何未发生十次以上的单词
LABEL.build_vocab(train)
train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=batch_size, device=device, shuffle=True)


#-----------Functions to accomplish attention--------------#
def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight) 
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0,1)
        if(nonlinearity=='tanh'):
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if(s is None):
            s = _s_bias
        else:
            s = torch.cat((s,_s_bias),0)
    return s.squeeze()

def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if(nonlinearity=='tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze()

def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 0).unsqueeze(0)


#-----------------------------------------------------------------#
#-----------------------------------------------------------------#

class SentimentNet(nn.Module):
    def __init__(self, vocab_size, embed_size, pad_idx, num_hiddens, num_layers, bidirectional, labels, ngpu, batch_size, dropout, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.ngpu = ngpu
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx = pad_idx)
#        self.embedding = nn.Embedding.from_pretrained(weight)
#        self.embedding.weight.requires_grad = True
#        self.output_size = output_size
        self.batch_size = batch_size
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.LSTM(input_size=embed_size,hidden_size=self.num_hiddens,num_layers=num_layers,bidirectional=self.bidirectional,dropout=0)
        
        if bidirectional == True:
            self.weight_W_word = nn.Parameter(torch.Tensor(2* self.num_hiddens,2*self.num_hiddens))
            self.bias_word = nn.Parameter(torch.Tensor(2* self.num_hiddens,1))
            self.weight_proj_word = nn.Parameter(torch.Tensor(2*self.num_hiddens, 1))
            self.decoder = nn.Linear(num_hiddens * 2, labels)
        else:
            self.weight_W_word = nn.Parameter(torch.Tensor(self.num_hiddens, self.num_hiddens))
            self.bias_word = nn.Parameter(torch.Tensor(self.num_hiddens,1))
            self.weight_proj_word = nn.Parameter(torch.Tensor(self.num_hiddens, 1))
            self.decoder = nn.Linear(num_hiddens, labels)
            
        self.softmax_word = nn.Softmax()
        self.weight_W_word.data.uniform_(-0.1, 0.1)
        self.weight_proj_word.data.uniform_(-0.1,0.1)
       

    def forward(self, inputs, text_lengths):
        embedded = self.dropout(self.embedding(inputs))
#        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=False, enforce_sorted=False)
        states, hidden = self.encoder(embedded)
        word_squish = batch_matmul_bias(states, self.weight_W_word, self.bias_word, nonlinearity='tanh')
        word_attn = batch_matmul(word_squish, self.weight_proj_word)
        word_attn_norm = self.softmax_word(word_attn.transpose(1,0))
        word_attn_vectors = attention_mul(states, word_attn_norm.transpose(1,0))
        outputs = self.decoder(word_attn_vectors)
        
        return outputs
    
    def init_hidden(self):
        if self.bidirectional == True:
            return Variable(torch.zeros(2, self.batch_size, self.num_hiddens))
        else:
            return Variable(torch.zeros(1, self.batch_size, self.num_hiddens))
        
        
vocab_size = len(TEXT.vocab) 
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]      
net = SentimentNet(vocab_size=vocab_size, embed_size=embed_size, pad_idx=PAD_IDX, num_hiddens=num_hiddens, num_layers=num_layers,
                   bidirectional=bidirectional, labels=labels, ngpu=ngpu, batch_size=batch_size, dropout = dropout)
net.to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    net = nn.DataParallel(net, list(range(ngpu)))
 
    
#count the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

print(f'The model has {count_parameters(net):,} trainable parameters')


pretrained_embeddings = TEXT.vocab.vectors
print(pretrained_embeddings.shape)
net.embedding.weight.data.copy_(pretrained_embeddings)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
net.embedding.weight.data[UNK_IDX] = torch.zeros(embed_size)
net.embedding.weight.data[PAD_IDX] = torch.zeros(embed_size)
print(net.embedding.weight.data)


#--------------------------------------------#    
loss_function = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)
#optimizer = optim.SGD(net.parameters(), lr=lr)



def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
#    rounded_preds = torch.round(torch.sigmoid(preds))
    preds = torch.argmax(F.softmax(preds, dim=1), 1)
    correct = (preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        text, text_lengths = batch.text
        
        label = batch.label
        
        label = label.type(torch.LongTensor)
        
        text = text.to(device)
        
        text_lengths = text_lengths.to(device)
        
        label = label.to(device)
        
        predictions = model(text, text_lengths).squeeze()
        
        loss = criterion(predictions, label)
        
        acc = binary_accuracy(predictions, label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text, text_lengths = batch.text
            
            text = text.to(device)
            
            label = batch.label
        
            label = label.type(torch.LongTensor)
            
            text_lengths = text_lengths.to(device)
        
            label = label.to(device)
            
            predictions = model(text, text_lengths).squeeze()
            
            loss = criterion(predictions, label)
            
            acc = binary_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



losses = []
accuracy = []
for epoch in range(num_epochs):

    start_time = time.time()
    
    train_loss, train_acc = train(net, train_iter, optimizer, loss_function, device)
#    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
#    if valid_loss < best_valid_loss:
#        best_valid_loss = valid_loss
    torch.save(net.state_dict(), 'tut2-model.pt')
    
    losses.append(train_loss)
    accuracy.append(train_acc)
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
#    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

print("training loss:")
print(losses)
print("training accuracy")
print(accuracy)
losses = np.array(losses)
acc = np.array(accuracy)
np.savetxt("loss.txt", losses)
np.savetxt("acc.txt", acc)

    
net.load_state_dict(torch.load('tut2-model.pt'))
test_loss, test_acc = evaluate(net, test_iter, loss_function, device)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')