#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 12:11:31 2018

@author: runzhitian
"""

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torchvision.utils import save_image
import numpy as np
from torch.nn import functional as F
import torch.autograd as autograd
import os 
from torch.distributions.multivariate_normal import MultivariateNormal

sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 25
BATCH_SIZE = 64
LR = 1e-3        # learning rate
DOWNLOAD_MNIST = False
N_TEST_IMG = 5
ZDIMS = 25     
ngpu = 1
ROUND = 100000000000

# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,                        # download it if you don't have it
)

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


class VAE(nn.Module):
    def __init__(self, ngpu):
        
        super(VAE, self).__init__()
        
        self.ngpu = ngpu
        # ENCODER
        # 28 x 28 pixels = 784 input pixels, 400 outputs
        self.fc1 = nn.Linear(784, 400)
        # rectified linear unit layer from 400 to 400
        # max(0, x)
        self.relu = nn.ReLU()
        self.fc21 = nn.Linear(400, ZDIMS)  # mu layer
        self.fc22 = nn.Linear(400, ZDIMS)  # logvariance layer
        self.mean_bn = nn.BatchNorm1d(ZDIMS)
        self.var_bn  = nn.BatchNorm1d(ZDIMS)   
        # this last layer bottlenecks through ZDIMS connections
        self.h1_drop = nn.Dropout(0.2)
        self.z_drop  = nn.Dropout(0.2)
        
        
        self.encoder = nn.Sequential(
            
            nn.Linear(784, 400),   #input=parameter+noise
            nn.ReLU(),
            
            nn.Linear(400, 400),   #input=parameter+noise
            nn.BatchNorm1d(400),
            nn.ReLU(),
            
            nn.Linear(400, 100),
            nn.BatchNorm1d(100),
#            nn.Dropout(0.2),
            nn.ReLU(),
            
         
            
            )




        # DECODER
        # from bottleneck to hidden 400
        self.fc3 = nn.Linear(ZDIMS, 400)
        self.bn3 = nn.BatchNorm1d(400)
        # from hidden 400 to 784 outputs
        self.fc4 = nn.Linear(400, 784)
        self.sigmoid = nn.Sigmoid()
        
    def encode(self, x):
        
        h1 = self.relu(self.fc1(x))  # type: Variable
#        h1 = self.h1_drop(h1)
#        return self.mean_bn(self.fc21(h1)), self.var_bn(self.fc22(h1))
        return self.fc21(h1), self.fc22(h1)
        
    def reparameterize(self, mu, logvar):
        
        if self.training:
            # multiply log variance with 0.5, then in-place exponent
            # yielding the standard deviation
            
        
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)

        else:
            
            return mu
    
        
    def Real_sampler(self, mu, std):
        
        
        m = torch.distributions.normal.Normal(mu, std)
        data = m.sample()
        
#        eps = torch.randn_like(std)
#        data = eps.mul(std).add_(mu)
#        data = torch.normal(mu, std)
        return data
        
    
    def Variance_reduction(self, mu, log_var):
         
#        print(torch.min(log_var))
#        print(torch.argmin(log_var))
        var = torch.exp(log_var) + 1e-6
#        print(torch.min(var))
#        print(torch.argmin(var))
#        assert not torch.isinf(var).any()
        
        sample = self.Real_sampler(mu, var)
        b = self.decode(sample)
        return b

    
    def Sampling(self, mu, log_var):
        
        std = torch.exp(log_var/2)
#        var = torch.exp(log_var)
#        gaussian_int = self.Real_sampler(mu=torch.floor(10000*mu)/10000, var=torch.floor(10000*var)/10000+1e-10)
        gaussian_int = self.Real_sampler(mu=torch.floor(ROUND*mu)/ROUND, std=torch.floor(ROUND*std)/ROUND)
#        print("original mu:", mu)
#        print("sample mu", torch.floor(10000*mu)/10000)
#        print("original std:", std)
#        print("sample std", torch.floor(10000*std)/10000)
        
        mu = mu - torch.floor(mu*ROUND)/ROUND
        std = std - torch.floor(std*ROUND)/ROUND   
#        var = var - torch.floor(var*10000)/10000+1e-10
        
       
#        eps = torch.randn_like(gaussian_int)
#        sample = mu + eps * std + gaussian_int
        
        
#        sample = mu + var + gaussian_int
        sample = mu + std + gaussian_int 
        
        
#        print(torch.min(std_))
#        print(torch.argmin(std_))
#        print("std", std[26,11])
#        print("std_", std_[26,11])
#        print("cal", std[26,11]-torch.floor(10000*std[26,11])/10000+1e-10)
#        assert not torch.isnan(std.abs().pow(0.5)).any()
        
        
        return sample
        
    def decode(self, z):
        h3 = self.relu(self.fc3(z))
#        h3 = self.bn3(h3)
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        
#        h = self.encoder(x.view(-1, 784))
#        mu = self.fc21(h)
#        logvar = self.fc22(h)
        
        mu, logvar = self.encode(x.view(-1, 784))
        
        z = self.reparameterize(mu, logvar)
#        z = self.Sampling(mu, logvar)
        
#        b = self.Variance_reduction(mu, logvar)
#        z = self.z_drop(z)
        return self.decode(z), mu, logvar



# custom weights initialization called on netG and netD  
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.2)      #use a larger variance to initilize the model
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)      #
        m.bias.data.fill_(0)  



model = VAE(ngpu).to(device)
#model.apply(weights_init)
if (device.type == 'cuda') and (ngpu > 1):
    model = nn.DataParallel(model, list(range(ngpu)))
    
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.99, 0.999))

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False) 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#    return BCE + KLD
    return BCE, KLD
        
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        
        optimizer.zero_grad()
        data = data.to(device)
        recon_batch, mu, logvar = model(data)
#        loss = loss_function(recon_batch, data, mu, logvar)
        BCE, KLD = loss_function(recon_batch, data, mu, logvar)
        loss = BCE+KLD
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if (batch_idx+1) % 100 == 0:
            print ("Epoch[{}/{}], Step [{}/{}], Reconstruction Loss: {:.4f}, KL Div: {:.4f}" 
                   .format(epoch+1, EPOCH, batch_idx+1, len(train_loader), BCE.item(), KLD.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    
    
    
    
    
#    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
    
    return train_loss / len(train_loader.dataset)

import time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
 
    
collect_loss=[]  
model.load_state_dict(torch.load('vae'))
for epoch in range(1, EPOCH + 1):
    start_time = time.time()
    
    l = train(epoch)
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: {epoch:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    
    collect_loss.append(l)
    
    torch.save(model.state_dict(), 'VAE')
    
    with torch.no_grad():
        sample = torch.randn(BATCH_SIZE, ZDIMS).to(device)
        sample = model.decode(sample)
        save_image(sample.view(BATCH_SIZE, 1, 28, 28),
                   './samples/zdim100_' + str(epoch) + '.png')
        
collect_loss = np.array(collect_loss)
