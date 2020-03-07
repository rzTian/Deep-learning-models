#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 23:49:57 2018

@author: runzhitian
"""

from torch.utils.data.dataset import Dataset
from torch.utils import data
from torchvision import transforms
import torch
import os 
from PIL import Image
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy
import torch.nn as nn

import load_affnist
BATCH_SIZE = 128
EPOCH = 20
LR = 0.001  
ngpu = 1 

class affNIST(data.Dataset):
    def __init__(self,  root, transform, train=True, target_transform=None, download=False):
        
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.downlaod = download
        self.root = root
 
#        if download:
#            self.download()
 
#        if not self._check_exists():
#            raise RuntimeError('Dataset not found.' +
#                               ' You can use download=True to download it')
 
        if self.train:
            self.train_data, self.train_labels = load_affnist.load_train_affNIST()
            self.train_data = torch.from_numpy(self.train_data.T.reshape(-1, 40, 40))
            self.train_labels = torch.from_numpy(self.train_labels)
        else:
            self.test_data, self.test_labels = load_affnist.load_test_affNIST()
            self.test_data = torch.from_numpy(self.test_data.T.reshape(-1, 40, 40))
            self.test_labels = torch.from_numpy(self.test_labels)
 
    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
        
 
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
 
        if self.transform is not None:
            img = self.transform(img)
 
        if self.target_transform is not None:
            target = self.target_transform(target)
 
        return img, target
 
    def __len__(self):
        
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
        
        
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    
train_data = affNIST(root='./train/', transform = transforms.ToTensor())
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = affNIST(root='./train/', transform = transforms.ToTensor(), train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:5000]/255.   # shape from (320000, 40, 40) to (320000, 1, 40, 48), value in range(0,1)
test_y = test_data.test_labels[:5000]

#plot one example
print('train_data size:', train_data.train_data.shape)                 # (50000, 40, 40)
print('train_labels szie:', train_data.train_labels.shape)               # (50000)
plt.imshow(train_data.train_data[100], cmap='gray')
plt.title('%i' % train_data.train_labels[100])
plt.show()

print('test_data size:', test_data.test_data.shape)                 # (320000, 40, 40)
print('test_labels szie:', test_data.test_labels.shape)               # (320000)
plt.imshow(test_data.test_data[10000], cmap='gray')
plt.title('%i' % test_data.test_labels[10000])
plt.show()

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

class CNN(nn.Module):
    def __init__(self, ngpu):
        super(CNN, self).__init__()
        self.ngpu = ngpu
        self.conv1 = nn.Sequential(         # input shape (1, 40, 40)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=256,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (256, 40, 40)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (256, 20, 20)
        )
        self.conv2 = nn.Sequential(         # input shape (256, 20, 20)
            nn.Conv2d(256, 256, 5, 1, 2),     # output shape (256, 20, 20)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (256, 10, 10)
        )
        
        self.conv3 = nn.Sequential(         # input shape (256, 10, 10)
            nn.Conv2d(256, 128, 5, 1, 2),     # output shape (128, 10, 10)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (128, 5, 5)
        )

        
        self.FCL = torch.nn.Sequential(
                
                torch.nn.Linear(128*5*5, 328),
#                torch.nn.Dropout(0.5),  # drop 50% of the neuron
                torch.nn.ReLU(),
                torch.nn.Linear(328, 192),
#                torch.nn.Dropout(0.5),  # drop 50% of the neuron
                torch.nn.ReLU(),
                torch.nn.Linear(192, 10),
                torch.nn.Dropout(0.5),
                )
        

        
        # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 128 * 5 * 5)
        output = self.FCL(x)
        return output

cnn = CNN(ngpu).to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  

# training and testing

print('training loop start...')
cnn.train()
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        output = cnn(b_x)               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

#        if step % 50 == 0:
#            test_output = cnn(test_x)
#            pred_y = torch.max(test_output, 1)[1].data.numpy()
#            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
#            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            
    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())


print('model saving...')
torch.save(cnn.state_dict(), 'cnn_model.pt')
print('done!')


#load model..
device = torch.device('cpu')
cnn.load_state_dict(torch.load('cnn_model.pt', map_location=device), strict=False)

print('testing....')
cnn.eval()
test_output = cnn(test_x)
pred_y = torch.max(test_output, 1)[1].data.numpy()
accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
print('| test accuracy: %.2f' % accuracy)

            
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')           
            