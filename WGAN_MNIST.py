#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 11:50:24 2018

@author: runzhitian
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import os
import numpy as np

if not os.path.exists('./resultsWGAN-zdim5'):
    os.mkdir('./resultsWGAN-zdim5')


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out


batch_size = 64
num_epoch = 100
z_dimension = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image processing
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
# MNIST dataset
mnist = datasets.MNIST(
    root='./data/', train=True, transform=img_transform, download=False)
# Data loader
dataloader = torch.utils.data.DataLoader(
    dataset=mnist, batch_size=batch_size, shuffle=True)


#WGAN
class WGAN_discriminator(nn.Module):
    def __init__(self):
        super(WGAN_discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2), nn.Linear(256, 1))

    def forward(self, x):
        x = self.dis(x)
        return x




# Generator
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 256), nn.ReLU(True), nn.Linear(256, 784), nn.Tanh())

    def forward(self, x):
        x = self.gen(x)
        return x

G = generator().to(device)
W_D = WGAN_discriminator().to(device)



criterion = nn.BCELoss()

Wg_optimizer = torch.optim.RMSprop(G.parameters(), lr=5e-5)
Wd_optimizer = torch.optim.RMSprop(W_D.parameters(), lr=5e-5)

    
G_losses = []
D_losses = []    
print('WGAN training loop start...')    
for epoch in range(num_epoch):
    for i, (img, _) in enumerate(dataloader):
        num_img = img.size(0)
        # =================train discriminator
        img = img.view(num_img, -1)
        real_img = Variable(img)
        real_label = Variable(torch.ones(num_img))
        fake_label = Variable(torch.zeros(num_img))

        # compute loss of real_img
        real_out = W_D(real_img)
        real_scores = real_out  # closer to 1 means better

        # compute loss of fake_img
        z = Variable(torch.randn(num_img, z_dimension))
        fake_img = G(z)
        fake_out = W_D(fake_img)
        fake_scores = fake_out  # closer to 0 means better

        # bp and optimize
        d_loss = -(torch.mean(real_out) - torch.mean(fake_out))
        Wd_optimizer.zero_grad()
        d_loss.backward()
        Wd_optimizer.step()
        #weight clipping
        for p in W_D.parameters():
            p.data.clamp_(-0.01, 0.01)

        # ===============train generator
        # compute loss of fake_img
        z = Variable(torch.randn(num_img, z_dimension))
        fake_img = G(z)
        output = W_D(fake_img)
        g_loss = -torch.mean(output)

        # bp and optimize
        Wg_optimizer.zero_grad()
        g_loss.backward()
        Wg_optimizer.step()

#        if (i + 1) % 100 == 0:
#            print('Epoch [{}/{}], Wd_loss: {:.6f}, Wg_loss: {:.6f} '
#                  'WD real: {:.6f}, WD fake: {:.6f}'.format(
#                      epoch, num_epoch, d_loss.data[0], g_loss.data[0],
#                      real_scores.data.mean(), fake_scores.data.mean()))
    if epoch == 0:
        real_images = to_img(real_img.data)
        save_image(real_images, './resultsWGAN-zdim5/WGAN_real_images.png')

    fake_images = to_img(fake_img.data)
    save_image(fake_images, './resultsWGAN-zdim5/WGAN_fake_images-{}.png'.format(epoch + 1))
    
    print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                  'D_realscores: {:.6f}, D_fakescores: {:.6f}'.format(
                      epoch, num_epoch, d_loss.data[0], g_loss.data[0],
                      real_scores.data.mean(), fake_scores.data.mean()))
    
    G_losses.append(g_loss.data[0])
    D_losses.append(d_loss.data[0])
    
G_losses = np.array(G_losses)
D_losses = np.array(D_losses)
print('G_losses', G_losses)
print('D_losses', D_losses)
