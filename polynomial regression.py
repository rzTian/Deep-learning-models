#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 12:00:16 2018

@author: runzhitian


This is a code for polynomial regression.
"""


import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

def getdata(N,sigma):   # generating training data
 x=torch.rand(N)
 mu = 0
 z = np.random.normal(mu, sigma, N)
 y=np.cos(2*x*np.pi)
 y=np.add(y,z)
 x=x.numpy()          
 y=y.numpy()
# plt.plot(x, y, 'bo')  
 return x,y



def getMSE(y_train,y):  
    return torch.mean((y-y_train)**2)
    


def fitdata(d,x_train,y_train,sigma):
    x_train = np.stack([x_train ** i for i in range(0, d+1)], axis=1)
    x_train = torch.from_numpy(x_train).float()                       
    y_train = torch.from_numpy(y_train).float().unsqueeze(1)  
    w = Variable(torch.randn(d+1,1), requires_grad=True)               
    x_train = Variable(x_train)
    y_train = Variable(y_train)
    
    count=0                         
    last_loss=100000
    for gd in range(50000):         
        if gd<10000:                
            lamda=0.1
        elif gd>=10000 and gd<20000:
            lamda=0.01
        elif gd>20000 and gd<30000:
            lamda=0.001
        else:
            lamda=0.0001
        
        y=torch.mm(x_train, w)       
        loss=getMSE(y_train,y)
        re_loss=loss+15*(w.data.norm())**2   #regularization
        delta=re_loss-last_loss
        count=count+1
        if count>=49998:             
            print('it is unfair, I iterate too much')
            
        if delta.abs()<0.0000001:    #set a threshold to stop training
            print(count)
            break
        else:
            last_loss=re_loss
        
        re_loss.backward()                          
        w.data = w.data - lamda*w.grad.data       
        w.grad.data.zero_()           
               
    Ein=loss
    x_test,y_test=getdata(1500,sigma)              #generating training data
    x_test = np.stack([x_test ** i for i in range(0, d+1)], axis=1) 
    x_test = torch.from_numpy(x_test).float() 
    y_test = torch.from_numpy(y_test).float().unsqueeze(1)
    y=torch.mm(x_test, w)    
    Eout=getMSE(y_test,y)
    Ein=Ein.data.numpy()        
    Eout=Eout.data.numpy()
    w=w.data.numpy()        
    return Ein, Eout, w     

def experiment(N,d,sigma):
    Ein_set=[]             
    Eout_set=[]
    w_set=[]
    for i in range(25):          
        x,y=getdata(N,sigma)
        Ein,Eout,w=fitdata(d,x,y,sigma)
        Ein_set.append(Ein)           
        Eout_set.append(Eout)
        w_set.append(w)
    
    Ein_set=np.array(Ein_set)
    Eout_set=np.array(Eout_set)
    w_set=np.array(w_set)     
    Ein_mean=np.mean(Ein_set)
    Eout_mean=np.mean(Eout_set)
    w_mean=np.mean(w_set,axis=0)        
    x2,y2=getdata(1500,sigma)
    x2 = np.stack([x2 ** i for i in range(0, d+1)], axis=1) 
    y_ave=np.dot(x2,w_mean)
    y2=np.array([y2]).T          
    Ebias=np.mean((y2-y_ave)**2)
    return Ein_mean,Eout_mean,Ebias



Ein_mean_set=[]
Eout_mean_set=[]
Ebias_set=[]

for d in range(16):     
#for N in [2,5,10,20,50,100,200]:
    Ein_mean,Eout_mean,Ebias=experiment(10,d,0.01)
    Ein_mean_set.append(Ein_mean)
    Eout_mean_set.append(Eout_mean)
    Ebias_set.append(Ebias)

Ein_mean_set=np.array(Ein_mean_set)
Eout_mean_set=np.array(Eout_mean_set) 
Ebias_set=np.array(Ebias_set) 
print(Ein_mean_set)
print(Eout_mean_set)
print(Ebias_set)





    
    
        


        
