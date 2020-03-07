#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 10:59:40 2018

@author: runzhitian
"""

import numpy as np
import scipy.io as spio
from matplotlib import pyplot as plt
from matplotlib import cm

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def visualization(x, y,count,index):
    x = np.reshape(x, (40, 40))
    
    plt.subplot(1,count,index)
    plt.imshow(x, cmap=cm.Greys_r)
    plt.title(y)
    plt.axis('off')   

def visualization_overlap(x0,x1, y0,y1,count,index):
    r = np.reshape(x0, (40, 40))
    g = np.reshape(x1, (40, 40))
    b = np.zeros_like(r)
    rgb = np.stack([r,g,b],-1)
    
    plt.subplot(1,count,index)
    plt.imshow(rgb)
    plt.title('R:('+ str(y0)+','+str(y1)+')')
    plt.axis('off')   

def load_test_affNIST():
    path = 'affnist/test.mat'
    dataset = loadmat(path)

    ans_set = dataset['affNISTdata']['label_int']
    test_set = dataset['affNISTdata']['image']
#    for i in test_set[:100]:
#         print (i)
#    print ('test_set',test_set.shape)# (1600, 320000)
#    print ('label_set',ans_set.shape)# (320000,)
    return test_set,ans_set

def load_train_affNIST():
    path = 'affnist/training.mat'
    dataset = loadmat(path)

    ans_set = dataset['affNISTdata']['label_int']
    train_set = dataset['affNISTdata']['image']
#    for i in train_set[:100]:
#         print (i)
#    print ('train_set',train_set.shape)#   (1600, 50000)
#    print ('label_set',ans_set.shape)#     (50000,)
    return train_set,ans_set

def load_validation_affNIST():
    path = 'affnist/validation.mat'
    dataset = loadmat(path)

    ans_set = dataset['affNISTdata']['label_int']
    validation_set = dataset['affNISTdata']['image']
#    for i in train_set[:100]:
#         print (i)
#    print ('train_set',train_set.shape)#   (1600, 50000)
#    print ('label_set',ans_set.shape)#     (50000,)
    return validation_set,ans_set

def write_labeldata(labeldata, outputfile):
  header = np.array([0x0801, len(labeldata)], dtype='>i4')
  with open(outputfile, "wb") as f:
    f.write(header.tobytes())
    f.write(labeldata.tobytes())

def write_imagedata(imagedata, outputfile):
  header = np.array([0x0803, len(imagedata), 28, 28], dtype='>i4')
  with open(outputfile, "wb") as f:
    f.write(header.tobytes())
    f.write(imagedata.tobytes())


if __name__ == '__main__':
    
    OVERLAP = not True
    count = 10

    test_set,ans_set = load_test_affNIST()    #分别为测试集(32000*1600)和测试集的标签320000个
    print ('min',np.min(test_set[0]),np.max(test_set[0]))
    write_labeldata(ans_set,"./test/t10k-labels-idx1-ubyte")
    write_imagedata(test_set,"./test/t10k-images-idx3-ubyte")

    train_set,ans_set = load_train_affNIST()
    print ('min',np.min(train_set[0]),np.max(train_set[0]))
    write_labeldata(ans_set,"./train/train-labels-idx1-ubyte")
    write_imagedata(train_set,"./train/train-images-idx3-ubyte")