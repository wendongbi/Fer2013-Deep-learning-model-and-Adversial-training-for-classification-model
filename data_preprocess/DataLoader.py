#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:37:59 2019

@author: wang
"""
import numpy as np
import torchvision.transforms as transforms

def load_data_from_file(data_path):
    """
    data_path: the path of .csv file
    each line is formed: label, feature, usage
    """
    #usage = ['Training', 'PublicTest', 'PrivateTest']
    train_data, train_label = [], []
    dev_data, dev_label = [], []
    test_data, test_label = [], []
    with open(data_path, 'r') as f:
        f.readline()
        for line in f:
            label, features, use = line.strip().split(',')
            feature = [int(item) for item in features.split()]
            if use == 'Training':
                train_data.append(feature)
                train_label.append(int(label))
            elif use == 'PublicTest':
                dev_data.append(feature)
                dev_label.append(int(label))
            elif use == 'PrivateTest':
                test_data.append(feature)
                test_label.append(int(label))
            else:
                raise NotImplementedError
    data = {
            'train': (np.array(train_data).reshape(-1,1,48,48)/255, np.array(train_label)),
            'dev': (np.array(dev_data).reshape(-1,1,48,48)/255, np.array(dev_label)),
            'test': (np.array(test_data).reshape(-1,1,48,48)/255, np.array(test_label))
            }
    return data

if __name__ == '__main__':
    data_path = './fer2013.csv'
    data = load_data_from_file(data_path)
    for key in data:
        X, Y = data[key]
        print(key, X.shape[0])