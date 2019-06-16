#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 11:08:23 2019

@author: wang
"""

import argparse
import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import torchvision.transforms as transforms
import scipy.misc

install_path = os.getcwd()
sys.path.append(install_path)

from util.dataloader import Datasets
from models.resnet import ResNet18

class BIM(object):
    def __init__(self,model, optimizer, criterion, alpha, eps):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.eps = eps
        self.alpha = alpha
        self.num_iters = math.ceil(min(self.eps+4,1.25*self.eps))

    def attack(self, orig_img, orig_label):
        img_bim = torch.tensor(orig_img.data, requires_grad=True)
        for i in range(self.num_iters):
            self.optimizer.zero_grad()
            output = self.model(img_bim)

            loss = self.criterion(output, orig_label)
            loss.backward()
            delta = self.alpha * torch.sign(img_bim.grad.data)
            img_bim = torch.tensor(img_bim.data + delta, requires_grad=True) # adversary without clipping
            
            clipped_delta = torch.clamp(img_bim.data-orig_img.data, -self.eps,self.eps) #clipping the delta
            img_bim = torch.tensor(orig_img.data+clipped_delta,requires_grad=True) # adding the clipped delta to original image
        return img_bim

parser = argparse.ArgumentParser(description="facial expression classifiers")
parser.add_argument("--method", type=str, default="ResNet")
parser.add_argument("--data_dir", type=str, default="data/fer2013.csv")
parser.add_argument("--model_dir", type=str, default="checkpoint/2/test_ckpt.t7")
parser.add_argument("--output_dir", type=str, default="data/BasicIterative_gray")
parser.add_argument("--alpha", type=float, default=0.002)
parser.add_argument("--eps", type=float, default=0.02)
parser.add_argument("--cuda", type=bool, default=True)

args = parser.parse_args()
    
device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = Datasets('./data/label_train_gray.csv','./data/image_gray/train', transform=transform_test)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=12)
testset = Datasets('./data/label_test_gray.csv','./data/image_gray/test', transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=12)
valset = Datasets('./data/label_val_gray.csv','./data/image_gray/val', transform=transform_test)
val_loader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=12)

# Model
print('==> Building model..')
if args.method=='ResNet':
    net = ResNet18()
else:
    raise NotImplementedError
net.to(device)

print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load(args.model_dir)
net.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-4)

def generate_adversial_examples(BIM, data_loader, stage):
    img_id = np.zeros(7)
    if not os.path.exists(os.path.join(args.output_dir, stage)):
        os.makedirs(os.path.join(args.output_dir, stage))
    BIM.model.eval()
    label_lists = []
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        examples = BIM.attack(inputs, targets)
        
        for idx, example in enumerate(examples):
            example = example.data.cpu().numpy()
            label = targets[idx].item()
            file_name = 'label_' + str(label) + '_img_'+str(int(img_id[label]))+'.jpg'
            out_path = os.path.join(args.output_dir, stage, file_name)
            img_id[label] += 1
            example = np.squeeze(example.transpose(1,2,0))
            scipy.misc.toimage(example, cmin=0.0, cmax=1.0).save(out_path)
            label_lists.append([file_name, label])
    with open(os.path.join(args.output_dir, stage+'_label.csv'), 'w') as f:
        f.write('\n')
        for file_name, label in label_lists:
            f.write(file_name+','+str(label)+'\n')
                   

if __name__ == "__main__":
    bim = BIM(net, optimizer, criterion, args.alpha, args.eps)
    generate_adversial_examples(bim, val_loader, 'val')
    generate_adversial_examples(bim, test_loader, 'test')