#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:05:37 2019

@author: wang
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from util.dataloader import Datasets
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from PIL import Image
import scipy.misc

from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from models.resnet import *
from torch.nn.utils import clip_grad_norm_
from utils import progress_bar
import numpy as np
import matplotlib.pyplot as plt

train_id = 13
resume_id = 12


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--optim', default='Adam', type=str)
parser.add_argument('--epoch', default=50, type=int, metavar='N')
parser.add_argument('--weight-decay','--wd',default=1e-4,type=float, metavar='W')
parser.add_argument('--output_dir', default='./data/adversial',type=str)
parser.add_argument('--magnitude', default=0.02,type=float)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--logspace', default=1, type=int)
args = parser.parse_args()

device = torch.device('cuda:3') if torch.cuda.is_available() else 'cpu'
best_acc_val = 0  # best test accuracy
best_acc_test = 0  # best test accuracy

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
save_path = 'checkpoint/' + str(train_id) + '/'

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomResizedCrop(44),
    transforms.RandomHorizontalFlip(),
    # transforms.Grayscale(),
    # transforms.ColorJitter(1, 1, 1, 0.5),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    # transforms.CenterCrop(42),
    # transforms.Grayscale(),
    # transforms.CenterCrop(44),
    transforms.ToTensor(),
    # transforms.Lambda(lambda crops:torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])



trainset = Datasets('./data/label_train_gray.csv','./data/image_gray/train', transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=12)


testset = Datasets('./data/label_test_gray.csv','./data/image_gray/test', transform=transform_test)
#testset = Datasets('./AdversialExamples/FastGradient_gray/test_label.csv','./AdversialExamples/FastGradient_gray/test_set', transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=12)

valset = Datasets('./data/label_val_gray.csv','./data/image_gray/val', transform=transform_test)
#valset = Datasets('./AdversialExamples/FastGradient_gray/val_label.csv','./AdversialExamples/FastGradient_gray/val_set', transform=transform_test)
val_loader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=12)


# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
#net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/' + str(resume_id) + '/test_ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
if args.optim == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
elif args.optim == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
logspace_lr = torch.logspace(np.log10(args.lr), np.log10(args.lr) - args.logspace, args.epoch)



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    loss_res = 0
    if args.logspace != 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = logspace_lr[epoch-start_epoch]
    else:
        adjust_learning_rate(optimizer, epoch)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        prob = F.softmax(outputs, dim=1)
        loss = criterion(outputs, targets)
        loss.backward()
        # clip_grad_norm_(parameters=net.parameters(), max_norm=0.1)

        optimizer.step()

        train_loss += loss.item() 
        loss_res = train_loss/(batch_idx + 1)

        max_num, predicted = prob.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
#        progress_bar(batch_idx, len(train_loader), 'Original Loss: %.3f | Acc: %.3f%% (%d/%d)'
#            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    
    if epoch >= start_epoch+1:
        adversialset = Datasets('./data/label_adversial_gray.csv', args.output_dir, transform=transform_train)
        adversial_loader = torch.utils.data.DataLoader(adversialset, batch_size=64, shuffle=True, num_workers=12)
        train_loss = 0
        correct = 0
        total = 0
        loss_res = 0
        for batch_idx, (inputs, targets) in enumerate(adversial_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            prob = F.softmax(outputs, dim=1)
            loss = criterion(outputs, targets)
            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item()
            
            max_num, predicted = prob.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
#            progress_bar(batch_idx, len(train_loader), 'Original Loss: %.3f | Acc: %.3f%% (%d/%d)'
#            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # gernerate examples
    img_id = np.zeros(7)
    label_lists = []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        inputs.requires_grad_()
        outputs = net(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        noise = inputs.grad
        examples = inputs + args.magnitude * torch.sign(noise)
        
        for idx, example in enumerate(examples):
            example = example.data.cpu().numpy()
            label = int(targets[idx].item())
            file_name = 'label_' + str(label) + '_img_'+str(int(img_id[label]))+'.jpg'
            out_path = os.path.join(args.output_dir, file_name)
            img_id[label] += 1
            example = np.squeeze(example.transpose(1,2,0))
            scipy.misc.toimage(example, cmin=0.0, cmax=1.0).save(out_path)
            label_lists.append([file_name, label])
    with open(os.path.join('./data', 'label_adversial_gray.csv'),'w') as f:
        f.write('\n')
        for file_name, label in label_lists:
            f.write(file_name+','+str(label)+'\n')
            
    return loss_res
        
        

    
def validation(epoch):
    global best_acc_val
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        flag = True
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            prob = outputs
            loss = criterion(outputs, targets)

            test_loss += loss.item()

            # normalization - softmax
            for i in range(len(prob)):
                prob[i] = prob.exp()[i] / prob.exp()[i].sum()
            max_num, predicted = prob.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

#            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('Val Accu: ' + str(100. * correct / total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc_val:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        torch.save(state, save_path + 'val_ckpt.t7')
        best_acc_val = acc
    return acc
def test(epoch):
    global best_acc_test
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    confusion = np.zeros((7, 7)).astype(int)
    with torch.no_grad():
        flag = True
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            prob = outputs
            loss = criterion(outputs, targets)

            test_loss += loss.item()

            # normalization - softmax
            for i in range(len(prob)):
                prob[i] = prob.exp()[i] / prob.exp()[i].sum()
            max_num, predicted = prob.max(1)
            


            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            for i in range(len(targets)):
                confusion[targets[i], predicted[i]]+=1
            
#            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print(confusion)
        # for i in range(len(confusion)):
        #     for j in range(len(confusion[0])):
        #         confusion[i][j] /= 255
        confusion_img = Image.fromarray(confusion.astype(np.uint8))
        img = confusion_img.resize((64, 64))
        img.save('./confusion.jpg')
        print('Test Accu: ' + str(100. * correct / total))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc_test:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        torch.save(state, save_path + 'test_ckpt.t7')
        best_acc_test = acc
    return acc


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

val_acc = []
test_acc = []
train_loss = []
if __name__ == "__main__":
    # validation(0)
    #test(0)
    for epoch in range(start_epoch, start_epoch+args.epoch):
        loss = train(epoch)
        acc1 = validation(epoch)
        acc2 = test(epoch)
        train_loss.append(loss)
        val_acc.append(acc1)
        test_acc.append(acc2)

    # plot and save
    # x = np.load('results/val_acc_{}.npy'.format(train_id))
    x = [i for i in range(len(val_acc))]
    # y1 = np.load('results/train_loss_{}.npy'.format(train_id))
    # y2 = np.load('results/val_acc_{}.npy'.format(train_id))
    # y3 = np.load('results/test_acc_{}.npy'.format(train_id))
    y1 = train_loss
    y2 = val_acc
    y3 = test_acc
    plt.figure(figsize=(10, 10)).tight_layout(pad=0.4, w_pad=3., h_pad=5.)
    plt.subplot(311)
    plt.plot(x, y1, color='blue')
    plt.title('train loss')

    plt.subplot(312)
    plt.plot(x, y2, color='red')
    plt.title('val accu(best_accu:' + str(best_acc_val) + ')')

    plt.subplot(313)
    plt.plot(x, y3)
    plt.title('test accu(best accu:' + str(best_acc_test) + ')')
    plt.savefig('results/result_{}.jpg'.format(train_id))
    # plt.show()
    # save results
    np.save('results/val_acc_{}.npy'.format(train_id), np.array(val_acc))
    np.save('results/test_acc_{}.npy'.format(train_id), np.array(test_acc))
    np.save('results/train_loss_{}.npy'.format(train_id), np.array(train_loss))

    

    
