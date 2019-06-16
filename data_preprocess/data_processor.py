import csv
from PIL import Image
import numpy as np
import os

emotions = {
    '0':'anger', #生气
    '1':'disgust', #厌恶
    '2':'fear', #恐惧
    '3':'happy', #开心
    '4':'sad', #伤心
    '5':'surprised', #惊讶
    '6':'normal', #中性
}


csv_reader = csv.reader(open('fer2013.csv', 'r'))
train_path = './image_gray/train/'
val_path = './image_gray/val/'
test_path = './image_gray/test/'
if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(val_path):
    os.makedirs(val_path)
if not os.path.exists(test_path):
    os.makedirs(test_path)
train_writer = csv.writer(open('./label_train_gray.csv', 'a', newline=''), dialect='excel')
test_writer = csv.writer(open('./label_test_gray.csv', 'a', newline=''), dialect='excel')
val_writer = csv.writer(open('./label_val_gray.csv', 'a', newline=''), dialect='excel')

#usage = ['Training', 'PublicTest', 'PrivateTest']
num_train = [0 for i in range(7)]
num_test = [0 for i in range(7)]
num_val = [0 for i in range(7)]

jump_first_line = True
for item in csv_reader:
    if jump_first_line:
        jump_first_line = False
        continue
    label, feature, use = item
    array = np.array(feature.split(' ')).reshape(48, 48)
    img = Image.fromarray(array.astype('uint8')).convert('L')
    if use == 'Training':
        img_name = 'label_' + label + '_img_' + str(num_train[int(label)]) +'_train.jpg'
        img.save(train_path + img_name)
        print('train img: ', np.shape(img), num_train[int(label)])
        num_train[int(label)] += 1
        train_writer.writerow([img_name, int(label)])
    if use == 'PublicTest':
        img_name = 'label_' + label + '_img_' + str(num_val[int(label)]) +'_val.jpg'
        img.save(val_path + img_name)
        print('val img: ', np.shape(img), label,  num_train[int(label)])
        num_val[int(label)] += 1
        val_writer.writerow([img_name, int(label)])
    if use == 'PrivateTest':
        img_name = 'label_' + label + '_img_' + str(num_test[int(label)]) +'_test.jpg'
        img.save(test_path + img_name)
        print('test img: ', np.shape(img), label,  num_train[int(label)])
        num_test[int(label)] += 1
        test_writer.writerow([img_name, int(label)])
    
    



# array = np.array(item[1].split(' ')).reshape(48, 48)
# img = Image.fromarray(array.astype('uint8')).convert('RGB')
# img.save('test.jpg')




