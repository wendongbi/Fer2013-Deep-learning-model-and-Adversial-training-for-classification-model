import numpy as np
from util import *
from sklearn.naive_bayes import MultinomialNB

# read preprocessed data
train_data_half_aug=read_data("../../mnist/mnist_train/mnist_train_cc1.0_data")
test_data_half_aug=read_data("../../mnist/mnist_test/mnist_test_cc1.0_data")
train_data_aug=read_data("../../mnist/mnist_train/mnist_train_cc1.0_crop45_data")
test_data_aug=read_data("../../mnist/mnist_test/mnist_test_cc1.0_crop45_data")
train_label=np.fromfile("../../mnist/mnist_train/mnist_train_label",dtype=np.uint8)
test_label=np.fromfile("../../mnist/mnist_test/mnist_test_label",dtype=np.uint8)

# Using naive bayes on keep-largest-cc datasets
bayes=MultinomialNB()
print 'largest CC:',
train_time,test_time,accu=run_model(bayes,train_data_half_aug,train_label,test_data_half_aug,test_label)
# print the performance
print 'train time:{} test time:{} accuracy:{}'.format(train_time,test_time,accu)

# Using naive bayes on crop-to-center datasets
bayes=MultinomialNB()
print 'largest CC+Crop to center:',
train_time,test_time,accu=run_model(bayes,train_data_aug,train_label,test_data_aug,test_label)
# print the performance
print 'train time:{} test time:{} accuracy:{}'.format(train_time,test_time,accu)