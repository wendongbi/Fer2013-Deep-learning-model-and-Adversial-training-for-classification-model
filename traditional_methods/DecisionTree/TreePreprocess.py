import numpy as np
from util import *
from sklearn.tree import DecisionTreeClassifier

# read preprocessed data
train_data_half_aug=read_data("../../mnist/mnist_train/mnist_train_cc1.0_data")
test_data_half_aug=read_data("../../mnist/mnist_test/mnist_test_cc1.0_data")
train_data_aug=read_data("../../mnist/mnist_train/mnist_train_cc1.0_crop45_data")
test_data_aug=read_data("../../mnist/mnist_test/mnist_test_cc1.0_crop45_data")
train_label=np.fromfile("../../mnist/mnist_train/mnist_train_label",dtype=np.uint8)
test_label=np.fromfile("../../mnist/mnist_test/mnist_test_label",dtype=np.uint8)

# Using decision tree on keep-largest-cc datasets
tree=DecisionTreeClassifier()
print 'largest CC:',
train_time,test_time,accu=run_model(tree,train_data_half_aug,train_label,test_data_half_aug,test_label)
# print the performance
print 'train time:{} test time:{} accuracy:{}'.format(train_time,test_time,accu)

# Using decision tree on crop-to-center datasets
tree=DecisionTreeClassifier()
print 'largest CC+Crop to center:',
train_time,test_time,accu=run_model(tree,train_data_aug,train_label,test_data_aug,test_label)
# print the performance
print 'train time:{} test time:{} accuracy:{}'.format(train_time,test_time,accu)