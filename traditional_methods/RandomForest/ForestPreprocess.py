import numpy as np
from sklearn.ensemble import RandomForestClassifier
from util import *

# read preprocessed data
train_data_half_aug=read_data("../../mnist/mnist_train/mnist_train_cc1.0_data")
test_data_half_aug=read_data("../../mnist/mnist_test/mnist_test_cc1.0_data")
train_data_aug=read_data("../../mnist/mnist_train/mnist_train_cc1.0_crop45_data")
test_data_aug=read_data("../../mnist/mnist_test/mnist_test_cc1.0_crop45_data")
train_label=np.fromfile("../../mnist/mnist_train/mnist_train_label",dtype=np.uint8)
test_label=np.fromfile("../../mnist/mnist_test/mnist_test_label",dtype=np.uint8)

# Performance on keep-largest-cc datasets
print 'largest CC:',
clf=RandomForestClassifier(n_estimators=200,n_jobs=10)
train_time,test_time,accu=run_model(clf,train_data_half_aug,train_label,test_data_half_aug,test_label)
print 'train time:{} test time:{} accuracy:{}'.format(train_time,test_time,accu)

# Performance crop-to-center datasets
print 'largest CC+Crop to center:',
clf=RandomForestClassifier(n_estimators=200,n_jobs=10)
train_time,test_time,accu=run_model(clf,train_data_aug,train_label,test_data_aug,test_label)
print 'train time:{} test time:{} accuracy:{}'.format(train_time,test_time,accu)