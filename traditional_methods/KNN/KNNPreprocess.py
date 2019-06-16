import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from util import *

# read preprocessed data
train_data_half_aug=read_data("../../mnist/mnist_train/mnist_train_cc1.0_data")
test_data_half_aug=read_data("../../mnist/mnist_test/mnist_test_cc1.0_data")
train_data_aug=read_data("../../mnist/mnist_train/mnist_train_cc1.0_crop45_data")
test_data_aug=read_data("../../mnist/mnist_test/mnist_test_cc1.0_crop45_data")
train_label=np.fromfile("../../mnist/mnist_train/mnist_train_label",dtype=np.uint8)
test_label=np.fromfile("../../mnist/mnist_test/mnist_test_label",dtype=np.uint8)

# Dimensionality reduction of keep-largest-cc datasets
pca=PCA(n_components=50)
train_data_reduce=pca.fit_transform(train_data_half_aug)
test_data_reduce=pca.transform(test_data_half_aug)

# Performance on keep-largest-cc datasets
print 'largest CC:',
knn=KNeighborsClassifier(n_neighbors=5,n_jobs=10)
train_time,test_time,accu=run_model(knn,train_data_reduce,train_label,test_data_reduce,test_label)
print 'train time:{} test time:{} accuracy:{}'.format(train_time,test_time,accu)

# Dimensionality reduction of crop-to-center datasets
pca=PCA(n_components=50)
train_data_reduce=pca.fit_transform(train_data_aug)
test_data_reduce=pca.transform(test_data_aug)

# Performance crop-to-center datasets
print 'largest CC+Crop to center:',
knn=KNeighborsClassifier(n_neighbors=5,n_jobs=10)
train_time,test_time,accu=run_model(knn,train_data_reduce,train_label,test_data_reduce,test_label)
print 'train time:{} test time:{} accuracy:{}'.format(train_time,test_time,accu)