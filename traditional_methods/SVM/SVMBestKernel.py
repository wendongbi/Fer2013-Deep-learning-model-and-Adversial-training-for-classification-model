import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsOneClassifier
from util import *

# read data
train_data=read_data("../../mnist/mnist_train/mnist_train_data")
test_data=read_data("../../mnist/mnist_test/mnist_test_data")
train_label=np.fromfile("../../mnist/mnist_train/mnist_train_label",dtype=np.uint8)
test_label=np.fromfile("../../mnist/mnist_test/mnist_test_label",dtype=np.uint8)

# Reduce data to 120 dimension.
pca=PCA(n_components=120)
train_data_reduce=pca.fit_transform(train_data)
test_data_reduce=pca.transform(test_data)

# Trying different kernels...
for kernel in ['linear','poly','rbf','sigmoid']:
    print 'kernel:{}'.format(kernel),
    # Using SVC with the target kernel
    clf=OneVsOneClassifier(SVC(kernel=kernel),n_jobs=20)
    train_time,test_time,accu=run_model(clf,train_data_reduce,train_label,test_data_reduce,test_label)
    # Print the performance.
    print 'train time:{} test time:{} accuracy:{}'.format(train_time,test_time,accu)