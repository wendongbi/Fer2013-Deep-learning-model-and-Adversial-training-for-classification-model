import numpy as np
from util import *
from sklearn.naive_bayes import MultinomialNB

# read data
train_data=read_data("../../mnist/mnist_train/mnist_train_data")
test_data=read_data("../../mnist/mnist_test/mnist_test_data")
train_label=np.fromfile("../../mnist/mnist_train/mnist_train_label",dtype=np.uint8)
test_label=np.fromfile("../../mnist/mnist_test/mnist_test_label",dtype=np.uint8)

# Using naive bayes
bayes=MultinomialNB()
train_time,test_time,accu=run_model(bayes,train_data,train_label,test_data,test_label)
# print the performance
print 'train time:{} test time:{} accuracy:{}'.format(train_time,test_time,accu)