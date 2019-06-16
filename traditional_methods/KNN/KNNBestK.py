import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from util import *
sns.set_style('darkgrid')

# read data
train_data=read_data("../../mnist/mnist_train/mnist_train_data")
test_data=read_data("../../mnist/mnist_test/mnist_test_data")
train_label=np.fromfile("../../mnist/mnist_train/mnist_train_label",dtype=np.uint8)
test_label=np.fromfile("../../mnist/mnist_test/mnist_test_label",dtype=np.uint8)

# Reduce the dimension of data to 50
pca=PCA(n_components=50)
train_data_reduce=pca.fit_transform(train_data)
test_data_reduce=pca.transform(test_data)

accuList=[]
# trying different K from 1 to 50
for i in xrange(1,51):
    print 'K={}'.format(i),
    # Using KNN with K=i
    knn=KNeighborsClassifier(n_neighbors=i,n_jobs=10)
    train_time,test_time,accu=run_model(knn,train_data_reduce,train_label,test_data_reduce,test_label)
    # print the performance
    print 'train time:{} test time:{} accuracy:{}'.format(train_time,test_time,accu)
    accuList.append(accu)

# Plot the accuracy figure.
plt.figure()
plt.plot(np.arange(1,51),accuList)
plt.xlabel('K')
plt.ylabel('accuracy')
plt.savefig('k-accu.pdf')