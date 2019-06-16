import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsOneClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from util import *
sns.set_style('darkgrid')

# read data
train_data=read_data("../../mnist/mnist_train/mnist_train_data")
test_data=read_data("../../mnist/mnist_test/mnist_test_data")
train_label=np.fromfile("../../mnist/mnist_train/mnist_train_label",dtype=np.uint8)
test_label=np.fromfile("../../mnist/mnist_test/mnist_test_label",dtype=np.uint8)

accuList=[]
runtimeList=[]
# trying different dimension numbers from 5 to 250
for i in xrange(5,251,5):
    # Dimensionality reduction to target dimension number
    pca=PCA(n_components=i)
    train_data_reduce=pca.fit_transform(train_data)
    test_data_reduce=pca.transform(test_data)
    print 'dim:{}'.format(i),
    # Using LinearSVC on i-dimensional data
    clf=OneVsOneClassifier(LinearSVC(),n_jobs=20)
    train_time,test_time,accu=run_model(clf,train_data_reduce,train_label,test_data_reduce,test_label)
    # print the performance
    print 'train time:{} test time:{} accuracy:{}'.format(train_time,test_time,accu)
    runtimeList.append(train_time+test_time)
    accuList.append(accu)

# Plot the accuracy figure.
plt.figure()
plt.plot(np.arange(5,251,5),accuList)
plt.xlabel('dimension')
plt.ylabel('accuracy')
plt.savefig('pca-accu.pdf') 

# Plot the runtime figure.
plt.figure()
plt.plot(np.arange(5,251,5),runtimeList,color='coral')
plt.xlabel('dimension')
plt.ylabel('runtime')
plt.savefig('pca-runtime.pdf')