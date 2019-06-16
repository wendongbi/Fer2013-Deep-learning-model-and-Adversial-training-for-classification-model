import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
# trying different decision tree numbers from 5 to 250
for n_model in xrange(5,251,5):
    print 'n_model:{}'.format(n_model),
    # Using random forest with n_model decision trees.
    clf=RandomForestClassifier(n_estimators=n_model,n_jobs=10)
    train_time,test_time,accu=run_model(clf,train_data,train_label,test_data,test_label)
    # print the performance
    print 'train time:{} test time:{} accuracy:{}'.format(train_time,test_time,accu)
    runtimeList.append(train_time+test_time)
    accuList.append(accu)

# Plot the accuracy figure.
plt.figure()
plt.plot(np.arange(5,251,5),accuList)
plt.xlabel('the number of trees')
plt.ylabel('accuracy')
plt.savefig('forest-accu.pdf')

# Plot the runtime figure.
plt.figure()
plt.plot(np.arange(5,251,5),runtimeList,color='coral')
plt.xlabel('the number of trees')
plt.ylabel('runtime')
plt.savefig('forest-runtime.pdf')