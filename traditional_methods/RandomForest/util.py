import numpy as np
import time

def read_data(filename):
    # read data from binary file
    data=np.fromfile(filename,dtype=np.uint8)
    fig_w=45
    # Each sample image reshaped into a row vector.
    data=data.reshape(-1,fig_w*fig_w)
    # Normalized between 0 and 1.
    data=data/255.
    return data

def run_model(model,train_data,train_label,test_data,test_label):
    # Train and log the time.
    tic=time.time()
    model.fit(train_data,train_label)
    toc=time.time()
    train_time=toc-tic

    # Test and log the time.
    tic=time.time()
    test_pred=model.predict(test_data)
    toc=time.time()
    test_time=toc-tic

    # Calculate accuracy manually.
    accu=(test_pred==test_label).sum()/float(len(test_label))
    return train_time,test_time,accu