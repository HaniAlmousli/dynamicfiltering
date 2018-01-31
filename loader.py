import pdb
import numpy as np
import os
import  pickle
import glob

def _grayscale(a):
    # return a.reshape(a.shape[0], 3, 32, 32).mean(1).reshape(a.shape[0], -1)
    tmp = a.reshape(a.shape[0], 3, 32, 32)
    return (tmp[:,0,:,:]*0.299  + tmp[:,1,:,:]*0.587 + tmp[:,2,:,:]*0.114 ).reshape([a.shape[0],-1])

def one_hot(x, n):
    x = np.array(x)
    return np.eye(n)[x]

def _load_batch_cifar10(filename, dtype='float32'):
    """
    load a batch in the CIFAR-10 format
    """
    data_dir = "/home/hani/Data/"
    data_dir_cifar10 = os.path.join(data_dir, "cifar-10-batches-py")

    path = os.path.join(data_dir_cifar10, filename)
    batch = pickle.load(open(path,'rb'),encoding='bytes')
    data = batch[b'data'] / 255.0 # scale between [0, 1]
    # labels = one_hot(batch[b'labels'], n=10) # convert labels to one-hot representation
    labels = np.asarray(batch[b'labels'],'int64') # convert labels to one-hot representation
    return data.astype(dtype), labels

def cifar10Normalized(dtype='float32', grayscale=False):
    # train
    x_train = []
    t_train = []
    for k in range(5):
        x, t = _load_batch_cifar10("data_batch_%d" % (k + 1), dtype=dtype)
        x_train.append(x)
        t_train.append(t)

    x_train = np.concatenate(x_train, axis=0)
    t_train = np.concatenate(t_train, axis=0)

    # test
    x_test, t_test = _load_batch_cifar10("test_batch", dtype=dtype)

    if grayscale:
        x_train = _grayscale(x_train)
        x_test = _grayscale(x_test)
    
    # pdb.set_trace()
    m = np.mean(x_train,axis=0)
    s = np.std (x_train,axis=0)
    x_train = (x_train-m)/s
    x_test =  (x_test -m)/s
    # pdb.set_trace()
    return [(x_train, t_train), (x_test, t_test)]

def cifar10UnNormalized(dtype='float32', grayscale=False):
    # train
    x_train = []
    t_train = []
    for k in range(5):
        x, t = _load_batch_cifar10("data_batch_%d" % (k + 1), dtype=dtype)
        x_train.append(x)
        t_train.append(t)

    x_train = np.concatenate(x_train, axis=0)
    t_train = np.concatenate(t_train, axis=0)

    # test
    x_test, t_test = _load_batch_cifar10("test_batch", dtype=dtype)

    if grayscale:
        x_train = _grayscale(x_train)
        x_test = _grayscale(x_test)
    
    # pdb.set_trace()
    return [(x_train, t_train), (x_test, t_test)]    



