import tensorflow as tf
import numpy as np
import pdb

rng = np.random.RandomState(123)



def getActivation(txtactivation,x):
    if txtactivation == "tanh":
        return tf.nn.tanh(x)
    elif txtactivation == "sigmoid":
        return tf.nn.sigmoid(x)
    elif txtactivation == "rectifier":
        return tf.nn.relu(x)
    elif txtactivation == "rectifier1":
        # return tf.where(tf.less_equal(0,1),o, tf.zeros_like(o))
        return tf.minimum(tf.nn.relu(x),1)
        # return tf.nn.relu6(x)
    elif txtactivation == "softmax":
        return tf.nn.softmax(x)
    elif txtactivation == "linear":
        return x
    
class HiddenLayerWxsd(object):
    def __init__(self, seed, input, n_in, n_out, rank=3, name="hidden", txtactivation="rectifier"):
        self.input = input
        self.txtactivation = txtactivation
        with tf.variable_scope(name):
            #WL is global
            self.WL = tf.get_variable(name="WL", shape=[n_in, rank],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=seed))
            #WR is learnt from x
            self.WR = tf.get_variable(name="WR", shape=[n_in, rank*n_out],
                          initializer=tf.contrib.layers.xavier_initializer(seed=seed+1))

            # self.WL = tf.Variable(tf.random_normal([n_in, n_in*rank] ,seed=seed ,stddev=0.0001),name="WL")
            # self.WR = tf.Variable(tf.random_normal([n_in, rank*n_out],seed=seed ,stddev=0.0001),name="WR")
            
            self.WbIn = tf.Variable(tf.random_normal([rank*n_out],seed=seed),name='bIn')
            
            self.b = tf.Variable(tf.random_normal([n_out],seed=seed),name='b')
            print(name+" has: "+str(n_in*n_in*rank+n_in*rank*n_out))
        
        self.dynamic_part = tf.nn.relu(tf.add(tf.matmul(self.input, self.WR),self.WbIn))
        self.dynamic_part_reshaped = tf.reshape(self.dynamic_part,[-1,rank,n_out])
        self.Wx_linear = tf.einsum('ijk,lj->ilk',self.dynamic_part_reshaped,self.WL)
        self.Wx=self.Wx_linear
        # self.Wx = tf.nn.tanh(self.Wx_linear)
        self.output = getActivation(self.txtactivation,tf.add(tf.einsum('ij,ijk->ik',input,self.Wx),self.b))
    
class HiddenLayer(object):
    def __init__(self, seed, input, n_in, n_out, name="hidden", txtactivation="rectifier"):
        self.input = input
        self.txtactivation = txtactivation
        with tf.variable_scope(name):
            self.W = tf.get_variable(name="W", shape=[n_in, n_out],initializer=tf.contrib.layers.xavier_initializer(seed=seed))
            self.b = tf.Variable(tf.random_normal([n_out],seed=seed),name='b')

        self.lin_output = tf.add(tf.matmul(self.input, self.W) , self.b)
        self.output = getActivation(txtactivation,self.lin_output)
        self.params = [self.W, self.b]
        
class MLPWD(object):
    def __init__(self,x,y,conf, n_in,outClass=10):
        self.conf=conf
        self.infoLayers=conf['infoLayers']
        self.rank = conf['rank']
        self.layers=[] #contains the actual model
        self.activation = self.conf['activation']
        self.n_in = n_in
        self.x = x
        self.y = y
        
        seed = conf['seed']
        currentInputSize= n_in
        inputTensor=self.x

        for i in range(len(self.infoLayers)):
            self.layers.append(
                    HiddenLayerWxsd(seed=seed, input=inputTensor, n_in=currentInputSize, n_out=self.infoLayers[i],rank=self.rank[i],
                                    name="hiddenwxsd"+str(i), txtactivation=self.activation[i]))
            seed +=1
            currentInputSize = self.infoLayers[i]
            inputTensor = self.layers[-1].output
        
        self.layers.append(
                    HiddenLayer(seed=seed, input=inputTensor, n_in=currentInputSize, n_out=outClass,
                                name = 'MLPoutput',txtactivation="linear"))
        # denom = tf.reduce_sum(tf.exp(self.layers[-1].output),axis=1)
        # self.probs = tf.exp(self.layers[-1].output)/denom
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                             labels=self.y, logits=self.layers[-1].output, name='cross_entropy_per_example')
        self.cost = tf.reduce_mean(self.cross_entropy, name='cross_entropy')

        self.y_pred = tf.equal(tf.cast(tf.argmax(self.layers[-1].output, 1),'int64'), self.y)
        self.sum_accuracy = tf.reduce_sum(tf.cast(self.y_pred, tf.float32))
        self.accuracy =  100*tf.reduce_mean(tf.cast(self.y_pred, tf.float32))             
