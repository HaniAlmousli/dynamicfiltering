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

TYPE=tf.float32


class HiddenLayerWx(object):
    def __init__(self, seed, input, n_in, n_out, name="hidden", txtactivation="rectifier"):
        self.input = input
        self.txtactivation = txtactivation
        with tf.variable_scope(name):
            self.WIn = tf.get_variable(name="WIn", shape=[n_in, n_in*n_out],initializer=tf.contrib.layers.xavier_initializer(seed=seed),dtype=TYPE)
            self.WbIn = tf.Variable(tf.random_normal([n_in*n_out],seed=seed,dtype=TYPE),name='bIn')            
            self.b = tf.Variable(tf.random_normal([n_out],seed=seed,dtype=TYPE),name='b')

        self.lin_outputIn = tf.add(tf.matmul(self.input, self.WIn) , self.WbIn)
        self.outputIn = tf.nn.tanh(self.lin_outputIn)
        self.Wx = tf.reshape(self.outputIn,[-1,n_in,n_out])
        self.output = getActivation(self.txtactivation,tf.add(tf.einsum('ij,ijk->ik',input,self.Wx),self.b))
        
        
class HiddenLayer(object):
    def __init__(self, seed, input, n_in, n_out, name="hidden", txtactivation="rectifier"):
        self.input = input
        self.txtactivation = txtactivation
        with tf.variable_scope(name):
            self.W = tf.get_variable(name="W", shape=[n_in, n_out],initializer=tf.contrib.layers.xavier_initializer(seed=seed),dtype=TYPE)
            self.b = tf.Variable(tf.random_normal([n_out],seed=seed,dtype=TYPE),name='b')

        self.lin_output = tf.add(tf.matmul(self.input, self.W) , self.b)
        self.output = getActivation(txtactivation,self.lin_output)
        self.params = [self.W, self.b]    

class MLPWx(object):
    def __init__(self,x,y,conf, n_in,outClass=10):
        self.conf=conf
        self.infoLayers=conf['infoLayers']
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
                    HiddenLayerWx(seed=seed, input=inputTensor, n_in=currentInputSize, n_out=self.infoLayers[i],
                                    name="hidden"+str(i), txtactivation=self.activation[i]))
            seed +=1
            currentInputSize = self.infoLayers[i]
            inputTensor = self.layers[-1].output
        
        self.layers.append(
                    HiddenLayer(seed=seed, input=inputTensor, n_in=currentInputSize, n_out=outClass,
                                name = 'MLPoutput',txtactivation="linear"))
  
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                             labels=self.y, logits=self.layers[-1].output, name='cross_entropy_per_example')
        self.cost = tf.reduce_mean(self.cross_entropy, name='cross_entropy')+conf['lambda']* tf.norm(self.layers[0].WIn)

        self.y_pred = tf.equal(tf.cast(tf.argmax(self.layers[-1].output, 1),'int64'), self.y)

      

        self.sum_accuracy = tf.reduce_sum(tf.cast(self.y_pred, TYPE))
        
        self.accuracy =  100*tf.reduce_mean(tf.cast(self.y_pred, TYPE))             
