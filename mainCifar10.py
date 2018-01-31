import tensorflow as tf
import numpy as np
from MLPWD import *
from loader import *
import pickle
import gzip
import pdb

TYPE=tf.float32



conf={
    'infoLayers':[300],
    'rank':[300],
    'activation': ['linear'],
    'n_epochs':100,
    'batch_size':100,
    'learning_rate':1e-3,
    'l2':0,
    'seed':123,
    'optim':'rmsprop',
    'lrdr':0.96,
    
     #rmsprop
    'decay_rate':0.99,
    'outpath':'/home/hani/ModelOut/dynamicfiltering/cifar10WD'
}

bs= conf['batch_size']


print ("Loading  data")
tr,te = cifar10Normalized(grayscale=False)

print ("Data was loaded")
sampleSize,featureSize = tr[0].shape
testSize =te[0].shape[0]
batch_order = np.arange(int(sampleSize / conf['batch_size']))



mode = tf.Variable(1,trainable=False,name='mode') # 1:tr 2:va 3:te
index = tf.Variable(0,trainable=False,name='index')
# learning_rate = tf.Variable(conf['learning_rate'],trainable=False,name='learning_rate')
batch = tf.Variable(0, dtype='int32',trainable=False)
learning_rate = tf.train.exponential_decay(
      conf['learning_rate'],                # Base learning rate.
      batch * conf['batch_size'],           # Current index into the dataset.
      sampleSize,                           # Decay step.
      conf['lrdr'],                # Decay rate.
      staircase=False)
# pdb.set_trace()
trXTensor = tf.Variable(tr[0],name='trXTensor',trainable=False)
teXTensor = tf.Variable(te[0],name='teXTensor',trainable=False)

trYTensor = tf.Variable(tr[1],name='trYTensor',trainable=False)
teYTensor = tf.Variable(te[1],name='teYTensor',trainable=False)

bsValTe=250
tebatch_order = np.arange (int(testSize / bsValTe))

x = tf.cond(tf.equal(mode,1), 
               lambda:tf.slice(trXTensor,[index*bs,0],[bs,featureSize]),
               lambda:tf.slice(teXTensor,[index*bsValTe,0],[bsValTe,featureSize]))

y = tf.cond(tf.equal(mode,1), 
                   lambda:tf.slice(trYTensor,[index*bs],[bs]), 
                   lambda:tf.slice(teYTensor,[index*bsValTe],[bsValTe]))


model = MLPWD(x,y,conf, n_in=featureSize, outClass=10)                     

if conf['optim'] =='rmsprop':
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,decay=conf['decay_rate']).minimize(model.cost,
                                                                                              global_step=batch)
else:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(model.cost,
                                                                                        global_step=batch)


init_op = tf.group(tf.global_variables_initializer())

sess =tf.Session()

sess.run(init_op)
saver = tf.train.Saver(max_to_keep=2)

trCost=0
lrPatience=1
validFreq= 1
c_learning_rate = conf['learning_rate']


epoch= 0
output={
  'optimizer':optimizer,
  'lr':learning_rate,
  'cost': model.cost
  # 'tmp': model.layers[1].tmp
}

bestTest=0
while epoch < conf['n_epochs']:
  epoch += 1
  batch_LL=0
  for batch in batch_order:

      res= sess.run(output, feed_dict={index: batch, mode:1})
      batch_LL+=res['cost']
  batch_LL /=conf['batch_size']
  print ("Epoch {0} finished. learning rate: {1} TrLL: {2}".format(epoch,res['lr'], batch_LL)) 
  # pdb.set_trace()


testAccuracy=0
for batch in tebatch_order:
    testAccuracy+= sess.run(model.sum_accuracy, feed_dict={index: batch, mode:3})
testAccuracy = 100*(testAccuracy/testSize)
saver.save(sess, conf['outpath'],global_step=epoch)
saver.export_meta_graph(conf['outpath']+'.meta')
pickle.dump([conf,batch_LL,testAccuracy],open(conf['outpath']+'.pkl','wb'))
print ("\n ******** Epoch {0} finished. TeAccuracy: {1}\n".format(epoch, testAccuracy))

