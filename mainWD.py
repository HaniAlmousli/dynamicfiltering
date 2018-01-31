import tensorflow as tf
import numpy as np
from MLPWD import *
import pickle
import gzip
import pdb



conf={
    'infoLayers':[300],
    'rank':[300],
    'activation': ['linear'],
    'n_epochs':100,
    'batch_size':100,
    'learning_rate':1e-3,
    'seed':123,
    'optim':'rmsprop',
    # 'lambda':0,
    'lrdr':0.95,#0.95
     #adam
    'b1':0.9,
    'b2':0.99,
     #rmsprop
    'decay_rate':0.95,
    'outpath':'/home/hani/ModelOut/dynamicfiltering/WD/MNISTWD'
}

bs= conf['batch_size']


print ("Loading  data")
# tr,va,te = cifar10WithValid()
with gzip.open('/home/hani/Data/mnist.pkl.gz', 'rb') as f:
    try:
        tr, va, te = pickle.load(f, encoding='latin1')
    except:
        tr, va, te = pickle.load(f)

print(va[0].shape[0],te[0].shape[0])
# assert (va[0].shape[0]%bs ==0) & (te[0].shape[0]%bs ==0)

print ("Data was loaded")
sampleSize,featureSize = tr[0].shape
validSize=va[0].shape[0]
testSize =te[0].shape[0]
batch_order = np.arange(int(sampleSize / conf['batch_size']))



mode = tf.Variable(1,trainable=False,name='mode') # 1:tr 2:va 3:te
index = tf.Variable(0,trainable=False,name='index')
# learning_rate = tf.Variable(conf['learning_rate'],trainable=False,name='learning_rate')
batch = tf.Variable(0, dtype='int32')
learning_rate = tf.train.exponential_decay(
      conf['learning_rate'],                # Base learning rate.
      batch * conf['batch_size'],           # Current index into the dataset.
      sampleSize,                           # Decay step.
      conf['lrdr'],                # Decay rate.
      staircase=False)
# pdb.set_trace()
trXTensor = tf.Variable(tr[0],name='trXTensor',trainable=False)
vaXTensor = tf.Variable(va[0],name='vaXTensor',trainable=False)
teXTensor = tf.Variable(te[0],name='teXTensor',trainable=False)

trYTensor = tf.Variable(tr[1],name='trYTensor',trainable=False)
vaYTensor = tf.Variable(va[1],name='vaYTensor',trainable=False)
teYTensor = tf.Variable(np.asarray(te[1],'int64'),name='teYTensor',trainable=False)


bsValTe=100
valbatch_order = np.arange(int(validSize / bsValTe))
tebatch_order = np.arange (int(testSize / bsValTe))

x = tf.cond(tf.equal(mode,1),
             lambda:tf.slice(trXTensor,[index*bs,0],[bs,featureSize]), 
             lambda:tf.cond(tf.equal(mode,2), 
                             lambda:tf.slice(vaXTensor,[index*bsValTe,0],[bsValTe,featureSize]),
                             lambda:tf.slice(teXTensor,[index*bsValTe,0],[bsValTe,featureSize])))

y = tf.cond(tf.equal(mode,1),
             lambda:tf.slice(trYTensor,[index*bs],[bs]), 
             lambda:tf.cond(tf.equal(mode,2), 
                             lambda:tf.slice(vaYTensor,[index*bsValTe],[bsValTe]),
                             lambda:tf.slice(teYTensor,[index*bsValTe],[bsValTe])))

model = MLPWD(x,y,conf, n_in=featureSize, outClass=10)           
# model = RMLPBN(x,y,conf, n_in=featureSize, outClass=10)           

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
bestRes=()
bestValid = -np.inf
trCost=0
clrDec = 0
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

while epoch < conf['n_epochs']:
    epoch += 1
    batch_LL=0
    for batch in batch_order:
        # pdb.set_trace()

        res= sess.run(output, feed_dict={index: batch, mode:1})
        # res= sess.run(output, feed_dict={index: batch, mode:1,model.phase:True})
        batch_LL+=res['cost']
    batch_LL /=conf['batch_size']
    print ("Epoch {0} finished. learning rate: {1} TrLL: {2}".format(epoch,res['lr'], batch_LL)) 

    if epoch%validFreq ==0:
        valAccuracy=0
        for batch in valbatch_order:
            valAccuracy+= sess.run(model.sum_accuracy, feed_dict={index: batch, mode:2})
            # valAccuracy+= sess.run(model.sum_accuracy, feed_dict={index: batch, mode:2,model.phase:False})
        valAccuracy = 100*(valAccuracy/validSize)
        print (">> Epoch {0} finished. TrLL: {1}, VaAccuracy {2}".format(epoch, batch_LL, valAccuracy))
        if valAccuracy>=bestValid:
            bestValid=valAccuracy
            testAccuracy=0
            for batch in tebatch_order:
                testAccuracy+= sess.run(model.sum_accuracy, feed_dict={index: batch, mode:3})
                # testAccuracy+= sess.run(model.sum_accuracy, feed_dict={index: batch, mode:3,model.phase:False})
            # pdb.set_trace()
#             pickle.dump(sess.run(model.layers[0].params[0]),open('/home/hani/ModelOut/dynamicfiltering/firstTrial/regular.pkl','wb'))
            testAccuracy = 100*(testAccuracy/testSize)
            print ("\n ******** Epoch {0} finished. TeAccuracy: {1}, VaAccuracy {2}\n".format(epoch, testAccuracy, valAccuracy))
            clrDec = 0

            saver.save(sess, conf['outpath'],global_step=epoch)
            # Generates MetaGraphDef.
            saver.export_meta_graph(conf['outpath']+'.meta')
            pickle.dump([conf,batch_LL,valAccuracy,testAccuracy],open(conf['outpath']+'.pkl','wb'))
    # pdb.set_trace()
