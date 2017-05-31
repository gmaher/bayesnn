import tensorflow as tf
import tensorflow.contrib.bayesflow as bayes
import tensorflow.contrib.layers as layers
import tensorflow.contrib.distributions as distributions
import matplotlib.pyplot as plt
import numpy as np

def FC_bayes(x,shape,activation,scope,init=1e-3, bias=True):
    """
    initializer for a fully-connected layer with tensorflow
    inputs:
        -shape, (tuple), input,output size of layer
        -activation, (string), activation function to use
        -init, (float), multiplier for random weight initialization
    """
    with tf.variable_scope(scope):
        if init=='xavier':
            init = np.sqrt(2.0/(shape[0]+shape[1]))
        W_mu = tf.Variable(tf.random_uniform(shape, -init,init), name='W_mu')
        W_sig = tf.Variable(tf.ones(shape), name='W_sig')
        W_sig = tf.log(1.0+tf.exp(W_sig))
        W_noise = tf.placeholder(shape=shape,dtype=tf.float32,name='W_eps')
        b_mu = tf.Variable(tf.random_uniform([shape[1]],-init,init), name = 'b_mu')
        b_sig = tf.Variable(tf.ones([shape[1]]), name = 'b_sig')
        b_sig = tf.log(1.0+tf.exp(b_sig))
        b_noise = tf.placeholder(shape=shape[1],dtype=tf.float32,name='b_eps')

        W_samp = W_mu + W_sig*W_noise
        b_samp = b_mu + b_sig*b_noise

        #reg = tf.log(tf.reduce_prod(W_sig))+tf.log(tf.reduce_prod(b_sig))
        Norm_w = distributions.Normal(loc=W_mu,scale=W_sig)
        Norm_b = distributions.Normal(loc=b_mu,scale=b_sig)
        N01_w = distributions.Normal(loc=tf.zeros(shape=shape),
            scale=tf.ones(shape=shape))
        N01_b = distributions.Normal(loc=tf.zeros(shape=shape[1]),
            scale=tf.ones(shape=shape[1]))

        reg = tf.reduce_sum(distributions.kl(Norm_w,N01_w)) +\
            tf.reduce_sum(distributions.kl(Norm_b,N01_b))
        if activation == 'relu':
            activation = tf.nn.relu
        elif activation == 'sigmoid':
            activation = tf.nn.sigmoid
        elif activation == 'tanh':
            activation = tf.tanh
        else:
            activation = tf.identity
        if bias:
            h = tf.matmul(x,W_samp)+b_samp
        else:
            h = tf.matmul(x,W_samp)
        a = activation(h)
        return a,W_noise,b_noise, reg

X = 1.5*(2*np.random.rand(1000)-1)
Y = X**2 + np.random.randn((1000))*0.025+1
inds = (X<-0.3)+(X>0.3)
X_train = X[inds].reshape((-1,1))
Y_train = Y[inds].reshape((-1,1))
# plt.scatter(X,Y,linewidth=2)
# plt.show()
#
# inds = (X<-0.3)+(X>0.3)
# plt.scatter(X[inds],Y[inds],linewidth=2)
# plt.show()

x_tf = tf.placeholder(shape=[None,1],dtype=tf.float32)
y_tf = tf.placeholder(shape=[None,1],dtype=tf.float32)

a1,we1,be1,r1 = FC_bayes(x_tf,(1,50),'relu','fc1')
a2,we2,be2,r2 = FC_bayes(a1,(50,1),None,'fc2')

#loss = tf.reduce_mean(tf.square(y_tf-a2))
loss = tf.reduce_mean(tf.square(y_tf-a2))+0.0005*(r1+r2)

opt = tf.train.AdamOptimizer(1e-4)
train = opt.minimize(loss)
N = len(X_train)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(80000):
    sel = np.random.choice(N,size=32)
    x = X_train[sel]
    y = Y_train[sel]
    e1 = np.random.rand(1,50)
    e2 = np.random.rand(50)
    e3 = np.random.rand(50,1)
    e4 = np.random.rand(1)

    _,l = sess.run([train,loss],{x_tf:x,y_tf:y,we1:e1,be1:e2,we2:e3,be2:e4})
    print '{}: loss = {}'.format(i,l)

plt.figure()
Xar = np.arange(-1.5,1.5,3.0/100).reshape((-1,1))
plt.scatter(X_train,Y_train)
for i in range(20):
    e1 = np.random.rand(1,50)
    e2 = np.random.rand(50)
    e3 = np.random.rand(50,1)
    e4 = np.random.rand(1)

    yhat = sess.run(a2,{x_tf:Xar,we1:e1,be1:e2,we2:e3,be2:e4})
    plt.plot(Xar,yhat,linewidth=2,color='r')
plt.show()
