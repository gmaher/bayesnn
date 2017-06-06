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
        factor = np.sqrt(2.0/shape[0])
        init = np.log(np.exp(factor)-1)
        W_mu = tf.Variable(tf.zeros(shape), name='W_mu')
        W_sig = tf.Variable(tf.ones(shape)*init, name='W_sig')
        W_sig = tf.log(1.0+tf.exp(W_sig))
        W_noise = tf.placeholder(shape=shape,dtype=tf.float32,name='W_eps')
        b_mu = tf.Variable(tf.zeros([shape[1]]), name = 'b_mu')
        b_sig = tf.Variable(tf.ones([shape[1]])*init, name = 'b_sig')
        b_sig = tf.log(1.0+tf.exp(b_sig))
        b_noise = tf.placeholder(shape=shape[1],dtype=tf.float32,name='b_eps')

        W_samp = W_mu + W_sig*W_noise
        b_samp = b_mu + b_sig*b_noise

        #reg = tf.log(tf.reduce_prod(W_sig))+tf.log(tf.reduce_prod(b_sig))
        Norm_w = distributions.Normal(loc=W_mu,scale=W_sig)
        Norm_b = distributions.Normal(loc=b_mu,scale=b_sig)
        N01_w = distributions.Normal(loc=tf.zeros(shape=shape),
            scale=tf.ones(shape=shape)*factor)
        N01_b = distributions.Normal(loc=tf.zeros(shape=shape[1]),
            scale=tf.ones(shape=shape[1])*factor)

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
Y = 0.5*np.sin(X*2*np.pi)+X**2 +1

inds = (X<-0.3)+(X>0.3)
inds2 = (X>=-0.3)*(X<=0.3)
Y[inds] += np.random.randn(1000)[inds]*0.2
Y[inds2] += np.random.randn(1000)[inds2]*0.2
X_train = X.reshape((-1,1))
Y_train = Y.reshape((-1,1))
# plt.scatter(X,Y,linewidth=2)
# plt.show()
#
# inds = (X<-0.3)+(X>0.3)
# plt.scatter(X[inds],Y[inds],linewidth=2)
# plt.show()
N1 = 100
N2 = 1
x_tf = tf.placeholder(shape=[None,1],dtype=tf.float32)
y_tf = tf.placeholder(shape=[None,1],dtype=tf.float32)

a1,we1,be1,r1 = FC_bayes(x_tf,(1,N1),'tanh','fc1')
a2,we2,be2,r2 = FC_bayes(a1,(N1,N2),None,'fc2')
a3,we3,be3,r3 = FC_bayes(a2,(N2,1),None,'fc3')
#loss = tf.reduce_mean(tf.square(y_tf-a2))
loss = tf.reduce_mean(tf.square(y_tf-a2))+0.000001*(r1+r2)

opt = tf.train.AdamOptimizer(5e-3)
#opt = tf.train.RMSPropOptimizer(5e-3)
train = opt.minimize(loss)
N = len(X_train)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(10000):
    sel = np.random.choice(N,size=100)
    x = X_train[sel]
    y = Y_train[sel]
    e1 = np.random.randn(1,N1)
    e2 = np.random.randn(N1)
    e3 = np.random.randn(N1,N2)
    e4 = np.random.randn(N2)
    e5 = np.random.randn(N2,1)
    e6 = np.random.randn(1)

    _,l = sess.run([train,loss],{x_tf:x,y_tf:y,we1:e1,be1:e2,we2:e3,be2:e4,we3:e5,be3:e6})
    print '{}: loss = {}'.format(i,l)

plt.figure()
Xar = np.arange(-2.5,2.5,5.0/100).reshape((-1,1))
plt.scatter(X_train,Y_train)
for i in range(20):
    e1 = np.random.randn(1,N1)
    e2 = np.random.randn(N1)
    e3 = np.random.randn(N1,N2)
    e4 = np.random.randn(N2)
    e5 = np.random.randn(N2,1)
    e6 = np.random.randn(1)

    yhat = sess.run(a2,{x_tf:Xar,we1:e1,be1:e2,we2:e3,be2:e4,we3:e5,be3:e6})
    plt.plot(Xar,yhat,linewidth=2,color='r')
plt.show()
