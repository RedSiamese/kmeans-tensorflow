import tensorflow as tf
import numpy as np

import time
from tensorflow.examples.tutorials.mnist import input_data
import cv2

from sklearn.cluster import k_means

class kmeans(object):

    def __init__(self):

        self.x=tf.placeholder(tf.float32, [None, None])
        self.cl=tf.placeholder(tf.float32, [None, None])

        cl2=tf.expand_dims(tf.transpose(self.cl), 0)
        x2=tf.expand_dims(self.x,-1)
        cl2=tf.tile(cl2,[tf.shape(self.x)[0]]+[1]*len(self.cl.get_shape().as_list()) )
        x2=tf.tile(x2,[1]*len(self.x.get_shape().as_list())+[tf.shape(self.cl)[0]])

        dis=tf.reduce_sum(tf.square(x2-cl2),1)
        dis=tf.to_float(tf.equal(dis-tf.expand_dims(tf.reduce_min(dis,1),1), 0.))
        self.laber=tf.argmax(dis,1)
        self.fcl=tf.matmul(tf.div(dis,tf.reduce_sum(dis,0)),self.x,True)
        self.stop_flag=tf.reduce_sum(self.fcl-self.cl)

    def predict(self, data, c):

        assert type(data) in [list, np.array, np.ndarray], 'Input data must be a list or an array'
        assert len(np.shape(data))>1, 'Input data shape must be at least 2-D'
        data = np.reshape(data,[-1,np.prod(np.shape(data)[1:])])
        assert type(c) in [list, np.array, np.ndarray], 'Input c must be a list or an array'
        c = np.reshape(c,[-1,np.prod(np.shape(c)[1:])])

        dl=None
        with tf.Session() as sess:
            dl=sess.run(self.laber,feed_dict={self.x:data,self.cl:c})
        return dl

    def fit(self, data, c, times_limit=-1, cnum=None):

        assert type(data) in [list, np.array, np.ndarray], 'Input data must be a list or an array'
        assert len(np.shape(data))>1, 'Input data shape must be at least 2-D'
        data = np.reshape(data,[-1,np.prod(np.shape(data)[1:])])
        assert type(c) in [list, np.array, np.ndarray, str], 'Input c must be a list or an array or \'random\''
        if type(c) is str and c is 'random':
            assert type(cnum) is int, 'Input cnum must be an int'
            indices=np.random.choice(len(data),cnum,replace=False)
            c = data[indices, :]
        else:
            c = np.reshape(c,[-1,np.prod(np.shape(c)[1:])])

        nc,c,dl,times,sf = c,None,None,0,1
        with tf.Session() as sess:
            while sf!=0. and (times_limit==-1 or times<times_limit):
                times = times+1
                nc,dl,sf=sess.run([self.fcl,self.laber,self.stop_flag],feed_dict={self.x:data,self.cl:nc})
        return nc,dl
    




data=input_data.read_data_sets("./MNIST", one_hot=True).test.images[0:1000,:]

k=kmeans()
print('start')

start=time.time()
cc,dl,_=k_means(np.array(data),10,max_iter=10000,init='random')
print(time.time()-start)

start=time.time()
cc,dl=k.fit(np.array(data),c='random',cnum=10)
print(time.time()-start)



for i in range(10):
    img=np.reshape(cc[i],(28,28))
    cv2.imshow('0',img)
    cv2.waitKey()
    for j in range(500):
        if dl[j]==i:
            img=data[j].reshape((28,28))
            cv2.imshow('1',img)
            cv2.waitKey()  

