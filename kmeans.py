import tensorflow as tf
import numpy as np
import time

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

    def predict(self, data, c):

        assert type(data) is list or type(data) is np.array
        data = np.reshape(data,[-1,np.prod(np.shape(data)[1:])])
        assert type(c) is list or type(data) is np.array or type(data) is str
        c = np.reshape(c,[-1,np.prod(np.shape(c)[1:])])

        dl=None
        with tf.Session() as sess:
            dl=sess.run(self.laber,feed_dict={self.x:data,self.cl:c})
        return dl

    def fit(self, data, c, times_limit=-1, cnum=None):

        assert type(data) is list or type(data) is np.array
        data = np.reshape(data,[-1,np.prod(np.shape(data)[1:])])
        assert type(c) is list or type(data) is np.array
        if type(c) is str:
            if c=='random':
                assert type(cnum) is int
                c=[data[int(np.random.random()*(len(data)-1))] for i in range(cnum)]
        else:
            c = np.reshape(c,[-1,np.prod(np.shape(c)[1:])])

        nc,c,dl,times = c,None,None,0
        while (not np.all(np.equal(np.array(nc),np.array(c)))) and (times_limit==-1 or times<times_limit):
            c, times = nc, times+1
            with tf.Session() as sess:
                nc,dl=sess.run([self.fcl,self.laber],feed_dict={self.x:data,self.cl:nc})
        return nc,dl
    


data_c=[[np.random.random()*1000, np.random.random()*1000],
            [np.random.random()*1000, np.random.random()*1000],
            [np.random.random()*1000, np.random.random()*1000],
            [np.random.random()*1000, np.random.random()*1000],
            [np.random.random()*1000, np.random.random()*1000]]
data=[]
for i,j in data_c:
    for t in range(10000):
        data+=[[i*np.random.normal(1,0.2),j*np.random.normal(1,0.2)]]

np.random.shuffle(data)
c=[data[int(np.random.random()*(len(data)-1))] for i in range(5)]

k=kmeans()
print('start')
start=time.time()
c,dl=k.fit(data,c)
print(time.time()-start)

print(data_c)
print(c)
print(dl)
