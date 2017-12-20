#!coding=UTF-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, acti_fun=None):
    #rows = in_size, cols = out_size
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    #b一般不为0,所以需要加0.1
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if acti_fun is None:
        #未提供额外函数的时候,默认为线性函数
        outputs = Wx_plus_b
    else:
        #根据提供的激活函数处理结果
        outputs = acti_fun(Wx_plus_b)
    return outputs

#np.newaxis把vector变为300*1的mat, 注意更改生成的数据的float类型
x_data = np.linspace(-10,10,100000, dtype=np.float32)[:, np.newaxis]
#设置一个noise的值,参数分别未均值,方差,shape
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

#None means无论输入多少sample都ok.因为输入只有一个特征，所以这里是1
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#隐藏层有10个神经元,所以out_size=10, 浅层确实多种激活函数都可以使用
layer1 = add_layer(x_data, 1, 10, acti_fun=tf.nn.tanh)
prediction = add_layer(layer1,10, 1, acti_fun=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

#内部是learning rate,后面是train的目的,目的是最小化loss
train_step = tf.train.AdamOptimizer(0.2).minimize(loss)

#init = tf.initialize_all_variables()  这个方式可以废弃了
init = tf.global_variables_initializer() #定义初始化全局所有变量的节点

sess = tf.Session()
sess.run(init)  #执行初始化

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_data, y_data)
plt.ion()
plt.show(block=False) #在plt.show之后不让程序暂停

for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0:
        try:
            ax.lines.remove(lines[0])
        except:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs:x_data, ys:y_data})
        #must have 'lines=',否则前面的remove没有对象
        lines = ax.plot(x_data, prediction_value, 'g-', lw=3)
        #暂停方便看到plt的过程, 再用plt.show是没有效果的了.
        plt.pause(0.2)