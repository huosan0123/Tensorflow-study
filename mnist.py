#!coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function:
        outputs = activation_function(Wx_plus_b)
    else:
        outputs = Wx_plus_b
    return outputs

# 不规定有多少个sample,但是规定每个sample的向量长度
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# 定义loss函数,这里以cross entropy作为loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))

# train step
train = tf.train.GradientDescentOptimizer(0.65).minimize(cross_entropy)

def compute_accuracy(x_vs, y_vs):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:x_vs})
    right_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_vs,1))
    accuracy = tf.reduce_mean(tf.cast(right_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:x_vs, ys:y_vs})
    return result

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={xs: batch_xs, ys:batch_ys})
        if i % 50 == 0:
            print(compute_accuracy(mnist.test.images, mnist.test.labels))

# 很奇怪啊,这个后面怎么就降到了0.098,特别小的准确率. 是因为过拟合了吗?