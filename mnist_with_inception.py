#!/usr/bin/python
#!coding=utf-8
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
lr = tf.placeholder(dtype=tf.float32, shape=None)
input_x = tf.reshape(x, [-1, 28, 28, 1])

with tf.variable_scope('mnist'):
    a = tcl.conv2d(input_x, 16, [3, 3], [1, 1], padding='SAME')
    a_bn = tf.layers.batch_normalization(a)
    b = tcl.conv2d(input_x, 16, [5, 5], [1, 1], padding='SAME')
    b_bn = tf.layers.batch_normalization(b)
    c = tf.concat([a_bn, b_bn], axis=3)
    """
    a2 = tcl.conv2d(c, 64, [3, 3], [1, 1], padding='SAME')
    a2_bn = tf.layers.batch_normalization(a2)
    b2 = tcl.conv2d(c, 64, [5, 5], [1, 1], padding='SAME')
    b2_bn = tf.layers.batch_normalization(b2)
    c2 = tf.concat([a2_bn, b2_bn], axis=3)
    """
    a3 = tcl.conv2d(c, 128, [3, 3], [1, 1], padding="SAME")
    a3_bn = tf.layers.batch_normalization(a3)
    b3 = tcl.conv2d(c, 128, [5, 5], [1, 1], padding='SAME')
    b3_bn = tf.layers.batch_normalization(b3)
    c3 = tf.concat([a3_bn, b3_bn], axis=3)

    c3_flat = tcl.flatten(c3)
    print(c3_flat.shape)
    #fc1 = tcl.fully_connected(c3_flat, 512)
    weight = tf.get_variable('w', shape=[200704, 512])
    bias = tf.get_variable('b', shape=[512])
    fc1 = tf.nn.relu(tf.matmul(c3_flat, weight) + bias)
    #fc2 = tcl.fully_connected(fc1, 32)

    out = tcl.fully_connected(fc1, 10, activation_fn = None)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)

correct = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    base_lr = 0.001
    for i in range(400):
        batch_x, batch_y = mnist.train.next_batch(100)
        _, _loss = sess.run([train_op, loss], feed_dict={x: batch_x, y: batch_y, lr:base_lr})
        print('loss is {}'.format(_loss))
        if i%100 == 0 and i != 0:
            base_lr *= 0.9

#        if i % 10000 == 0:
 #           print("step {} loss is {}".format(i, _loss))
  #          accuracy, val_loss = 0.0, 0.0
   #         for j in range(100):
    #            _x, _y = mnist.test.next_batch(100)
     #           right, x_loss = sess.run([acc, loss], feed_dict={x:_x, y:_y})
      #          accuracy += right
       #         val_loss += x_loss
        #    print('[TEST]step={},lr ={} ||  acc={}, loss={}'.format(i/50, base_lr, accuracy, val_loss/100))
