#!/usr/bin/python
#!coding=utf-8
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.examples.tutorials.mnist import input_data
import sys
"""
TF version:1.3.1; python version: 2.7.9; GPU P40
This file aims to compare the batch_norm layer's effect.
Really find something different!
Run this with mnist_bn0.py
My code is not clean, but I will keep going!
"""

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
training = tf.placeholder(dtype=tf.bool)
input_x = tf.reshape(x, [-1, 28, 28, 1])
phase = training

with tf.variable_scope('mnist'):
    a = tcl.conv2d(input_x, 16, [3, 3], [1, 1], padding='SAME')
    a_bn = tf.layers.batch_normalization(a, training=phase)
    #a_relu = tf.nn.relu(a_bn)
    b = tcl.conv2d(input_x, 16, [5, 5], [1, 1], padding='SAME')
    b_bn = tf.layers.batch_normalization(b, training=phase)
    #b_relu = tf.nn.relu(b_bn)
    c = tf.concat([a_bn, b_bn], axis=3)

    a2 = tcl.conv2d(c, 64, [3, 3], [1, 1], padding='SAME')
    a2_bn = tf.layers.batch_normalization(a2, training=phase)
    b2 = tcl.conv2d(c, 64, [5, 5], [1, 1], padding='SAME')
    b2_bn = tf.layers.batch_normalization(b2, training=phase)
    c2 = tf.concat([a2_bn, b2_bn], axis=3)

    a3 = tcl.conv2d(c2, 128, [3, 3], [1, 1], padding="SAME")
    a3_bn = tf.layers.batch_normalization(a3, training=phase)
    b3 = tcl.conv2d(c2, 128, [5, 5], [1, 1], padding='SAME')
    b3_bn = tf.layers.batch_normalization(b3, training=phase)
    c3 = tf.concat([a3_bn, b3_bn], axis=3)

    c3_flat = tcl.flatten(c3)
    print(c3_flat.shape)
    #fc1 = tcl.fully_connected(c3_flat, 512)
    out = tcl.fully_connected(c3_flat, 10, activation_fn=None)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = tf.train.AdamOptimizer().minimize(loss)

correct = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_acc, train_loss = 0.0, 0.0
    for i in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        _, _loss, _acc = sess.run([train_op, loss, acc], feed_dict={x: batch_x, y: batch_y, training:True})
        train_acc += _acc
        train_loss += _loss
        if (i+1) % 50 == 0:
            print('STEP={}, loss is {}, acc is {} '.format(i, train_loss/50, train_acc/50))
            train_acc = 0.0
            train_loss = 0.0
            sys.stdout.flush()

        if (i+1) % 100 == 0:
            accuracy, val_loss = 0.0, 0.0
            for j in range(100):
                _x, _y = mnist.test.next_batch(100)
                right, x_loss = sess.run([acc, loss], feed_dict={x:_x, y:_y, training:False})
                accuracy += right
                val_loss += x_loss
            print('[TEST]step={}, || loss={}, acc={}'.format(i, val_loss/100, accuracy/100))
