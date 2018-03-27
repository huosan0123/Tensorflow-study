import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.examples.tutorials.mnist import input_data
import sys

"""
Using a simple CNN net to learn how to train and test in the same tf.Graph and tf.Session.
Not using reuse_all_variable, I will try it later!
"""
class Model(object):

    def __init__(self):
        self.xs = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.ys = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        self.lr = tf.placeholder(dtype=tf.float32)
        self.input_x = tf.reshape(self.xs, [-1, 28, 28, 1])
        with tf.variable_scope('cnn') as scope:
            logits = self.build_net()

        self.correct = tf.equal(tf.argmax(logits, 1), tf.argmax(self.ys, 1))
        self.acc =tf.reduce_mean(tf.cast(self.correct, dtype=tf.float32))

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.ys))
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


    def build_net(self):   # very simple net, just to test
        with tf.variable_scope('net'):
            a = tcl.conv2d(self.input_x, 32, [3, 3], [1, 1], padding="SAME")
            a_pool = tf.nn.max_pool(a, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", name='pooling_a')
            b = tcl.conv2d(a, 128, [5, 5], [1, 1], padding="SAME")
            b_pool = tf.nn.max_pool(b, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", name='pooling_b')

            b_flat = tcl.flatten(b_pool)

            fc_1 = tcl.fully_connected(b_flat, 512)
            fc_2 = tcl.fully_connected(fc_1, 10)
        return fc_2



with tf.Graph().as_default():
    with tf.Session() as sess:
        net = Model()
        sess.run(tf.global_variables_initializer())

        def test():
            val_loss, val_acc = 0.0, 0.0
            step = 0
            for i in range(100):
                test_x, test_y = mnist.test.next_batch(100)
                _acc, _loss = sess.run([net.acc, net.loss], feed_dict={net.xs:test_x, net.ys: test_y})
                val_loss += _loss
                val_acc += _acc
            val_loss /= 100
            val_acc /= 100
            print("[TEST] loss={}, acc={}".format(val_loss, val_acc))

        train_acc, train_loss = 0.0, 0.0
        run_ops = [net.acc, net.loss, net.train_op]
        mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
        for i in range(1000):
            batch_x, batch_y = mnist.train.next_batch(100)
            _acc, _loss, _ = sess.run(run_ops, feed_dict={net.xs: batch_x, net.ys:batch_y, net.lr:0.001})

            train_acc += _acc
            train_loss += _loss

            if i % 50 == 0:
                train_loss /= 50
                train_acc /= 50
                print('[Train] loss={}, acc={}'.format(train_loss, train_acc))
                loss = 0
                acc = 0
                sys.stdout.flush()

            if i % 50 == 0:
                test()
