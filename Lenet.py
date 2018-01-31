import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Convolution layer 1. Define weight and bias first, then convolution and max_pooling.
# 5*5 denotes filter size; 1 denotes input no. of channel, 6 denotes output no. of channel = number of filters
conv1_w = tf.Variable(tf.random_normal([5, 5, 1, 6], mean=0.0, stddev=0.1) / 2)
conv1_b = tf.Variable(tf.zeros([6]))
# note its input are data and filter is weight; stride is of every dimension
conv1 = tf.nn.conv2d(x_image, conv1_w, strides=[1, 1, 1, 1], padding="VALID")
relu_1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))

# max pooling 1
pool1 = tf.nn.max_pool(relu_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
# After max pooling ,we got an image with 12*12 in size and 6 in depth

# Convolution layer 2
conv2_w = tf.Variable(tf.random_normal([5, 5, 6, 16], stddev=0.1) / 4)
conv2_b = tf.Variable(tf.zeros([16]))
conv2 = tf.nn.conv2d(pool1, conv2_w, strides=[1,1,1,1], padding="VALID")
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))

# max pooling 2. After this, we got an image with 4*4 in size and 16 in depth/channel
pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

# fc layer 1. Change pool2's output Tensor's dimension
pool2_flat = tf.reshape(pool2, [-1, 4 * 4 * 16])
fc1_w = tf.Variable(tf.random_normal([4*4*16, 128]) / 10)
fc1_b = tf.Variable(tf.zeros([128]))
fc1 = tf.nn.relu(tf.matmul(pool2_flat, fc1_w) + fc1_b)
fc1_drop = tf.nn.dropout(fc1, 0.7)

# fc layer 2
fc2_w = tf.Variable(tf.random_normal([128, 10]))
fc2_b = tf.Variable(tf.zeros([10]))
fc2 = tf.matmul(fc1_drop, fc2_w) + fc2_b      # no activation

# go through Softmax
y_pre = tf.nn.softmax(fc2)

# loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pre), axis=1))
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# compute acc
right_pre = tf.equal(tf.argmax(y, 1), tf.argmax(y_pre, 1))
acc = tf.reduce_mean(tf.cast(right_pre, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(40):
        for j in range(108):
            batch_x, batch_y = mnist.train.next_batch(512)
            a, b = sess.run([cross_entropy, train], feed_dict={x: batch_x, y:batch_y})
            print("loss", i, a)
        print(sess.run(acc, feed_dict={x: mnist.test.images, y:mnist.test.labels}))