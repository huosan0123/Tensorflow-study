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

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def biases_variable(shapes):
    initial = tf.constant(0.1, shape=shapes)
    return tf.Variable(initial)

def con2d(x, W):
    # strides = [1, x_movement, y_movement, 1]
    # must have strides[1] = strides[4] = 1
    return tf.nn.conv2d(x, W, [1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# 不规定有多少个sample,但是规定每个sample的向量长度
xs = tf.placeholder(tf.float32, [None, 784])/255
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

#-1 means don't process number of samples
# 1 means the number of channels, which here is black or white, so set to 1
x_image = tf.reshape(xs, [-1, 28, 28, 1])
print(x_image.shape)  #[n_sample, 28,28, 1]

# patch is 5X5, in_size = 1, out_size = 32
# 感受野5*5, 输入层的厚度是1, 输出的厚度是32
# **************convolution layer 1
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = biases_variable([32])
h_conv1 = tf.nn.relu(con2d(x_image, W_conv1) + b_conv1) #output size 28 *28 *32
h_pool1 = max_pool_2x2(h_conv1)             #output size 14 * 14 * 32


# convolution layer 2
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = biases_variable([64])
h_conv2 = tf.nn.relu(con2d(h_pool1, W_conv2) + b_conv2) # output size 14 *14 * 64
h_pool2 = max_pool_2x2(h_conv2)     # output size is  7 * 7 * 64


# function layer 1
W_fun1 = weight_variable([7*7*64, 1024])
b_fun1 = biases_variable([1024])
h_pool2flat = tf.reshape(h_pool2, [-1, 7*7*64])  # [n_sample, 7,7,64] ->> [n_samples, 7*7*64]
h_fun1 = tf.nn.relu(tf.matmul(h_pool2flat, W_fun1) + b_fun1)
h_fun1_drop = tf.nn.dropout(h_fun1, keep_prob)

# function layer 2
W_fun2 = weight_variable([1024,10])
b_fun2 = biases_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fun1_drop, W_fun2) + b_fun2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
# train step
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

def compute_accuracy(x_vs, y_vs):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:x_vs, keep_prob:1.0})
    right_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_vs,1))
    accuracy = tf.reduce_mean(tf.cast(right_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:x_vs, ys:y_vs, keep_prob:1.0})
    return result

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={xs: batch_xs, ys:batch_ys, keep_prob:0.5})  # feed in keep_prob
        if i % 50 == 0:
            print(compute_accuracy(mnist.test.images, mnist.test.labels))