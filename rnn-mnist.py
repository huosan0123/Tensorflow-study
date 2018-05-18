import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# tested on python=3..2, tf=1.7.0
# Use RNN to classify mnist

tf.set_random_seed(1)
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

n_inputs = 28   # input data in every step
n_steps = 28    # time steps
n_hidden_units = 128
batch_size = 128
n_classes = 10

# input x is 3D, 1st D is batch size, 2nd D is n_rows, 3rd D is n_cols. 2 and 3 are pixels of a image
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

w1 = tf.Variable(tf.random_normal([n_inputs, n_hidden_units]))
w2 = tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
bias1 = tf.Variable(tf.constant(0.1, shape=[n_hidden_units,]))
bias2 = tf.Variable(tf.constant(0.1, shape=[n_classes,]))
weights = {"in": w1, "out": w2}
bias = {"in": bias1, "out": bias2}


def rnn(X, weight, bias):
    # RNN consists of 3 parts: input layer, cell, output layer.
    # transform input X to 2D
    # X_in = tf.matmul(X, weight["in"]) + bias["in"]
    # X_in = tf.reshape(X_in, [-1, n_steps, n_inputs])

    # define cells
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)

    # value of time_major depends on the shape of inputs into rnn cell
    # output is a Tensor of shape [batch_size, max_time, cell.output_size], or [max_time, batch_size, cell.output_size]
    output, final_state = tf.nn.dynamic_rnn(lstm_cell, X, time_major=False, dtype=tf.float32)
    print(output.shape)  # shape (batch, 28, 128)
    print(output[:, -1, :].shape) # shape (batch, 128)
    results = tf.matmul(output[:, -1, :], weights['out']) + bias['out']  # shape (batch, 10)

    return results


prediction = rnn(x, weights, bias)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1, 0):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape([-1, n_steps, n_inputs])
        sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
        if i % 10 == 0:
            a, b = sess.run([loss, acc], feed_dict={x: batch_x, y: batch_y})
            print(a, b)
        if i % 50 == 0:
            _x, _y = mnist.train.next_batch(batch_size)
            _x = _x.reshape([-1, n_steps, n_inputs])
            _loss, _acc = sess.run([loss, acc], feed_dict={x:_x, y:_y})
            print("TEST: loss={}, acc={}".format(_loss, _acc))
