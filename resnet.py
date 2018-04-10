import tensorflow as tf
import tensorflow.contrib.layers as tcl

"""
This a implementation of ResNet at http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006.
"""
def bn_relu(x, is_training=True):
	# Use bn before relu. Note in TF: batch_norm should with tf.device('cpu:0')
    with tf.device('cpu:0'):
        x = tcl.batch_norm(x, is_training=is_training)
    y = tf.nn.relu(x)
    return y

def single_resblock(x, filters_list, k_sizes, func=bn_relu, is_training=True):
	# This is for workflow with no conv in left branch. Since strides in every part are 1, so here don't explicit state.
	# filters_list contains three num_filter.
    r1 = tcl.conv2d(x, filters_list[0], [k_sizes[0], k_sizes[0]], [1, 1], activation_fn=func)
    r2 = tcl.conv2d(r1, filters_list[1], [k_sizes[1], k_sizes[1]], [1, 1], activation_fn=func)
    r3 = tcl.conv2d(r2, filters_list[2], [k_sizes[2], k_sizes[2]], [1, 1], activation_fn=None)
    with tf.device('cpu:0'):
        r3_bn = tcl.batch_norm(r3, is_training=is_training)

    # Merge residual and x
    res = tf.add(x, r3_bn)
    return tf.nn.relu(res)


def two_resblock(x, filters_list, k_sizes, strides, func=bn_relu, is_training=True):
    # filters_list contains 4 num_filter; strides contains four num_stride; k_sizes have 4 kernel size
    # left branch of conve
    l1 = tcl.conv2d(x, filters_list[0], [k_sizes[0], k_sizes[0]], [strides[0], strides[0]], activation_fn=None)
	# batch_norm must be processed in cpu
    with tf.device('cpu:0'):
        l1_bn = tcl.batch_norm(l1, is_training=is_training)
    print(l1_bn.shape)

    # right branch
    r1 = tcl.conv2d(x, filters_list[1],  [k_sizes[1], k_sizes[1]], [strides[1], strides[1]], activation_fn=func)
    r2 = tcl.conv2d(r1, filters_list[2],  [k_sizes[2], k_sizes[2]], [strides[2], strides[2]], activation_fn=func)
    r3 = tcl.conv2d(r2, filters_list[3],  [k_sizes[3], k_sizes[3]], [strides[3], strides[3]], activation_fn=None)
    with tf.device('cpu:0'):
        r3_bn = tcl.batch_norm(r3, is_training=is_training)
    print(r3_bn.shape)
    # Add two branch
    res2 = tf.add(l1_bn, r3_bn)
    return tf.nn.relu(res2)

class ResNet(object):
    """
    build computation graph of ResNet
    """
    def __init__(self):
        
        self.images = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
        self.input_labels = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name = 'dropout_keep_prob')
        self.is_training = tf.placeholder(dtype=tf.bool, name = 'is_training')
        self.lr = tf.placeholder(dtype=tf.float32)

        with tf.variable_scope("resnet") as scope:
            self.train_logits = self.construct_net(self.is_training)

        # compute acc
        self.result_label = tf.argmax(self.input_labels,1)
        self.result_pred = tf.argmax(self.train_logits, 1)

        self.correct = tf.equal(tf.argmax(self.input_labels, 1), tf.argmax(self.train_logits, 1))
        self.acc = tf.reduce_mean(tf.cast(self.correct, tf.float32))

        self.parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

        # training part
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.train_logits, labels=self.input_labels))
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


    def construct_net(self, is_training):
        with tf.variable_scope("CNN-ResNet"):
            conv1 = tcl.conv2d(self.images, 64, [7, 7], [2, 2], padding='SAME', activation_fn=bn_relu)
            pool1 = tf.nn.max_pool(conv1, [1,3,3,1], [1,2,2,1], padding='SAME')
            res2a_relu = two_resblock(pool1,[256, 64, 64, 256], [1, 1, 3, 1], [1, 1, 1, 1],func=bn_relu, is_training=is_training)
            res2b_relu = single_resblock(res2a_relu, [64, 64, 256],[1, 3, 1], is_training=is_training)
            res2c_relu = single_resblock(res2b_relu, [64, 64, 256],[1, 3, 1], is_training=is_training)
            res3a_relu = two_resblock(res2c_relu, [512, 128,128, 512], [1, 1, 3, 1], [2, 2, 1, 1], is_training=is_training)
            res3b_relu = single_resblock(res3a_relu, [128, 128, 512], [1, 3, 1], is_training=is_training)
            res3c_relu = single_resblock(res3b_relu, [128, 128, 512], [1, 3, 1], is_training=is_training)
            res3d_relu = single_resblock(res3c_relu, [128, 128, 512], [1, 3, 1], is_training=is_training)
            # res 4 block
            res4a_relu = two_resblock(res3d_relu, [1024, 256, 256, 1024], [1, 1, 3, 1], [2, 2, 1, 1], is_training=is_training)
            res4b_relu = single_resblock(res4a_relu, [256, 256, 1024], [1, 3, 1], is_training=is_training)
            res4c_relu = single_resblock(res4b_relu, [256, 256, 1024], [1, 3, 1], is_training=is_training)
            res4d_relu = single_resblock(res4c_relu, [256, 256, 1024], [1, 3, 1], is_training=is_training)
            res4e_relu = single_resblock(res4d_relu, [256, 256, 1024], [1, 3, 1], is_training=is_training)
            res4f_relu = single_resblock(res4e_relu, [256, 256, 1024], [1, 3, 1], is_training=is_training)
            # res 5 block
            res5a_relu = two_resblock(res4f_relu, [2048, 512, 512, 2048], [1, 1, 3, 1], [2, 2, 1, 1], is_training=is_training)
            res5b_relu = single_resblock(res5a_relu, [512, 512, 2048], [1, 3, 1], is_training=is_training)
            res5c_relu = single_resblock(res5b_relu, [512, 512, 2048], [1, 3, 1], is_training=is_training)

            # pool layers
            res_pool = tf.nn.avg_pool(res5c_relu, [1, 7, 7, 1], [1,1,1,1], padding='VALID')

        # Concate two tensors(tensor from image, and tensor from video_info). May need dropout considering the acc
        image_fea = tcl.flatten(res_pool)
        fc = tf.contrib.layers.fully_connected(image_fea, 10, activation_fn = tf.nn.relu)
        return fc

#####################
Here is your train part , you should build the graph first, feed in data, train data and evaluate acc/loss
#####################