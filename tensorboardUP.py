#!coding=UTF-8
import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, n_layer, acti_fun=None):
    # rows = in_size, cols = out_size
    layer_name = 'layer%s' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope("weights"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='WStupid')
            # #################添加这样一样
            tf.summary.histogram(layer_name+'/Weights', Weights)
    # b一般不为0,所以需要加0.1
        with tf.name_scope("biaes"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='StupidB')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('y_hat'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if acti_fun is None:
            # 未提供额外函数的时候,默认为线性函数
            outputs = Wx_plus_b
        else:
            # 根据提供的激活函数处理结果
            outputs = acti_fun(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs

#   np.newaxis把vector变为300*1的mat, 注意更改生成的数据的float类型
x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
# 设置一个noise的值,参数分别未均值,方差,shape
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# None means无论输入多少sample都ok.因为输入只有一个特征，所以这里是1
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# 隐藏层有10个神经元,所以out_size=10, 浅层确实多种激活函数都可以使用
layer1 = add_layer(xs, 1, 10, n_layer=1, acti_fun=tf.nn.relu)
prediction = add_layer(layer1,10, 1, n_layer=2, acti_fun=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)

# 内部是learning rate,后面是train的目的,目的是最小化loss
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
# tf.train.SummaryWriter soon be deprecated, use following
# 把各种summary打包合并放到session.graph上面
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs/', sess.graph)

# init = tf.initialize_all_variables()  这个方式可以废弃了
init = tf.global_variables_initializer() # 定义初始化全局所有变量的节点
sess.run(init)  # 执行初始化

# 查看summary必须要有train data,并运行整个过程
for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0:
        # summary 只有在run之后才有效
        result = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
        writer.add_summary(result, i)
