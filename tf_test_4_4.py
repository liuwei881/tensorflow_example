# coding=utf-8

import tensorflow as tf

# 指数衰减学习率, 开始学习率大, 往后学习率会逐步减小
# tf.train.exponential_decay
# 对应公式为
# decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
# decayed_learning_rate 为每一轮优化时使用的学习率, learning_rate 为事先设定的学习率
# decay_rate 为衰减系数, decay_steps 为衰减速度

global_step = tf.Variable(0)
# 通过exponential_decay函数生成学习率
# 0.1 初始学习率 100, 0.96, staircase=True，一起表示 每训练100轮学习率乘以0.96
learning_rate = tf.train.exponential_decay(
	0.1, global_step, 100, 0.96, staircase=True)

learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

# 带正则化的损失函数
lambdal = 0.001
w = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w)

# tf.reduce_mean(tf.square()) 均方误差损失函数
loss = tf.reduce_mean(tf.square(y_ - y)) + tf.contrib.layers.l2_regularizer(lambdal)(w)

# 样例
weights = tf.constant([[1.0, -2.0], [-3.0, 4.0]])
with tf.Session() as sess:
	# 绝对值相加结果乘以0.5
	print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))
	# 平方和除以2再乘以0.5
	print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))
