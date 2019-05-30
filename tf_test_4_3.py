# coding=utf-8

import tensorflow as tf

batch_size = n
STEPS = 100

# 数据
x = tf.placeholder(tf.float32, shape=(batch_size, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(batch_size, 1), name='y-input')

# 定义损失函数及优化算法函数
loss = ...
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 开始训练
with tf.Session() as sess:
	# 参数初始化
	# 迭代更新参数
	for i in range(STEPS):
		# 随机打乱之后再选取会有更好的优化效果
		current_X, current_Y = ...
		sess.run(train_step, feed_dict={x: current_X, y_: current_Y})