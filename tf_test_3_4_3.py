# coding=utf-8

import tensorflow as tf

# weights = tf.Variable(tf.random_normal([2, 3], stddev=2))
# w2 = tf.Variable(weights.initialized_value())
# w3 = tf.Variable(weights.initialized_value() * 2.0)
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

x = tf.constant([[0.7, 0.9]])
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(y))