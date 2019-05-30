# coding=utf-8

import tensorflow as tf

# 在foo命名空间内创建名字为v的变量
# with tf.variable_scope('foo'):
#     v = tf.get_variable(
#         'v', [1], initializer=tf.constant_initializer(1.0)
#     )

# # reuse=True, 只能获取已经创建过的变量.
# with tf.variable_scope('foo', reuse=True):
#     v1 = tf.get_variable('v', [1])
#     print(v == v1)

# with tf.variable_scope('bar'):
#     v = tf.get_variable('v', [1])


# with tf.variable_scope('root'):
#     print(tf.get_variable_scope().reuse)
#     with tf.variable_scope('foo', reuse=True):
#         print(tf.get_variable_scope().reuse)
#         with tf.variable_scope('bar'):
#             print(tf.get_variable_scope().reuse)

v1 = tf.get_variable('v', [1])
print(v1.name)

with tf.variable_scope('foo'):
    v2 = tf.get_variable('v', [1])
    print(v2.name)

with tf.variable_scope('foo'):
    with tf.variable_scope('bar'):
        v3 = tf.get_variable('v', [1])
        print(v3.name)
    v4 = tf.get_variable('v1', [1])
    print(v4.name)