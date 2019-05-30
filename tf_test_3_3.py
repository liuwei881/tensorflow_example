# coding=utf-8

import tensorflow as tf

a = tf.constant([1, 2], name='a', dtype='int32')
b = tf.constant([3, 4], name='b', dtype='int32')

result = a + b

# with tf.Session() as sess:
#     print(sess.run(result))
# sess = tf.Session()
# with sess.as_default():
#     print(result.eval())
# sess = tf.InteractiveSession()
# print(result.eval())
# sess.close()

config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=True)
sess1 = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=config)