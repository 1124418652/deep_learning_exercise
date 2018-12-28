# -*- coding: utf-8 -*-
import numpy as np 
import tensorflow as tf 
import time

"""
tensorflow 程序分三步：
创建变量
初始化
运行
"""


y_hat = tf.constant(36, name = "y_hat")
y = tf.constant(39, name = "y")

loss = tf.Variable((y - y_hat)**2, name = "loss")

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    print(session.run(loss))

def linear_function():
    np.random.seed(1)
    X = np.random.randn(3,1)
    W = np.random.randn(4,3)
    b = np.random.randn(4,1)

    Y = tf.add(tf.matmul(W, X), b)

    sess = tf.Session()
    result = sess.run(Y)
    sess.close()
    return result

print(linear_function())
