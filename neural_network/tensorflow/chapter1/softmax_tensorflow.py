#!/usr/bin/env python
import numpy as np 
import tensorflow as tf 
from mnist_test import *

# load the training_data
training_data = load_data(train_data_file, 'data')
training_labels = load_data(train_label_file, 'label')
training_data = training_data.reshape(training_data.shape[0], 
	training_data.shape[1] * training_data.shape[2])
training_labels = one_hot(training_labels).T
# load the testing_data
testing_data = load_data(test_data_file, 'data')
testing_labels = load_data(test_label_file, 'label')
testing_data = testing_data.reshape(testing_data.shape[0],
	testing_data.shape[1] * testing_data.shape[2])
testing_labels = one_hot(testing_labels).T

# set the node
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

z = tf.matmul(x, W) + b
z = z - tf.reduce_mean(z, axis = 0)

# y = tf.exp(z)   # calculate y with the model
# y_sum = tf.reduce_sum(y, axis = 1, keep_dims = True)
# y = tf.divide(y, y_sum)
y = tf.nn.softmax(z)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = \
	tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y + 10e-8), axis = 1), axis = 0)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

step = 100
for i in range(500):
	# cross_entropy_value = sess.run(cross_entropy, feed_dict = {
	# 	x: training_data[:1000],
	# 	y_: training_labels[:1000]
	# 	})
	# print(cross_entropy_value)

	# train_step_res = sess.run(train_step, feed_dict = {
	# 	x: training_data[i*step: (i+1)*step, :],
	# 	y_: training_labels[i*step: (i+1)*step, :]
	# 	})

	train_step_res = sess.run(train_step, feed_dict = {
		x: training_data[:],
		y_: training_labels[:]})
	

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict = {
		x: testing_data,
		y_: testing_labels}))
