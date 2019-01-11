#!/usr/bin/env python
import os
import numpy as np 
import tensorflow as tf 
from mnist_test import *

# load the data from binary file
train_data = load_data(train_data_file)
train_labels = load_data(train_label_file, 'label')
test_data = load_data(test_data_file)
test_labels = load_data(test_label_file, 'label')

train_data = np.reshape(train_data, 
	(train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))
train_labels = one_hot(train_labels)
test_data = np.reshape(test_data,
		(test_data.shape[0], test_data.shape[1], test_data.shape[2], 1))
test_labels = one_hot(test_labels)

# the nodes of convolution neural network
x_image = tf.placeholder(tf.float32, [None, 28, 28, 1])
y_ = tf.placeholder(tf.float32, [10, None])

# the first convolution layer
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W): # use padding, the dimension of output doesn't change
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1,2,2,1],
						  strides = [1,2,2,1], padding = 'SAME')

# calculate the first convolution layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# calculate the second convolution layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# fully connect layer
W_fc1 = weight_variable([1024, 7 * 7 * 64])
b_fc1 = bias_variable([1024, 1])
h_pool2_flat = tf.transpose(tf.reshape(h_pool2, [-1, 7 * 7 * 64]))
h_fc1 = tf.nn.relu(tf.matmul(W_fc1, h_pool2_flat) + b_fc1)

# use dropout regularization
keep_prob = tf.placeholder(tf.float32) # dropout value, 0.5 when training and 1 when testing
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([10, 1024])
b_fc2 = bias_variable([10, 1])
y_conv = tf.matmul(W_fc2, h_fc1_drop) + b_fc2

# calculate cross-entropy
y = tf.nn.softmax(y_conv)
cross_entropy = tf.reduce_mean(tf.reduce_sum(y_ * tf.log(y), axis = 0))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(10):
	sess.run(train_step, feed_dict = {
			x_image: train_data[:100],
			y_: train_labels[:, :100],
			keep_prob: 0.5
		})
	correction = tf.equal(tf.argmax(y_, 0), tf.argmax(y, 0))
	correction = tf.cast(correction, tf.float32)
	# print(correction)
	c = sess.run(correction, feed_dict = {
			x_image: test_data[:100],
			y_: test_labels[:, :100],
			keep_prob: 1
		})
	# print(c)
	# print(test_labels[:, :100])
	print(sess.run(y_conv, feed_dict = {
		x_image: test_data[:100],
		y_: test_labels[:, :100],
		keep_prob: 1
		}))
	y_predict = tf.argmax(y, axis = 0)
	test_label = tf.argmax(test_labels[:, :100], axis = 0)
	# print(sess.run(y_predict, feed_dict = {
		# x_image: test_data[:100],
		# keep_prob: 1
		# }))
	# print(test_label.eval())

# cross_entropy = tf.reduce_mean(
# 		tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv)
# 	)
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# correct_prediction = tf.equal(tf.argmax(y_, 0), tf.argmax(y_conv, 0))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())

# for i in range(1000):
# 	train_accuracy = accuracy.eval(feed_dict = {
# 			x_image: test_data[:100],
# 			y_: test_labels[:, :100],
# 			keep_prob: 1.0
# 		})
# 	print("step: %d, training accuracy: %g" % (i, train_accuracy))
# 	train_step.run(feed_dict = {
# 			x_image: test_data[:100],
# 			y_: test_labels[:, :100],
# 			keep_prob: 0.5
# 		})