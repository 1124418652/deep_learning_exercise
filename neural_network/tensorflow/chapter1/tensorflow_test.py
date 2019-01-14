# -*- coding: utf-8 -*-

from __future__ import division
import os
import struct
import numpy as np 
import tensorflow as tf 
from mnist_test import load_data 

test_data_file = '../mnist_database/t10k-images-idx3-ubyte'
test_label_file = '../mnist_database/t10k-labels-idx1-ubyte'
train_data_file = '../mnist_database/train-images-idx3-ubyte'
train_label_file = '../mnist_database/train-labels-idx1-ubyte'

train_data = load_data(train_data_file, 'data')
train_label = load_data(train_label_file, 'label')
test_data = load_data(test_data_file, 'data')
test_label = load_data(test_label_file, 'label')


class Softmax():
    
    def __init__(self):
        pass

    def initialize(self):
        x = tf.placeholder(tf.float32, (784, None), name = 'x')
        y_ = tf.placeholder(tf.float32, (10, None))
        W = tf.Variable(tf.zeros((10, 784)))
        b = tf.Variable(tf.zeros((10, 1)))
        y = tf.nn.softmax(tf.matmul(W, x) + b)

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis = 1, keep_dims = True))
