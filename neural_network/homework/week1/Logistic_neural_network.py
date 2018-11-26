# -*- coding: utf-8 -*-
# Author: xhj

import os
import h5py
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image


class Logistic_Neural_Network(object):

	def __init__(self):
		pass

	def load_test_data(self, file_path):
		with h5py.File(file_path, 'r') as fr:
			data_set = np.array(fr['test_set_x'][:])
			labels = np.array(fr['test_set_y'][:])
			list_class = np.array(fr['list_classes'][:])
		return data_set, labels, list_class

	def load_train_data(self, file_path):
		with h5py.File(file_path, 'r') as fr:
			data_set = np.array(fr['train_set_x'][:])
			labels = np.array(fr['train_set_y'][:])
			list_class = np.array(fr['list_classes'][:])
		return data_set, labels, list_class

	def show_img(self, data_set, index = 0):
		fig = plt.figure(figsize = (6.4, 4.8))
		assert(index >= 0 and index <= data_set.shape[0])
		plt.imshow(data_set[index])
		plt.show()
	
	def _sigmod(self, w, x, b):
		if not isinstance(w, np.matrixlib.defmatrix.matrix):
			w = np.mat(w)
		if not isinstance(x, np.matrixlib.defmatrix.matrix):
			x = np.mat(x)
		assert(w.shape[1] == x.shape[0])
		return 1 / (1 + np.exp(-(w * x + b)))
	
	def init_w_b(self, layer_num = 2, *args):
		layer = {}
		for i in range(layer_num):
			layer[i+1] = dict(w_size = args[i], \
							  w = np.random.randn(args[i][0], args[i][1]) / 10e6,\
							  b = np.mat(np.zeros(args[i][0])).T)
		return layer

	def describe_data(self, data_set):
		if not isinstance(data_set, np.ndarray):
			data_set = np.array(data_set)
		image_shape, data_num = data_set.shape[0], data_set.shape[1:]
		print("number of this data: %d" %(data_num))
		print("image size: ", image_shape)

	def forward_propagation(self, data_set, w1, b1, w2, b2):
		a1 = self._sigmod(w1, data_set, b1)
		a2 = self._sigmod(w2, a1, b2)
		return a1, a2

	def back_propagation(self, data_set, labels, iterate = 500):
		params = demo.init_w_b(2, [15,data_set.shape[0]],[1,15])
		w1 = params[1]['w']
		b1 = params[1]['b']
		w2 = params[2]['w']
		b2 = params[2]['b']
		m = 209

		for i in range(iterate):
			a1, a2 = self.forward_propagation(data_set, w1, b1, w2, b2)
			dz2 = 1 / m * (a2 - labels)
			dw2 = dz2 * a1.T
			db2 = dz2.sum(axis = 1)
			dz1 = np.multiply(w2.T * dz2, np.multiply(a1, 1 - a1))
			dw1 = dz1 * data_set.T
			db1 = dz1.sum(axis = 1)

			w2 -= 0.01 * dw2
			b2 -= 0.01 * db2
			w1 -= 0.01 * dw1
			b1 -= 0.01 * db1

		return w1, b1, w2, b2

if __name__ == '__main__':
	demo = Logistic_Neural_Network()
	data_set, labels = demo.load_train_data("datasets/train_catvnoncat.h5")[0:2]
	#demo.show_img(data_set, 25)
	data_set = data_set.reshape(data_set.shape[0],-1).T
	demo.describe_data(data_set)
	# params = demo.init_w_b(2, [5,data_set.shape[0]],[1,5])
	# w1 = params[1]['w']
	# b1 = params[1]['b']
	# w2 = params[2]['w']
	# b2 = params[2]['b']
	# demo.forward_propagation(data_set, w1, b1, w2, b2)
	demo.back_propagation(data_set, labels, 100)