# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.getcwd(), 'multi_layer_neural_network.py'))
from multi_layer_neural_network import *

__all__ = ['ReLU_NN']


class ReLU_NN(Mult_layer_network):

	def __init__(self):
		pass

	def _activate_func(self, data_set, w, b, type):
		if not isinstance(data_set, np.matrixlib.defmatrix.matrix):
			data_set = np.mat(data_set)
		if not isinstance(w, np.matrixlib.defmatrix.matrix):
			w = np.mat(w)
		if not isinstance(b, np.matrixlib.defmatrix.matrix):
			b = np.mat(b)

		assert(data_set.shape[0] == w.shape[1])
		assert(w.shape[0] == b.shape[0])

		z = w * data_set + b
		# z = (z - z.mean(axis = 1))
		# z = z / np.power(z, 2).mean(axis = 1)
		if 'relu' == type.lower():
			a = np.maximum(0, z)
			da_dz = np.where(z >= 0, 1, 0)
		elif 'sigmod' == type.lower():
			a = 1 / (1 + np.exp(-z))
			da_dz = np.multiply(1-a, a)
		return a, da_dz, z

	def forward_propgation(self, data_set, w_array, b_array):
		layer_num = len(b_array)
		a_array = {}
		z_array = {}
		da_dz = {}
		a_array[0] = data_set
		a_array[1], da_dz[1], z_array[1] = self._activate_func(a_array[0], w_array[1], b_array[1], 'relu')
		a_array[2], da_dz[2], z_array[2] = self._activate_func(a_array[1], w_array[2], b_array[2], 'sigmod')
		return a_array, da_dz, z_array

	def back_propgation(self, data_set, labels, \
						iteration = 1000, \
						learning_rate = 0.3\
						):
		w_array, b_array = demo.init_network(num_layer = 2, 
					  					 nodes = {1: 10, 2: 1}, 
					  					 feature_num = data_set.shape[0])
		# print(w_array[1].shape)
		m = len(labels)
		for i in range(iteration):
			rate = (1 / float(1 + 0.005 * i)) * learning_rate

			a_array, da_dz, z_array = self.forward_propgation(data_set, w_array, b_array)

			res = np.where(a_array[2] >= 0, 1, 0)
			res = np.mat(res)
			cost = -1 / m * np.sum(np.multiply((1-labels), np.log(1-a_array[2]+10e-10) +\
									np.multiply(labels, np.log(a_array[2] + 10e-10))))

			# print(a_array[2])
			da2 = (a_array[2] - labels) / (np.multiply(a_array[2], 1 - a_array[2]))
			dz2 = np.multiply(da2, da_dz[2])
			dw2 = dz2 * a_array[1].T
			db2 = dz2.sum(axis = 1)
			dz1 = np.multiply(w_array[2].T * dz2, da_dz[1])
			dw1 = dz1 * a_array[0].T
			db1 = dz1.sum(axis = 1)

			w_array[1] -= 1 / m * rate * dw1
			b_array[1] -= 1 / m * rate * db1
			w_array[2] -= 1 / m * rate * dw2
			b_array[2] -= 1 / m * rate * db2 

			# print(cost)
			if i % 100 == 0:
				print(cost, rate)
		return w_array, b_array
		

	def testing(self, data_set, labels, w_array, b_array):
		num = len(labels)
		a_array, da_dz, z_array = self.forward_propgation(data_set, w_array, b_array)
		res = np.where(a_array[2] >= 0.5, 1, 0)
		print(res)
		print(labels)
		print(np.abs(res - labels).sum() / num)

if __name__ == '__main__':
	demo = ReLU_NN()

	# load data set
	train_filename = os.path.join(os.getcwd(), \
				"../datasets/train_catvnoncat.h5")	
	test_filename = os.path.join(os.getcwd(), \
				"../datasets/test_catvnoncat.h5")
	train_dataSet, train_labels = demo.load_data(train_filename, 'train')
	test_dataSet, test_labels = demo.load_data(test_filename, 'test')
	train_dataSet = np.mat(train_dataSet)
	test_dataSet = np.mat(test_dataSet)
	train_dataSet = (train_dataSet - train_dataSet.mean(axis = 1)) / 255 

	test_dataSet = (test_dataSet - test_dataSet.mean(axis = 1)) / 255


	data_set1 = np.random.randn(4097, 209)
	labels = np.random.randn(209)
	labels = np.where(labels > 0, 1, 0)

	w_array, b_array = demo.back_propgation(train_dataSet, train_labels)
	demo.testing(test_dataSet, test_labels, w_array, b_array)
	# print(train_dataSet)
	# demo.back_propgation(test_dataSet, test_labels)
	# print(labels)