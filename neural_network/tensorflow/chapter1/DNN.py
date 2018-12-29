# -*- coding: utf-8 -*-

import mnist_test
import numpy as np 
import matplotlib.pyplot as plt 

class DNN(object):

	def __init__(self, layers = 3, network_struct = []):
		if layers != len(network_struct):
			print("can't construct the network")
			return
		self.layers = layers
		self.network_struct = network_struct

	def set_hyper_parameters(self, **kwargs):

		if 'lamda' in kwargs.keys():
			self.lamda = kwargs['lamda']
		if 'learn_rate' in kwargs.keys():
			self.learn_rate = kwargs['learn_rate']

	def inititalize_parameters(self, nodes_of_layers, training_data_size):
		"""
		inittialize the parameters of every layers of this network,
		use the list 'nodes_of_layers', which stores the numbers of
		every layers

		@ params:
		nodes_of_layers: one dimention list type
		training_data_size: tupple or list with 2 values
			the size of training dataset

		@ return:
		True: BOOL
			return Ture if initialize the parameters without any errors
		False: BOOL
			return False if can not initialize the parameters correctly 
		"""

		assert(self.layers == len(nodes_of_layers))
		assert(2 == len(training_data_size))
		self.w_array = [np.array([0])]
		self.b_array = [np.array([0])]
		features, nums = training_data_size

		# initialize the parameters of layer one
		self.w_array.append(np.random.randn(nodes_of_layers[0], features)
							* np.sqrt(1 / nums))
		self.b_array.append(np.zeros((nodes_of_layers[0], 1)))

		for layer in range(1, self.layers):
			self.w_array.append(np.random.randn(nodes_of_layers[layer],
								nodes_of_layers[layer - 1])
								* np.sqrt(1 / nodes_of_layers[layer - 1]))
			self.b_array.append(np.zeros((nodes_of_layers[layer], 1)))
		return self.w_array, self.b_array

	def forward_activate(self, a_prev, w, b, func_type):
		"""
		The activation of every nodes in current layer

		@ params:
		a_prev: array_like
			the input dataset of this layer, which is the output of previous
			layer
		w, b: array_like 
			the parameters of current layer
		func_type: string
			the activation function which was used in current layer

		@ return:
		a: array_like
			the output of current layer's activation function, contains
			a_prev, w, b, z
		cache: tuple
			the cache of current layer
		"""

		z = np.dot(w, a_prev) + b
		if 'sigmod' == func_type.lower(): 
			a = 1 / (1 + np.exp(-z))
		elif 'relu' == func_type.lower():
			a = np.where(z >= 0, z, 0)
		elif 'leaky relu' == func_type.lower():
			a = np.where(z >= 0, z, 0.01 * z)
		elif 'tanh' == func_type.lower():
			a = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

		cache = (a_prev, w, b, z)
		return a, cache

	def backward_activate(self, da, a, cache, func_type):
		"""
		backforward propgation of this layer

		@ params:
		da: array_like
			the output of layer behind in backward propagation
		cache: the parameters stored in forward propgation
		func_type: the activation which was used in current layer
		"""

		a_prev, w, b = cache
		if 'sigmod' == func_type.lower():
			dz = np.multiply(np.multiply(a, (1 - a)), da)
		elif 'relu' == func_type.lower():
			if a > 0:
				dz = da 
			else:
				dz = np.multiply(0, da)
		elif 'leaky relu' == func_type.lower():
			if a >= 0:
				dz = da 
			else:
				dz = np.multiply(0.01, da)
		elif 'tanh' == func_type.lower():
			dz = np.multiply(1 - np.multiply(a, a), da)
		else:
			print("activate type error: %s" %(func_type))
			return

		dw = np.dot(dz, a_prev.T)
		db = np.sum(dz, axis = 1, keepdims = True)
		da_prev = np.dot(w.T, dz)

		return da_prev, dw, db

	def training(self, train_data, train_labels, iterates = 1, 
				 mini_batch = False, mini_size = 0):

		data_features, data_numbers = train_data.shape 
		label_classes, label_numbers = train_labels.shape 
		assert(data_features == w_array[1].shape[1])
		assert(label_classes == w_array[-1].shape[0])
		assert(data_numbers == label_numbers)

		if not mini_batch:
			loss = []
			for i in range(iterates):    # begin the iteration of upgrade parameters
				a_prev = train_data
				for l in range(1, self.layers + 1):
					a, cache = self.forward_activate(a_prev, self.w_array[l],
									self.b_array[l], self.network_struct[l - 1])
					a_prev = a 
				


if __name__ == '__main__':
	data = np.random.randn(3, 10)
	label = np.random.randint(0, 2, 20).reshape((2, 10))
	print(data, label)
	dnn = DNN(3, ['relu', 'relu', 'relu'])
	w_array, b_array = dnn.inititalize_parameters([2, 3, 2], (3, 10))
	dnn.training(data, label)