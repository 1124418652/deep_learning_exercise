# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np 
import matplotlib.pyplot as plt

__all__ = ['Mult_layer_network']


class Mult_layer_network(object):

	def __init__(self):
		pass

	def load_data(self, filename, type = 'train'):
		if not os.path.exists(filename):
			raise(ValueError)
		try:
			with h5py.File(filename, 'r') as fr:
				if 'train' == type.lower():
					data_set = fr['train_set_x'][:]
					labels = fr['train_set_y'][:]
				elif 'test' == type.lower():
					data_set = fr['test_set_x'][:]
					labels = fr['test_set_y'][:]
			return [data_set.reshape(data_set.shape[0], -1).T, \
					labels]
		except:
			print("Can't open the file!")
			raise(ValueError)

	def init_network(self, num_layer = 3, **kwargs):
		"""
		parameters of this function:
		@ num_layer: int类型，表示神经网络的层数，不包括输入层
		             num_layer == len(nodes)
		parameters in kwargs:
		@ feature_num: int类型，每个样本的特征数（用于初始化第一层的节点）
		@ nodes: list类型，表示每一层网络中的节点数
		"""

		if not isinstance(num_layer, int): raise(ValueError)
		np.random.seed(12345)
		nodes = [20, 10, 1]
		if 'nodes' in kwargs: nodes = kwargs['nodes']
		assert('feature_num' in kwargs)
		assert(len(nodes) == num_layer)
		feature_num = kwargs['feature_num']

		w_array = {}         # 存储每一层的 w
		b_array = {}
		w_array[1] = np.random.randn(nodes[1], feature_num) \
				       * np.sqrt(1 / feature_num)
		b_array[1] = np.zeros((nodes[1], 1))
		for layer in range(2, num_layer + 1, 1):
			w_array[layer] = np.random.randn(nodes[layer], nodes[layer - 1])\
						   * np.sqrt(1 / nodes[layer - 1])
			b_array[layer] = np.zeros((nodes[layer], 1))
		return w_array, b_array

	def _activate_func(self, data_set, w, b, type = "sigmod"):
		"""
		parameters of this function:
		@ data_set: 输入的样本数据（每个样本占一列，每个特征占一行）
		@ w, b: 线性部分的参数
		@ type: 使用的激活函数的类型（'sigmod', 'relu'）
		"""

		if not isinstance(data_set, np.matrixlib.defmatrix.matrix):
			data_set = np.mat(data_set)
		if not isinstance(w, np.matrixlib.defmatrix.matrix):
			w = np.mat(w)
		if not isinstance(b, np.matrixlib.defmatrix.matrix):
			b = np.mat(b)

		assert(w.shape[1] == data_set.shape[0])
		assert(w.shape[0] == b.shape[0])

		z = w * data_set + b
		if "sigmod" == type.lower():
			a = 1 / (1 + np.exp(-z))
			da_dz = np.multiply(a, (1 - a))
		elif "relu" == type.lower():
			a = np.maximum(0, z)
			da_dz = np.where(z >= 0, 1, 0)
		return a, z, da_dz

	def _gradient_of_activate_func(self, a, type = 'sigmod'):
		"""
		parameters of this function:
		@ a: 激活函数求解得到的值
		@ type: 激活函数的类型
		"""

		if not isinstance(a, np.matrixlib.defmatrix.matrix):
			a = np.mat(a)
		if 'sigmod' == type.lower():
			da = np.multiply(a, (1 - a))
			assert(da.shape == a.shape)
		elif 'relu' == type.lower():
			da = np.where(a > 0, 1, 0)
		elif 'leaky_relu' == type.lower():
			da = np.where(a >= 0, 1, 0.01)
		elif 'tanh' == type.lower():
			da = 1 - np.power(a, 2)
			assert(a.shape == da.shape)
		return da

	def forword_propgation(self, data_set, w_array, b_array, activate_func):
		"""
		parameters in this function:
		@ data_set: 输入的样本数据（每个样本占一列， 每个特征占一行）
		@ w_array: dict类型，key 表示层，value 表示对应层的 w 矩阵
		@ b_array: dict类型，key 表示层，value 表示对应层的 b 数组
		@ activate_func: dict类型，每个元素表示一层网络中使用的激活函数,
						 每个元素的值取自（'sigmod', 'relu'）
		"""

		num_layer = len(b_array)
		a_array = {}
		z_array = {}
		da_dz = {}
		a_array[0] = data_set       # a_array[0] 为输入的样本数据
		for layer in range(1, num_layer + 1, 1): # w_array[0] 和 b_array[0] 为第一个隐藏层的参数
			a_array[layer], z_array[layer], da_dz[layer] = self._activate_func(a_array[layer - 1], \
						     w_array[layer], \
						     b_array[layer], \
						     activate_func[layer])
		return a_array, z_array, da_dz

	def back_propgation(self,
					    data_set,
					    labels,
					    step = 0.2,
					    rate_dacay = 0.005,
					    num_layer = 3,
					    nodes = {1:100, 2:20, 3:1},
					    activate_func = {1:'relu', 2:'relu', 3:'sigmod'},
					    iteration = 12500,
					    lamda = 0.01):
		"""
		parameters in this function:
		@ data_set: 输入的样本
		@ step: 更新网络的步长
		@ num_layer: 网络中总共包含的层数，不包括输入层
		@ nodes: dict类型，表示每一层网络中的节点数
		@ activate_func: dict类型，表示每一层网络所使用的激活函数
		@ iteratio: int类型，表示反向传播过程中迭代的次数
		"""

		feature_num, sample_num = data_set.shape
		w_array, b_array = demo.init_network(num_layer,\
											 feature_num = feature_num,\
											 nodes = nodes)
		cost = []
		for i in range(iteration):
			da = {}
			dz = {}
			dw = {}
			db = {}
			layer_iter = num_layer
			alpha = 1 / (1 + rate_dacay * i) * step       # 学习率（不断衰减）
			a_array, z_array, da_dz = self.forword_propgation(data_set, w_array,
											  b_array,
											  activate_func)
			cost.append(-1 / sample_num * \
						np.sum(np.multiply((1 - labels), \
							   np.log(1 - a_array[num_layer] + 10e-10)) \
							   + np.multiply(labels, \
							   np.log(a_array[num_layer] + 10e-10))))       # 使用最后一层的输出 a 计算 cost

			da[num_layer] = (a_array[num_layer] - labels + 10e-10 - 2 * 10e-10 * labels) \
					  / (np.multiply((1 - a_array[num_layer] + 10e-10), a_array[num_layer] + 10e-10)) # 计算最后一层网络的 da
			assert(da[num_layer].shape == a_array[num_layer].shape)
			while(1 <= layer_iter):
				# dz[layer_iter] = np.multiply(da[layer_iter],\
				# 					self._gradient_of_activate_func(a_array[layer_iter],\
				# 													activate_func[layer_iter]))
				dz[layer_iter] = np.multiply(da[layer_iter],\
									da_dz[layer_iter])
				dw[layer_iter] = dz[layer_iter] * a_array[layer_iter - 1].T
				assert(dw[layer_iter].shape == w_array[layer_iter].shape)
				db[layer_iter] = np.sum(dz[layer_iter], axis = 1)
				assert(db[layer_iter].shape == b_array[layer_iter].shape)

				w_array[layer_iter] -= 1 / sample_num * alpha * (dw[layer_iter] + lamda * w_array[layer_iter])
				b_array[layer_iter] -= 1 / sample_num * alpha * db[layer_iter]
				
				if layer_iter == 1:        # 计算完第一层的 dw 和 db 之后无需再计算 da[0]，因为 a[0] 未输入层
					break

				da[layer_iter - 1] = w_array[layer_iter].T * dz[layer_iter]
				assert[da[layer_iter - 1].shape == a_array[layer_iter - 1].shape]
				layer_iter -= 1
			if i % 100 == 0:
				print("iter: %d =========> cost: %s" %(i, cost[-1]))
		return w_array, b_array, cost 

	def testing(self, data_set, labels, w_array, b_array, activate_func):
		num = len(labels)
		labels = np.mat(labels)
		a_array, z_array, da_dz = self.forword_propgation(data_set, w_array, b_array, activate_func)
		ret = np.where(a_array[len(activate_func)] >= 0.5, 1, 0)
		print(ret)
		print(labels)
		print(a_array[len(activate_func)])
		print(np.abs(ret - labels).sum() / num)


	def save_to_file(self, filename, **kwargs):
		import json
		if 'w_array' in kwargs:
			w_array = kwargs['w_array']
			for key in w_array.keys():
				w_array[key] = w_array[key].tolist()
		if 'b_array' in kwargs:
			b_array = kwargs['b_array']
			for key in b_array.keys():
				b_array[key] = b_array[key].tolist()

		dict2json = dict(w_array = w_array, b_array = b_array, cost = cost, \
					activate_func = kwargs['activate_func'])
		with open(filename, 'w') as fw:
			json.dump(dict2json, fw)

	def load_params_from_file(self, filename):
		import json
		fr = open(filename, 'r')
		json2dict = json.load(fr)
		return json2dict

if __name__ == '__main__':
	train_filename = os.path.join(os.getcwd(), \
				"../datasets/train_catvnoncat.h5")	
	test_filename = os.path.join(os.getcwd(), \
				"../datasets/test_catvnoncat.h5")
	demo = Mult_layer_network()
	train_dataSet, train_labels = demo.load_data(train_filename, 'train')
	test_dataSet, test_labels = demo.load_data(test_filename, 'test')
	train_dataSet = np.mat(train_dataSet)
	test_dataSet = np.mat(test_dataSet)
	train_dataSet = (train_dataSet-train_dataSet.mean(axis = 1)) / 255
	test_dataSet = (test_dataSet-test_dataSet.mean(axis = 1)) / 255

	activate_func = {1:'sigmod', 2:'sigmod', 3:'sigmod', 4:'sigmod'}

	w_array, b_array, cost = demo.back_propgation(train_dataSet, train_labels, \
						 	 step = 0.2, \
						  	 num_layer = 4, \
						 	 nodes = {1: 100, 2: 50, 3: 10, 4: 1},\
							 activate_func = activate_func)
	params_filename = os.path.join(os.getcwd(),\
					"../datasets/params.txt")
	demo.save_to_file(params_filename, w_array = w_array, b_array = b_array, cost = cost,\
					activate_func = {1:'sigmod', 2:'relu', 3:'relu', 4:'sigmod'})
	demo.testing(test_dataSet, test_labels, w_array, b_array, activate_func)
	plt.plot(cost)
	plt.savefig("cost_curve.jpg")
