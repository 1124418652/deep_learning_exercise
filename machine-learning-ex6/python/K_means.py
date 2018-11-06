#usr/bin/env python
# -*- coding: utf-8 -*-
"""
# project: K-means
# author:  xhj
# email:   1124418652@qq.com
# date:    2018/ 11/06
"""

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from collections import namedtuple
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

__all__ = ['K_Means']

class K_Means(object):

	def __init__(self, K = 2):
		self.K = K

	def load_data(self, file_path):
		data_set = io.loadmat(file_path)
		x = data_set["X"]
		self.data_size, self.data_dim = x.shape
		return x

	def show_data(self, data):
		fig = plt.figure(figsize = (6, 6))
		ax = fig.add_subplot(111)
		ax.plot(data[:, 0], data[:, 1], "*")
		ax.set_title(u"原始图像")
		ax.set_xlabel(u"x0")
		ax.set_ylabel(u"x1")
		# plt.show()

	def init_k(self, k_num):
		random.seed(time.time())
		locs = set()
		i = 0
		while(i < k_num):
			loc = random.randint(0, self.data_size - 1)
			if loc not in locs:
				locs.add(loc)
				i += 1
		return locs

	def cluster(self, data, k_num = 3, iter = 100):
		c = np.zeros((self.data_size, iter))       # store the cluster center every simples belong to
		# lose = np.zeros(iter)
		result = {}
		result_array = []
		min_lose = np.power(data, 2).sum()

		init_kloc = self.init_k(k_num)             # initialize k for the first time
		k = []
		for loc in init_kloc:
			k.append(data[loc])

		i = 0
		while(i < iter):
			step = 0      # count the numbers of updating k
			while(True):
				for index, key in enumerate(data):
					distance = {}
					for class_label, center in enumerate(k):
						distance[class_label] = \
							np.multiply(center - key, center - key).sum()

					c[index, i] = sorted(distance.items(), key = \
						lambda x: x[1])[0][0]

				k_new = []
				diff = 0.0
				lose = 0.0

				for class_label in range(len(k)):
					tmp_array = data[np.where(c[:, i] == class_label)]
					tmp_pointer = tmp_array.sum(axis = 0) / len(tmp_array)
					diff += (tmp_pointer - k[class_label]).sum()
					# print(np.power(tmp_array - tmp_pointer, 2).sum())
					lose += np.power(tmp_array - tmp_pointer, 2).sum()
					k_new.append(tmp_pointer)

				if(abs(diff) < 1e-6):
					result_array.append({"i": i, "lose": lose, "k": k})
					print("step of %d is: %d" %(i, step))
					break          # 跳出更新k的循环
				else:
					step += 1
					k = k_new

			init_kloc = self.init_k(k_num)        # initalize the k array again
			k = []
			for loc in init_kloc:
				k.append(data[loc])

			i += 1

		sorted_res = sorted(result_array, key = lambda x: x['lose'])
		return sorted_res[0]['k'], sorted_res[0]['lose']

	def choose_k(self, data, max_k, iter = 100):
		pass

def demo():
	kmeans = K_Means()
	base_path = os.getcwd()
	file_path = os.path.join(base_path, "../../machine-learning-ex7/ex7/ex7data2.mat")
	x = kmeans.load_data(file_path)
	kmeans.show_data(x)
	# locs = kmeans.init_k(2)

	# x_center = []
	# for loc in locs:
	# 	x_center.append(x[loc])

	# kmeans.show_data(x)
	# plt.plot(x_center[0][0], x_center[0][1], "+")
	# plt.plot(x_center[1][0], x_center[1][1], "+")
	# plt.show()

	k, lose = kmeans.cluster(x)
	plt.plot(k[0][0], k[0][1], "o")
	plt.plot(k[1][0], k[1][1], "o")
	plt.plot(k[2][0], k[2][1], "o")
	plt.show()

	# print(k)

def main():
	demo()

if __name__ == "__main__":
	main()
