#usr/bin/env python
# -*- coding: utf-8 -*-
"""
# project: exercise2
# author:  xhj
# email:   1124418652@qq.com
# date:    2018/ 10/19
"""

import os
import re
import csv
import time
import random
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt


class EXE2_1(object):

	def __init__(self, *args, **kwarg):
		"""
		# initial the logistic model
		# @alpha : learning rate
		"""

		if 1 == len(args):
			self.alpha = args[0]
		else: self.alpha = 1

	def load_data(self, path):
		self.data_set = []

		with open(path, "r") as fr:
			freader = csv.reader(fr)

			for i in freader:
				self.data_set.append(i)

		self.data_set = np.array(self.data_set, dtype = float)
		self.w_array = np.mat(np.zeros(len(self.data_set[0]) - 1))
		self.b = 0.0

	def data_segment(self, rate = 0.7):               # segment the data_set into training and testing data set
		num = len(self.data_set)

		if 0 == len(self.data_set):
			print("Haven't load the data!")
			exit(1)

		training_num = int(len(self.data_set) * rate)
		training_index = set()

		while(training_num >= len(training_index)):
			index = random.randint(0, num - 1)
			if index not in training_index:
				training_index.add(index)

		self.training_data = np.array(self.data_set[[i for i in training_index]])
		self.test_data = np.array(self.data_set[[i for i in range(num) if i not in training_index]])

	def sigmod(self, data, w_array, b):
		return 1 / (1 + np.exp(-(w_array * data + b)))

	def training(self, iterate = 100, type = "NO", lamda = 1):
		"""
		Training the model:  
		@iterate: number of iterations in training
		@type: "REGULAR": regularization when calculate the dw
		"""
		try:
			data = np.mat(self.training_data[:, : -1]).T
			label = self.training_data[:, -1]

		except:
			print("Haven't segment the data set!")
			data = np.mat(self.data_set[:, : -1]).T
			label = self.data_set[:, -1]

		m = len(label)
		for i in range(iterate):
			dz = 1 / m * (self.sigmod(data, self.w_array, self.b) - label)
			# size of dz: (1, m)
			# print(dz.shape)
			dw = dz * data.T
			# size of dw: (1, len(data))
			db = np.sum(dz, axis = 1)

			if re.match(r"^re*", type.lower()):
				dw += lamda / m * self.w_array

			self.w_array -= self.alpha * dw
			self.b -= db

	def testing(self, *args):
		if 0 == len(self.test_data):
			if None == args:
				print("Don't have testing data set!")
				return 1
			else:
				data = args[0]

		data = np.mat(self.test_data[:, : -1]).T
		label = self.test_data[:, -1]
		test_res = self.sigmod(data, self.w_array, self.b)
		res = np.where(test_res > 0.5, 1, 0)
		error = np.abs(res - label).sum(axis = 1) / len(label)
		# print("w: %s\tb: %s" %(self.w_array, self.b))
		print("w: %s\tb: %s\n" %(self.w_array, self.b))
		print("Error of this model: %s" %format(error[0]))
		if 2 == len(data[:, 0]):
			x = np.linspace(self.data_set[:, 0].min(), self.data_set[:, 0].max(), 100)
			y = (-self.b[0, 0] - x * self.w_array[0, 0]) / self.w_array[0, 1]
			plt.plot(x, y)
		plt.show()

	def show_data(self):
		negtive = self.data_set[self.data_set[:, -1] == 0]       # select the negtive pointers
		positive = self.data_set[self.data_set[:, -1] == 1]      # select the positive pointers

		plt.plot(negtive[:, 0], negtive[:, 1], "rx", label = "negtive pointers")
		plt.plot(positive[:, 0], positive[:, 1], "go", label = "positive pointers")
		# plt.ylim(self.data_set[:, 1].min(), self.data_set[:, 1].max() * 1.1)
		plt.legend(loc = "upper right")
		plt.grid(True)
		# plt.show()


class EXE2_2(EXE2_1):

	def set_feature(self, index):
		self.index = index
		new_dataSet = np.ones((len(self.data_set), self.index * self.index))

		for i in range(self.index):
			for j in range(self.index):
				new_dataSet[:, i * self.index + j] = np.power(self.data_set[:, 0], i)\
					* np.power(self.data_set[:, 1], j)

		self.data_set = new_dataSet
		self.w_array = np.mat(np.zeros(len(self.data_set[0]) - 1))

def demo1():
	path = "../ex2/ex2data1.txt"
	exe2_1 = EXE2_1(0.001)
	exe2_1.load_data(path)
	exe2_1.show_data()
	exe2_1.data_segment(rate = 0.7)
	exe2_1.training(500, "regular", lamda = 1)
	exe2_1.testing()

def demo2():
	path = "../ex2/ex2data2.txt"
	exe2_2 = EXE2_2(0.001)
	exe2_2.load_data(path)
	exe2_2.show_data()
	exe2_2.set_feature(6)
	exe2_2.data_segment()
	exe2_2.training(600, type = "regular")
	exe2_2.testing()

def main():
	demo2()

if __name__ == '__main__':
	main()

