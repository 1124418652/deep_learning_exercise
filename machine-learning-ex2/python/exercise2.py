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
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

path = "../ex2/ex2data1.txt"

class EXE2(object):
	def __init__(self, *args, **argw):
		"""
		initial the logistic model
		@alpha: learning rate
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
		if 0 == len(self.data_set):
			print("Haven't load the data!")
			exit(1)

		training_num = int(len(self.data_set) * rate)
		self.training_data = np.array(self.data_set[: training_num + 1])
		self.test_data = np.array(self.data_set[training_num + 1: ])

	def sigmod(self, data, w_array, b):
		return 1 / (1 + np.exp(-(w_array * data + b)))

	def training(self, iterate = 100, type = "NO", lamda = 0.01):
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
			
			if re.match(r"^re*$", type.lower()):
				dw += lamda / m * np.multiply(w, w)

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
		print("Error of this model: %s" %format(error[0]))
		x = np.arange(self.data_set.min(), 0.1, self.data_set.max())

	def show_data(self):
		negtive = self.data_set[self.data_set[:, -1] == 0]       # select the negtive pointers
		positive = self.data_set[self.data_set[:, -1] == 1]      # select the positive pointers

		plt.plot(negtive[:, 0], negtive[:, 1], "rx", label = "negtive pointers")
		plt.plot(positive[:, 0], positive[:, 1], "go", label = "positive pointers")
		plt.legend()
		plt.grid(True)
		plt.show()

def main():
	exe2 = EXE2(0.001)
	exe2.load_data(path)
	exe2.show_data()
	exe2.data_segment()
	exe2.training(300)
	exe2.testing()

if __name__ == '__main__':
	main()
