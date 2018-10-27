#usr/bin/env python
# -*- coding: utf-8 -*-
"""
project: support vector machine
author: xhj
"""

import os
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import io


class SVM(object):

	def __init__(self):
		pass

	def show_data(self, x, y):
		x_positive = []
		x_negtive = []

		for index, val in enumerate(y):
			if 1 == val:
				x_positive.append(x[index])
			else:
				x_negtive.append(x[index])

		x_negtive, x_positive = np.array(x_negtive), np.array(x_positive)
		fig = plt.figure(figsize = (6, 4))
		plt.plot(x_positive[:, 0], x_positive[:, 1], "g+", label = "positive pointers")
		plt.plot(x_negtive[:, 0], x_negtive[:, 1], "ro", label = "negtive pointers")
		plt.show()

def demo():
	basePath = os.getcwd()
	filePath = os.path.join(basePath, "../ex6/ex6data1.mat")
	data_set = io.loadmat(filePath)
	x, y = data_set["X"], data_set['y']
	# print(data_set)
	demo1 = SVM()
	demo1.show_data(x, y)

def main():
	demo()

if __name__ == '__main__':
	main()