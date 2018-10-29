#usr/bin/env python
# -*- coding: utf-8 -*-
"""
# project: Exercise 5(Logistic regression)
# author:  xhj
# email:   1124418652@qq.com
# date:    2018/ 10/25
"""

import os
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import io 
from pylab import mpl 
from math import pow
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


class Logistic_Regression(object):

	def __init__(self):
		pass

	def show_data(self, x, y):
		# fig = plt.figure((6, 4)
		plt.plot(x, y, "x")
		plt.title("测试数据")
		plt.xlabel("x")
		plt.ylabel("y")

		x_new = np.linspace(x.min(), x.max(), 100)
		y_new = self.w * x_new + self.b
		plt.plot(x_new, y_new)
		plt.show()

	def linear_regression(self, x, y, least_squares = False, iterate = 2000, alpha = 0.001):
		self.w = 0.0
		self.b = 0.0
		num = len(x)
		lose = []

		if False == least_squares:
			for i in range(iterate):
				yhat = self.w * x + self.b 
				diff = yhat - y
				# print(1 / (2 * num) * pow(diff.sum(), 2))
				l = 1 / (2 * num) * pow(diff.sum(), 2)
				lose.append(l)
				dw = 1 / num * np.multiply(diff, x).sum()
				db = 1 / num * diff.sum()
				print(np.multiply(diff, x))

				self.w -= alpha * dw 
				self.b -= alpha * db 

		plt.plot(range(iterate), lose)
		plt.show()


def demo1():
	basePath = os.getcwd()
	dataPath = os.path.join(basePath, "../ex5/ex5data1.mat")
	matData = io.loadmat(dataPath)
	[x, y, xtest, ytest, xval, yval] = matData["X"], matData['y'], matData["Xtest"], \
		matData["ytest"], matData["Xval"], matData["yval"]

	print(x)
	lr = Logistic_Regression()
	# lr.show_data(x, y)
	lr.linear_regression(x, y)
	lr.show_data(x, y)
	

def main():
	demo1()

if __name__ == '__main__':
	main()