#usr/bin/env python
# -*- coding: utf-8 -*-
"""
project: support vector machine
author: xhj
"""

import os
import copy
import random
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

	def modify_data(self, x, y, del_last = True):
		y_new = []
		x_new = []
		
		for index, val in enumerate(y):
			if 1 == val:
				y_new.append(1)
			else:
				y_new.append(-1)
			x_new.append(x[index].tolist())

		if True == del_last:
			y_new.pop()
			x_new.pop()

		return x_new, y_new

	def _selectJ(self, i, m):
		j = i 
		while(j == i):
			j = int(random.uniform(0, m))

		return j

	def _clip_alpha(self, alpha, H, L):
		if alpha > H:
			alpha = H

		elif alpha < L:
			alpha = L

		return alpha

	def smoSimple(self, data_set, labels, C = 0.6, toler = 0.01, maxIter = 100):
		data_set = np.mat(data_set)
		labels = np.mat(labels).T
		rows, cols = data_set.shape
		alphas = np.mat(np.zeros((rows, 1)))
		b = 0;
		iter = 0

		# print(data_set)
		"""
		更新 alpha 的公式为： alpha2_new = alpha2_old + y2 * (E1 - E2) / (K11 + K22 - 2 * K12)
		"""
		while(iter < maxIter):           # 当找不到可以更新的合适的 alpha2 时，iter++
			alpha_pairs_changed = 0
			
			for i in range(rows):            # 通过遍历所有的样本点来选择 alpha1
				fxi = float(np.multiply(alphas, labels).T * \
					(data_set * data_set[i, :].T) + b)   # 将第i个点带入模型
				Ei = fxi - float(labels[i])

				if ((labels[i] * Ei < -toler) and (alphas[i] < C))\
					or ((labels[i] * Ei > toler) and (alphas[i] > 0)):
					j = self._selectJ(i, rows)
					fxj = float(np.multiply(alphas, labels).T * \
						(data_set * data_set[j, :].T) + b)
					Ej = fxj - float(labels[j])
					alpha_i_old = alphas[i].copy()
					alpha_j_old = alphas[j].copy()

					if(labels[i] != labels[j]):
						L = max(0, alphas[j] - alphas[i])
						H = min(C, C + alphas[j] - alphas[i])
					else:
						L = max(0, alphas[j] + alphas[i] - C)
						H = min(C, alphas[j] + alphas[i])

					if L == H:
						print("L == H")
						continue

					eta = 2.0 * data_set[i, :] * data_set[j, :].T - \
						data_set[i, :] * data_set[i, :].T- \
						data_set[j, :] * data_set[j, :].T                # 计算更新 alpha 的式子的分母

					if eta >= 0:
						print("eta >= 0")
						continue

					alphas[j] -= labels[j] * (Ei - Ej) / eta           # 计算 alpha 的新值
					alphas[j] = self._clip_alpha(alphas[j], H, L)      # 对 计算的得到的 alpha_j 增加限制条件

					if(abs(alphas[j] - alpha_j_old) < 0.00001):
						print("j not move enough!")
						continue

					# 通过等式 alpha_new_i * yi + alpha_new_j * yj = alpha_old_i * yi + alpha_old_j * yj
					alphas[i] += labels[i] * labels[j] * (alpha_j_old - alphas[j])

					b1 = b - Ei - labels[i] * (alphas[i] - alpha_i_old) * \
						data_set[i, :] * data_set[i, :].T - \
						labels[j] * (alphas[j] - alpha_j_old) * \
						data_set[j, :] * data_set[i, :].T
					b2 = b - Ej - labels[j] * (alphas[i] - alpha_j_old) * \
						data_set[j, :] * data_set[j, :].T - \
						labels[j] * (alphas[j] - alpha_j_old) * \
						data_set[j, :] * data_set[j, :].T

					if 0 < alphas[i] and C > alphas[i]:
						b = b1 

					elif 0 < alphas[j] and C > alphas[j]:
						b = b2

					else:
						b = (b1 + b2) / 2.0 

					alpha_pairs_changed += 1 

					print("iter: %d,\ti: %d,\tpairs changed: %d" %(iter, i, alpha_pairs_changed))
			if alpha_pairs_changed == 0:
				iter += 1 
			else:
				iter = 0 
			print("iter: %d" %(iter))

		# print(alphas, b)
		return alphas, b

	def predict(self, alphas, b, data_set, labels, x):
		data_set = np.mat(data_set)
		labels = np.mat(labels).T
		x = np.mat(x)
		y_p = np.multiply(alphas, labels).T * (data_set * x.T) + b

		print(y_p)
		if y_p >= 0:
			return 1
		else:
			return -1


def demo():
	basePath = os.getcwd()
	filePath = os.path.join(basePath, "../ex6/ex6data1.mat")
	data_set = io.loadmat(filePath)
	x, y = data_set["X"], data_set['y']
	# print(data_set)
	demo1 = SVM()
	demo1.show_data(x, y)

def demo2():
	import pandas as pd
	from sklearn import svm

	basePath = os.getcwd()
	filePath = os.path.join(basePath, "../ex6/ex6data1.mat")
	data_set = io.loadmat(filePath)
	x, y = data_set["X"], data_set['y']

	# print(x)
	s = pd.Series(x[0])
	print(s.dtypes)

def demo3():
	basePath = os.getcwd()
	filePath = os.path.join(basePath, "../ex6/ex6data1.mat")
	data_set = io.loadmat(filePath)
	x, y = data_set["X"], data_set["y"]
	app = SVM()
	# print(y)
	x_new, y_new = app.modify_data(x, y)
	# print(x_new)
	alpha, b = app.smoSimple(x_new, y_new)
	yp = app.predict(alpha, b, x_new, y_new, x_new[1])


def main():
	demo3()

if __name__ == '__main__':
	main()