#usr/bin/env python
# -*- coding: utf-8 -*-
"""
# project: Exercise 4(three layers Logistic network with regularization)
# author:  xhj
# email:   1124418652@qq.com
# date:    2018/ 10/24
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt 
from scipy import io 
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

__all__ = ["ANN"]


class ANN(object):

	def __init__(self, feat_num, w1_num, w2_num,**kwargv):
		"""
		# @data_set: 训练数据集
		# @labels: 训练数据集的标记
		# @num: 训练数据的样本量
		# @feat_num: 数据的特征个数
		"""

		# self.num = num
		self.feat_num = feat_num
		self.w1_array = np.mat(np.random.randn(w1_num, feat_num))
		self.b1 = np.mat(np.zeros(w1_num)).T
		self.w2_array = np.mat(np.random.randn(w2_num, w1_num))
		self.b2 = np.mat(np.zeros(w2_num)).T
		self.flag = False         # 用于标记该网络是否经过训练

	def show_img(self, data_set, rows, cols):
		"""
		# @rows: 图片中显示的行数
		# @cols: 图片中显示的列数
		"""

		img = []
		for i in range(rows * cols):
			img.append(np.array(data_set[i]).reshape(20, 20))   # matrix 类型只能是 2 维的

		# plt.imshow(img[1])
		img = np.concatenate(img, axis = 0).reshape(rows, cols, 20, 20)   # 将输入图像按行拼接
		img = np.transpose(img, (0, 2, 1, 3)).reshape(rows * 20, cols * 20)

		plt.imshow(img.T)
		plt.grid(True)
		plt.title(u"图片显示")
		plt.show()

	def __set_label(self, labels, label_num, num):
		mdlabels = np.mat(np.zeros((label_num, num)))           # mdlabels 为经过调整的样本标记
		labels = labels.T
		mdlabels[0] = np.where(labels == 10, 1, 0)              # 标记 10 表示数字 0

		for i in range(1, label_num):
			mdlabels[i] = np.where(labels == i, 1, 0)           # 分别调整各个标记的表示
		
		return mdlabels

	def sigmod(self, data_set, w_array, b):
		return 1 / (1 + np.exp(-(w_array * data_set + b)))

	def feed_prop(self, data_set, w1_array, b1, w2_array, b2):
		a1 = self.sigmod(data_set, w1_array, b1)
		a2 = self.sigmod(a1, w2_array, b2)
		return a1, a2

	def training(self, data_set, labels, iterate = 1500, alpha = 1, regular = False, lamda = 10):
		start = time.time()
		num = len(data_set)
		a0 = np.mat(data_set).T          
		label_num = len(set(labels.reshape(-1)))                # 获得总共的标记的类别数
		labels = self.__set_label(labels, label_num, num)

		for i in range(iterate):
			a1, a2 = self.feed_prop(a0, self.w1_array, self.b1, self.w2_array, self.b2)
			dz2 = 1 / num * (a2 - labels)
			dw2 = dz2 * a1.T
			db2 = dz2.sum(axis = 1)
			dz1 = np.multiply(self.w2_array.T * dz2, np.multiply(a1, 1 - a1))
			dw1 = dz1 * a0.T 
			db1 = dz1.sum(axis = 1)

			if False == regular:
				self.w2_array -= alpha * dw2
				self.b2 -= alpha * db2 
				self.w1_array -= alpha * dw1
				self.b1 -= alpha * db1

			else:
				self.w2_array -= alpha * (dw2 + lamda / num * self.w2_array)
				self.b2 -= alpha * db2
				self.w1_array -= alpha * (dw1 + lamda / num * self.w1_array)
				self.b1 -= alpha * db1

		self.flag = True
		print("Network training time: %.2f s" %(time.time() - start))

	def toJsonStyle(self):
		self.w1_array = self.w1_array.tolist()
		self.b1 = self.b1.tolist()
		self.w2_array = self.w2_array.tolist()
		self.b2 = self.b2.tolist()

	def load_model(self, annPath):
		fr = open(annPath, "r+")
		ann = json.load(fr)
		self.w1_array, b1 = ann["w1_array"], ann["b1"]
		self.w2_array, b2 = ann["w2_array"], ann["b2"]
		self.flag = ann["flag"]

	def testing(self, data_set, labels):
		if False == self.flag:
			print("Have not training the model!")
			return 1 

		num = len(data_set)
		data = np.mat(data_set).T 
		# labels = self.__set_label(labels, 10, num)
		labels = np.where(labels == 10, 0, labels).T.tolist()[0]

		if type(self.w1_array) != np.matrixlib.defmatrix.matrix:
			self.w1_array = np.mat(self.w1_array)
			self.b1 = np.mat(self.b1)
			self.w2_array = np.mat(self.w2_array)
			self.b2 = np.mat(self.b2)

		a1, a2 = self.feed_prop(data, self.w1_array, self.b1, self.w2_array, self.b2)
		res = np.argmax(a2, axis = 0).tolist()[0]        # 取出 a2 中概率最大的值的下标，即为类别
	
		error = 0.0
		for index, key in enumerate(res):
			if key != labels[index]:
				error += 1 	
		
		error /= num
		print("Accuracy of the network: %.3f" %(1 - error))

	def predict(self, data):
		if False == self.flag:
			print("Have not training the model!")
			return 1 

		data = np.mat(data).T 
		a1, a2 = self.feed_prop(data, self.w1_array, self.b1, self.w2_array, self.b2)
		print("The result of predict: ", np.argmax(a2, axis = 0).tolist()[0])

  
def demo():
	basePath = os.getcwd()
	dataPath = os.path.join(basePath, "..", "ex4", "ex4data1.mat")
	data = io.loadmat(dataPath)
	data_set = data["X"]
	labels = data["y"]

	exe4 = ANN(feat_num = 400, w1_num = 25, w2_num = 10)
	exe4.show_img(data_set, 100, 20)

	# exe4.training(data_set, labels, regular = True)

	annPath = "ann.txt"
	exe4.load_model(annPath)
	exe4.testing(data_set[000: 3000], labels[000: 3000])
	exe4.predict(data_set[[1, 100, 1000, 1400, 2000, 3000]])
	# fw = open(outPath, "w+")
	# exe4.toJsonStyle()
	# json.dump(exe4.__dict__, fw)
	# fw.close()

	

	# exe4.show_img(exe4.w1_array, 5, 5)


def main():
	demo()

if __name__ == '__main__':
	main()