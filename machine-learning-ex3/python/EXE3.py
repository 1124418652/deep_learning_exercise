#usr/bin/env python
# -*- coding: utf-8 -*-
"""
# project: the algrithom of gray scale transfrom
# author:  xhj
# email:   1124418652@qq.com
# date:    2018/ 9/18
"""

import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from scipy import io

__all__ = ["EXE3_1", "EXE3_2"]

class EXE3_1(object):
	"""
	使用逻辑斯蒂回归实现手写数字 0-9 的识别
	initial parameters:
	@ data_set: simple of model training 
	@ labels: the label of every element in data_set
	@ classes: the number of labels
	"""

	def __init__(self, data_set, labels, classes = 10):
		self.data_set = data_set              # 训练集
		self.labels = labels                  # 标记
		self.num = len(labels)                # 样本数目
		self.feature_num = len(data_set[0])   # 样本特征数
		self.label_num = classes              # 标记类别数
		# 因为需要分成 label_num 类，所以需要 label_num 组 w 值
		self.w1_array = np.mat(np.random.randn(self.label_num, self.feature_num) / 1000)
		self.b1 = np.mat(np.zeros(self.label_num)).T

	def show_img(self, row = 20, col = 20, img_row = 20, img_col = 20):
		img = []

		for i in range(row * col):
			img.append(self.data_set[i].reshape((img_row, img_col)))      # 读取指定数目的图片，存入数组

		plt.imshow(img[0])
		img = np.concatenate(img, axis = 0)
		img = img.reshape(row, col, img_row, img_col).transpose((0, 2, 1, 3))\
			.reshape((row * img_row, col * img_col))          # 拼接成一幅大图像
		# print(img.shape)
		# cv2.imshow("img", img)
		# cv2.waitKey(0)
		# plt.imshow(img)
		plt.show()

	def sigmod(self, data, w_array, b):
		return 1 / (1 + np.exp(-(w_array * data + b)))

	def training(self, iterate = 200, alpha = 0.1):
		"""
		# @iterate: 迭代次数
		# @alpha: 步长
		"""
		data = np.mat(self.data_set).T
		labels = np.tile(self.labels.T, 10).reshape(10, self.num)  # 构造一个与样本同维度的标签集
		labels[0] = np.where(labels[0] == 10, 1, 0)

		for i in range(1, 10):
			labels[i] = np.where(labels[i] == i, 1, 0)
		
		for i in range(iterate):
			da = 1 / self.num * (self.sigmod(data, self.w1_array, self.b1) - labels)
			dw = da * data.T
			db = da.sum(axis = 1)
			self.w1_array -= alpha * dw
			self.b1 -= alpha * db

	def predict(self, data):
		data = np.mat(data).T
		return np.argmax(self.sigmod(data, self.w1_array, self.b1))

	def testing(self, data_set, labels):
		data = np.mat(data_set).T
		res = np.argmax(self.sigmod(data, self.w1_array, self.b1), axis = 0)
		res = np.where(res == 0, 10, res)
		# print(res)
		error = np.sum(np.where(res == labels.T, 0, 1)) / len(data_set)
		print("Error of this model: ", error)

def demo1():
	path = "../ex3/ex3data1.mat"
	data = io.loadmat(path)
	data_set = data["X"]
	labels = data['y']
	exe3_1 = EXE3_1(data_set, labels)
	# exe3_1.show_img(50, 50)
	exe3_1.training()
	# print(exe3_1.predict(data_set[4900]))
	exe3_1.testing(data_set[1000:5000], labels[1000:5000])

def main():
	demo1()

if __name__ == '__main__':
	main()