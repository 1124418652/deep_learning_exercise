#usr/bin/env python
# -*- coding: utf-8 -*-
"""
# project: Exercise 3
# author:  xhj
# email:   1124418652@qq.com
# date:    2018/ 10/22
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from collections import namedtuple

__all__ = ["EXE3_1", "EXE3_2"]


class EXE3_1(object):
	"""
	使用逻辑斯蒂回归实现手写数字 0-9 的识别
	initial parameters:
	@ data_set: simple of model training
	@ labels: the label of every element in data_set
	@ classes: the number of labels
	"""

	count = 0

	def __len__(self):
		return EXE3_1.count

	def __init__(self, data_set, labels, classes = 10):
		self.data_set = data_set              # 训练集
		self.labels = labels                  # 标记
		self.num = len(labels)                # 样本数目
		self.feature_num = len(data_set[0])   # 样本特征数
		self.label_num = classes              # 标记类别数
		# 因为需要分成 label_num 类，所以需要 label_num 组 w 值
		self.w1_array = np.mat(np.random.randn(self.label_num, self.feature_num) / 1000)
		self.b1 = np.mat(np.zeros(self.label_num)).T
		EXE3_1.count += 1

	def show_img(self, row = 20, col = 20, img_row = 20, img_col = 20):
		img = []

		for i in range(row * col):
			img.append(self.data_set[i].reshape((img_row, img_col)))      # 读取指定数目的图片，存入数组

		# plt.imshow(img[0])
		img = np.concatenate(img, axis = 0)
		img = img.reshape(row, col, img_row, img_col).transpose((0, 2, 1, 3))\
			.reshape((row * img_row, col * img_col))          # 拼接成一幅大图像
		# print(img.shape)
		# cv2.imshow("img", img)
		# cv2.waitKey(0)
		plt.imshow(img)
		plt.show()

	def sigmod(self, data, w_array, b):
		return 1 / (1 + np.exp(-(w_array * data + b)))

	def training(self, iterate = 200, alpha = 0.1):
		"""
		# @iterate: 迭代次数
		# @alpha: 步长
		"""
		start = time.time()
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

		time_used = time.time() - start

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


class EXE3_2(EXE3_1):
	"""
	使用浅层（双层）逻辑斯蒂网络实现手写数字的识别
	"""
	flag = False    # 用于标记该模型是否已经经过训练，False：未训练

	def set_warray(self, w1_num, w2_num):
		self.w1_array = np.mat(np.random.randn(w1_num, self.feature_num) / 1000)
		self.w2_array = np.mat(np.random.randn(w2_num, w1_num) / 1000)    # 神经网络的 w 矩阵需要进行初始化
		self.b1 = np.mat(np.zeros(w1_num)).T
		self.b2 = np.mat(np.zeros(w2_num)).T

	def feed_prop(self, data_set):
		"""
		神经网络的前向传播，在执行该函数之前要先执行 set_warray()，设定神经网络的权值
		# @data_set: dims 为（feature，num）的数据向量
		"""
		a1 = self.sigmod(data_set, self.w1_array, self.b1)
		a2 = self.sigmod(a1, self.w2_array, self.b2)
		return a1, a2

	def set_labels(self, label_num, num, labels):
		label_array = np.tile(labels.T, label_num).reshape(10, num)
		label_array[0] = np.where(label_array[0] == 10, 1, 0)

		for i in range(1, label_num):
			label_array[i] = np.where(label_array[i] == i, 1, 0)

		return label_array

	def back_prop(self, iterate = 2000, alpha = 1):
		"""
		神经网络的反向传播，训练 w1_array, b1, w2_array, b2
		# @iterate：迭代次数
		# @alpha：步长
		"""
		start = time.time()
		a0 = np.mat(self.data_set).T
		labels = self.set_labels(self.label_num, self.num, self.labels)

		for i in range(iterate):
			a1, a2 = self.feed_prop(a0)
			dz2 = 1 / self.num * (a2 - labels)         # (dl / da2) * (da2 / dz)
			dw2 = dz2 * a1.T
			db2 = dz2.sum(axis = 1)
			dz1 = np.multiply(self.w2_array.T * dz2, np.multiply(a1, (1 - a1)))
			dw1 = dz1 * a0.T
			db1 = dz1.sum(axis = 1)
			self.w2_array -= alpha * dw2
			self.b2 -= alpha * db2
			self.w1_array -= alpha * dw1
			self.b1 -= alpha * db1

		EXE3_2.flag = True
		time_used = time.time() - start
		print("Time used of network training: %f s" %(time_used))

	def dict2json(self, file_path):
		if False == EXE3_2.flag:
			print("Have not trained the network!")
			return 1

		else:
			if 0 != len(file_path):
				fw = open(file_path, "w+")
				json.dump({"w1_array": self.w1_array.tolist(),
					"b1": self.b1.tolist(),
					"w2_array": self.w2_array.tolist(),
					"b2": self.b2.tolist(),
					"flag": EXE3_2.flag}, fw)
				fw.close()

			else:
				d = json.dumps({"w1_array": self.w1_array.tolist(),
					"b1": self.b1.tolist(),
					"w2_array": self.w2_array.tolist(),
					"b2": self.b2.tolist(),
					"flag": EXE3_2.flag})
				print(d)

	def network_from_json(self, file_path):
		try:
			fr = open(file_path, "r+")
			net_paras = json.load(fr)
		except:
			return 1

		return net_paras

	def testing(self, data, labels):
		if False == EXE3_2.flag:
			print("You have not training the model!\nRun function back_prop() first!")
			return 100

		else:
			num = len(data)                # 测试数据的数据量
			data = np.mat(data).T          # 将测试数据转置，以便带入模型计算
			a1, a2 = self.feed_prop(data)
			# a2 = a2.T
			res = np.zeros((len(a2), num)) # 记录测试结果
			labels = self.set_labels(10, num, labels)     # 将测试数据的标记改为（label_num, num）的格式

			error = 0.0
			for i in range(len(a2.T)):     # a2 为纵向排列的测试结果（纵向数据为分别属于该标签的概率）
				res[np.argmax(a2.T[i])][i] = 1
				if 1 != labels[np.argmax(a2.T[i])][i]:
					error += 1

			error /= num                   # 计算错误率
			print("The error of this network: ", error)

	def predict(self, data, *argv, **kwargv):

		net_paras = self.network_from_json(kwargv["file_path"])

		self.w1_array = np.mat(net_paras["w1_array"])
		self.w2_array = np.mat(net_paras["w2_array"])
		self.b1 = np.mat(net_paras["b1"])
		self.b2 = np.mat(net_paras["b2"])
		EXE3_2.flag = net_paras["flag"]

		data = np.mat(data).T
		a1, a2 = self.feed_prop(data)
		a2 = a2.T

		for i in range(len(a2)):
			print("The result of predict of picture %d is: %d" %(i, np.argmax(a2[i])))


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

def demo2():
	base_path = os.getcwd()
	path = "../ex3/ex3data1.mat"
	data = io.loadmat(path)
	data_set = data["X"]
	labels = data["y"]
	exe3_2 = EXE3_2(data_set, labels)
	# exe3_2.show_img()
	# exe3_2.set_warray(25, 10)
	# exe3_2.back_prop()

	jsonPath = os.path.join(base_path, "trained_network.txt")
	# exe3_2.dict2json(jsonPath)
	exe3_2.network_from_json(jsonPath)
	exe3_2.predict(data_set[000: 100], file_path = jsonPath)
	exe3_2.testing(data_set[5:4000], labels[5:4000])

def main():
	demo2()

if __name__ == '__main__':
	main()
