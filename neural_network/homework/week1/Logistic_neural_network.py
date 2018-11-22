# -*- coding: utf-8 -*-
# Author: xhj

import os
import h5py
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image


class Logistic_Neural_Network(object):

	def __init__(self):
		pass

	def load_test_data(self, file_path):
		with h5py.File(file_path, 'r') as fr:
			data_set = np.array(fr['test_set_x'][:])
			labels = np.array(fr['test_set_y'][:])
			list_class = np.array(fr['list_classes'][:])
		return data_set, labels, list_class

	def load_train_data(self, file_path):
		with h5py.File(file_path, 'r') as fr:
			data_set = np.array(fr['train_set_x'][:])
			labels = np.array(fr['train_set_y'][:])
			list_class = np.array(fr['list_classes'][:])
		return data_set, labels, list_class

	def show_img(self, data_set):
		fig = plt.figure(figsize = (6.4, 4.8))
		Image._show(Image.fromarray(data_set[1]))


if __name__ == '__main__':
	demo = Logistic_Neural_Network()
	data_set = demo.load_train_data("datasets/train_catvnoncat.h5")[0]
	demo.show_img(data_set)
	print(data_set.shape)
