# -*- coding: utf-8 -*-
# Author: xhj

import os
import h5py
import numpy as np 


class Logistic_Neural_Network(object):

	def __init__(self):
		pass

	def load_data(self, file_path):
		with h5py.File(file_path, 'r') as fr:
			data_set = np.array(fr['test_set_x'][:])
			labels = np.array(fr['test_set_y'][:])
			list_class = np.array(fr['list_classes'][:])
		print(data_set, labels, list_class)


if __name__ == '__main__':
	demo = Logistic_Neural_Network()
	demo.load_data("datasets/test_catvnoncat.h5")
