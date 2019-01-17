# -*- coding: utf-8 -*-
from __future__ import division
import os
import pickle
import numpy as np 


dir_path = "/home/xhj/work/deep_learning_exercise/neural_network/dataset/cifar-10-batches-py"

def load_data(file_name):
	file_path = os.path.join(dir_path, file_name)
	if not os.path.exists(file_path):
		return
	with open(file_path, 'rb') as fr:
		data_dict = pickle.load(fr, encoding = 'bytes')
	return data_dict

def construct_trainset():
	train_set = []
	train_label = []
	for i in range(1, 6):
		file_name = 'data_batch_' + str(i)
		data_dict = load_data(file_name)
		train_set.append(data_dict[b'data'])
		train_label.append(data_dict[b'labels'])
	train_set = np.concatenate(train_set, axis = 0)
	train_labels = np.concatenate(train_label)
	return train_set, train_labels

def construct_testset():
	data_dict = load_data('test_batch')
	test_set = data_dict[b'data']
	test_labels = data_dict[b'labels']
	return test_set, test_labels
