# -*- coding: utf-8 -*-

import struct
import numpy as np 

__all__ = ["load_data", "one_hot"]

def load_data(file_name, flag = 'data'):

	if 'data' == flag.lower():
		imgs = []
		with open(file_name, 'rb') as fr:
			buf = fr.read()
			index = 0

			# extract the head of file
			magic, numImages, numRows, numCols = struct.unpack_from('>IIII', buf, index)
			index += struct.calcsize('>IIII')
			
			img_size = numRows * numCols
			for i in range(numImages):
				fmt = '>' + str(img_size) + 'B'
				imgs.append(struct.unpack_from(fmt, buf, index))
				index += struct.calcsize(fmt)

		imgs = np.array(imgs)
		return imgs.reshape((numImages, numRows, numCols))

	elif 'label' == flag.lower():
		labels = []
		with open(file_name, 'rb') as fr:
			buf = fr.read()
			index = 0

			magic, numLabels = struct.unpack_from('>II', buf, index)
			index += struct.calcsize('>II')
			for i in range(numLabels):
				labels.extend(struct.unpack_from('>B', buf, index))
				index += struct.calcsize('>B')
		return labels

def one_hot(labels):
	"""
	change the label array into one_hot type
	"""
	if not np.ndarray == type(labels):
		labels = np.array(labels)
	num = labels.shape[0]
	new_labels = np.zeros((10, num))
	for index, value in enumerate(labels):
		new_labels[value][index] = 1
	
	return new_labels

if __name__ == '__main__':
	test_data_file = "../mnist_database/t10k-images-idx3-ubyte"
	test_label_file = "../mnist_database/t10k-labels-idx1-ubyte"
	train_data_file = "../mnist_database/train-images-idx3-ubyte"
	train_label_file = "../mnist_database/train-labels-idx1-ubyte"
	test_data = load_data(test_data_file)
	test_labels = load_data(test_label_file, 'label')
	train_data = load_data(train_data_file, 'data')
	train_labels = load_data(train_label_file, 'label')

	train_labels = one_hot(train_labels)