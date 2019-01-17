# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np 
import matplotlib.pyplot as plt 
from image_processing import zero_pad


def conv_forward(dataset, w, b, stride = 1, pad = 0):
	"""
	parameters:
	@ dataset: 神经网络中上一层的输出数据，
			   维数为（samples, img_height, img_width, channels）
	@ w: 卷积核矩阵,
		 维数为（kernel_num, f, f, w_channels）
	"""

	(m, height_prev, width_prev, channels_prev) = dataset.shape
	(kernel_num, f, f, w_channels) = w.shape

	img_height = (height_prev - f + 2 * pad) // stride + 1
	img_width = (width_prev - f + 2 * pad) // stride + 1

	# initialize the output image dataset
	img_output = np.zeros((m, img_height, img_width, kernel_num))
	padded_image = zero_pad(dataset, pad)

	for num_img in range(m):
		for num_channel in range(kernel_num):
			for height in range(img_height):
				for width in range(img_width):
					height_start = height * stride
					height_end = height_start + f 
					width_start = width * stride
					width_end = width_start + f

					img_output[num_img, height, width, num_channel] = \
						np.multiply(w[num_channel, :, :, :],\
									padded_image[num_img, height_start: height_end,
												 width_start: width_end, :]).sum() + b

	return img_output

def pool_forward(dataset, mode = 'max', stride = 2, f = 2):
	"""
	parameters:
	@ dataset: the image output from convolution layer
	@ mode: 'mean' or 'max'
	@ f: the size of pool
	@ stride: stride size
	"""

	(m, height_prev, width_prev, channels_prev) = dataset.shape
	img_height = (height_prev - f) // stride + 1
	img_width = (width_prev - f) // stride + 1
	img_output = np.zeros((m, img_height, img_width, channels_prev))

	if 'max' == mode:
		for num_img in range(m):
			for height in range(img_height):
				for width in range(img_width):
					height_start = height * stride
					height_end = height_start + f 
					width_start = width * stride
					width_end = width_start + f

					img_output[num_img, height, width, :] =\
						np.max(dataset[num_img, height_start: height_end, \
								width_start: width_end, :], axis = (0, 1))

	elif 'mean' == mode:
		for num_img in range(m):
			for height in range(img_height):
				for width in range(img_width):
					height_start = height * stride
					height_end = height_start + f 
					width_start = width * stride
					width_end = width_start + f 

					img_output[num_img, height, width, :] =\
						np.mean(dataset[num_img, height_start: height_end,\
								width_start: width_end, :], axis = (0, 1))

	return img_output

def pool_back_propogation(dz):
	"""
	parameters:
	@ dz: 池化层的敏感度，与池化层输出对维度一致
	"""

def initialize_full_connect_parameters(data_size, w_size_list, b_size_list):
	assert(len(w_size_list) == len(b_size_list))
	pass



def pre_propagation_layer(dataset, w, b, active_function,\
						  dropout, lamda):
	pass


# dataset = np.random.randn(100, 20, 20, 3)
# w = np.random.randn(5, 3, 3, 3)

# print(conv_forward(dataset, w, 1).shape)

a = np.array([
		[[[2,3],[3,4],[4,5],[5,6]],
		 [[3,4],[4,5],[5,6],[6,7]]]
	])

print(a.shape)

print(pool_forward(a))