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
	@ b: 卷积核的偏移系数，kernel_num 维的向量

	return:
	@ img_output: 卷积层前向传播的输出图片，
				  维数为（samples, img_height_new, img_width_new, kernel_num）
	@ cache: 用于该卷积层反向传播时求解的缓存参数
		dataset: 该层前向传播的输入图像
		w: 该层的卷积核矩阵
		stride: 卷积步长
		pad: pading值
	"""

	(m, height_prev, width_prev, channels_prev) = dataset.shape
	(kernel_num, f, f, w_channels) = w.shape

	img_height = (height_prev - f + 2 * pad) // stride + 1
	img_width = (width_prev - f + 2 * pad) // stride + 1
	dataset = np.pad(dataset, ((0, 0), 
							   (pad, pad),
							   (pad, pad), 
							   (0, 0)), 'constant', constant_values = 0)
	cache = (dataset, w, stride, pad)

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
												 width_start: width_end, :]).sum()\
												 + b[num_channel]

	return img_output, cache

def pool_forward(dataset, mode = 'max', stride = 2, f = 2):
	"""
	parameters:
	@ dataset: the image output from convolution layer
	@ mode: 'mean' or 'max'
	@ f: the size of pool
	@ stride: stride size

	return:
	@ img_output: 池化之后的输出数据
	@ cache: 用于该池化层反向传播时求解的缓存参数
		dataset: 该层前向传播时的输入图像
		mode: 池化的模式 "max" | "mean"
		stride: 池化时的移动步长
		f: 池的大小
	"""

	(m, height_prev, width_prev, channels_prev) = dataset.shape
	img_height = (height_prev - f) // stride + 1
	img_width = (width_prev - f) // stride + 1
	img_output = np.zeros((m, img_height, img_width, channels_prev))
	cache = (dataset, mode, stride, f)

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

	return img_output, cache

def pool_back_propogation(dz, cache):
	"""
	parameters:
	@ dz: 池化层的敏感度(d_lost / d_z)，与池化层输出对维度一致
	@ cache: 从池化层的前向传播中传递下来的参数
		pool_input: 池化层前向传播对应的输入数据
		mode: 池化层的类型，'mean' | 'max'
		stride: 池化层的步幅
		f: 池的大小
	"""

	pool_input, mode, stride, f = cache
	m, img_height, img_width, channels = dz.shape
	m, height_prev, width_prev, channels = pool_input.shape	
	dA_prev = np.zeros((m, height_prev, width_prev, channels))

	def get_mask(x, mode):
		m_height, m_width, m_channel = x.shape
		if 'mean' == mode:
			mask = np.ones((m_height, m_width, m_channel)) / (m_height * m_width)
		elif 'max' == mode:
			mask = x == np.max(x, axis = (0, 1))
		return mask

	for num_img in range(m):
		for height in range(img_height):
			for width in range(img_width):
				height_start = height * stride
				height_end = height_start + f 
				width_start = width * stride
				width_end = width_start + f

				x_slice = pool_input[num_img, height_start: height_end,\
									 width_start: width_end, :]
				mask = get_mask(x_slice, mode)
				dA_prev[num_img, height_start: height_end,\
						width_start: width_end, :] += np.multiply(mask, dz[num_img, height, width, :])
	return dA_prev

def conv_back_propogation(dz, cache):
	"""
	parameters:
	@ dz: 反向传播时传入卷积层的敏感度（d_cost / dz）
	@ cache: 对应的卷积层的缓存数据
		dataset: 该层前向传播的输入图像
		w: 该层的卷积核矩阵
		stride: 卷积步长
		pad: pading值

	return:
	@ dA_prev: 传递到
	@ dw:
	@ db:
	"""

	conv_input, w_before, stride, pad = cache
	m, img_height, img_width, channels = dz.shape
	_, height_prev, width_prev, channels_prev = conv_input.shape
	dA_prev_temp = np.zeros()


def initialize_full_connect_parameters(data_size, w_size_list, b_size_list):
	assert(len(w_size_list) == len(b_size_list))
	pass



def pre_propagation_layer(dataset, w, b, active_function,\
						  dropout, lamda):
	pass


dataset = np.random.randn(200, 20, 20, 3)
w = np.random.randn(10,5,5,3)
b = np.random.randn(10)
img_output, cache = conv_forward(dataset, w, b, 1, 2)


print(img_output.shape)
# img_output, cache = pool_forward(dataset)
# print(img_output.shape)
# print(pool_back_propogation(img_output, cache).shape)
