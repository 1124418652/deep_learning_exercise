#usr/bin/env python
# -*- coding: utf-8 -*-
"""
# project: PCA
# author:  xhj
# email:   1124418652@qq.com
# date:    2018/ 11/07
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from scipy import io
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

__all__ = ['PCA', 'Plot']


class PCA(object):

	def __init__(self):
		pass

	def __centralization(self, data):
		if np.ndarray != type(data):
			data = np.array(data)
		num = len(data)
		mean = data.sum(axis = 0) / num 
		return data - mean

	def covariance(self, data):
		central_data = np.mat(self.__centralization(data))
		simple_num, feat_num = central_data.shape
		cov_mat = central_data.T * central_data / (simple_num - 1)
		return cov_mat, central_data

	def choose_principal_comp(self, data):
		cov_mat, central_data = self.covariance(data)
		eigenvalue, eigenvector = np.linalg.eig(cov_mat)
		principal_vector = eigenvector[:, np.argmax(eigenvalue)]
		final_data = central_data * principal_vector
		return final_data


class Plot(object):

	figsize = (6.4, 4.8)
	dpi = 100
	left = None
	right  = None
	top = None
	bottom = None
	wspace = None
	hspace = None

	@property
	def title(self):
		return self.__title

	def show(self):
		pass


def demo():
	base_path = os.getcwd()
	file_path = os.path.join(base_path, "../ex7/ex7data2.mat")
	mat_data = io.loadmat(file_path)
	# print(mat_data)
	x = mat_data['X']
	pca = PCA()
	data_test = [[2.5, 2.4],
				[0.5, 0.7],
				[2.2, 2.9],
				[1.9, 2.2],
				[3.1, 3.0],
				[2.3, 2.7],
				[2, 1.6],
				[1, 1.1],
				[1.5, 1.6],
				[1.1, 0.9]]
	data_test2 = [[0.69, 0.49],
				[-1.31, -1.21],
				[0.39, 0.99],
				[0.09, 0.29],
				[1.29, 1.09],
				[0.49, 0.79],
				[0.19, -0.31],
				[-0.81, -0.81],
				[-0.31, -0.31],
				[-0.71, -1.01]]
	data_test = np.array(data_test)
	data_new = pca.choose_principal_comp(data_test)
	print(data_new)
	plt.plot(data_test[:, 0], data_test[:, 1], '+')
	plt.show()

def main():
	demo()

if __name__ == '__main__':
	main()
