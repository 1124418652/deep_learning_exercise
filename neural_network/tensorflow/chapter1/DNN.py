# -*- coding: utf-8 -*-

import mnist_test
import numpy as np 
import matplotlib.pyplot as plt 

class DNN(object):

	def __init__(self, network_struct = []):
		self.network_struct = network_struct

	def activate(self, z_array, mode = 'sigmod'):
		if 'sigmod' == mode.lower():
			a = 1 / np.exp(z)
			da = 