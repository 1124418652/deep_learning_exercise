#usr/bin/env python
# -*- coding: utf-8 -*-
"""
project: support vector machine
author: xhj
"""

import os
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import io


class SVM(object):

	def __init__(self):
		pass

	def show_data(self, x, y):
		
		

def demo():
	basePath = os.getcwd()
	filePath = os.path.join(basePath, "../ex6/ex6data1.mat")
	data_set = io.loadmat(filePath)
	x, y = data_set["X"], data_set['y']
	print(data_set)

def main():
	demo()

if __name__ == '__main__':
	main()