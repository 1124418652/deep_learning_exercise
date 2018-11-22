#usr/bin/env python
# -*- coding: utf-8 -*-
"""
# project: exercise 1
# author:  xhj
# email:   1124418652@qq.com
# date:    2018/ 10/19
"""

import os
import time
import csv
import numpy as np 

class EXE1(object):
	def __init__(self):
		pass

	def load_data(self, path):
		with open(path, "r") as fr:
			reader = csv.reader(fr)

			for i in reader:
				print(i)

def main():
	exe1 = EXE1()
	exe1.load_data("../ex1/ex1data2.txt")

if __name__ == '__main__':
	main()