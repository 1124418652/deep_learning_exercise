# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.getcwd(), 'multi_layer_neural_network.py'))
from multi_layer_neural_network import *

test_file = os.path.join(os.getcwd(), '../datasets/test_catvnoncat.h5')
params_file = os.path.join(os.getcwd(), '../datasets/params.txt')

model = Mult_layer_network()
test_dataSet, test_labels = model.load_data(test_file, type = 'test')
test_dataSet = np.mat(test_dataSet)
test_dataSet = (test_dataSet-test_dataSet.mean(axis = 1)) / 255

params = model.load_params_from_file(params_file)
w_array = params['w_array']
b_array = params['b_array']
activate_func = params['activate_func']

def change_key_type(d):
	pairs = [(int(key), value) for (key, value) in d.items()]
	return dict(pairs)

w_array, b_array, activate_func = map(change_key_type, [w_array, b_array, activate_func])

model.testing(test_dataSet, test_labels, w_array, b_array, activate_func)