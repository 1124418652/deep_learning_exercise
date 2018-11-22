# -*- coding: utf-8 -*-
# Author: xhj

import os
import h5py
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image


class Logistic_Neural_Network(object):

    def __init__(self):
        pass

    def load_test_data(self, file_path):
        with h5py.File(file_path, 'r') as fr:
            data_set = np.array(fr['test_set_x'][:])
            labels = np.array(fr['test_set_y'][:])
            list_class = np.array(fr['list_classes'][:])
        return data_set, labels, list_class

    def load_train_data(self, file_path):
        with h5py.File(file_path, 'r') as fr:
            data_set = np.array(fr['train_set_x'][:])
            labels = np.array(fr['train_set_y'][:])
            list_class = np.array(fr['list_classes'][:])
        return data_set, labels, list_class

    def show_img(self, data_set, index = 0):
        fig = plt.figure(figsize = (6.4, 4.8))
        assert(index >= 0 and index <= data_set.shape[0])
        plt.imshow(data_set[index])
        plt.show()
    
    def _sigmod(self, w, x, b):
        if not isinstance(w, np.matrixlib.defmatrix.matrix):
            w = np.mat(w)
        if not isinstance(x, np.matrixlib.defmatrix.matrix):
            x = np.mat(x)
        assert(w.shape[1] == x.shape[0])
        return 1 / (1 + np.exp(w * x + b))
    
    def init_w_b(self, layer_num = 2, *args):
        layer = {}
        for i in range(layer_num):
            layer[i+1] = dict(w_size = args[i], \
                              w = np.random.randn(args[i][0], args[i][1]),\
                              b = np.zeros(args[i][0]))
        return layer

    def describe_data(self, data_set):
        if not isinstance(data_set, np.ndarray):
            data_set = np.array(data_set)
        data_num, image_shape = data_set.shape[0], data_set.shape[1:]
        print("number of this data: %d" %(data_num))
        print("image size: ", image_shape)

    def forward_propagation(self, data_set, w, b):
        pass

    def back_propagation(self, data_set, labels, iterate = 100):
        pass

if __name__ == '__main__':
    demo = Logistic_Neural_Network()
    data_set = demo.load_test_data("datasets/test_catvnoncat.h5")[0]
    #demo.show_img(data_set, 25)
    #print(data_set.shape)
    demo.describe_data(data_set)
    print(demo._sigmod([1,2,3], [[2],[3],[2]], 10))
    demo.init_w_b(2, [4,2],[3,2])
