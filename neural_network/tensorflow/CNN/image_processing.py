# -*- coding: utf-8 -*-
import numpy as np 
from PIL import Image

__all__  = ['zero_pad']


def zero_pad(arr, pad):
	"""
	parameters:
	@ arr: image dataset,
		   with the dimensions of (samples, im_height, im_width, im_channels)
	@ pad: 整数，对图像数据集中对每个样本的每一个通道（R,G,B）进行填充

	return:
	@ padded_img: image after padded
	"""

	padded_img = np.pad(arr, ((0, 0),
							  (pad, pad),
							  (pad, pad),
							  (0, 0)),
						'constant', constant_values = 0)
	return padded_img
