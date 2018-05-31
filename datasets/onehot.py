#!/usr/bin/env python
# encoding: utf-8
"""
@version: 1.0
@author: liaoliwei
@contact: levio@pku.edu.cn
@file: onehot.py
@time: 2018/5/28 11:28
"""
import numpy as np


def onehot_transform(filename):
	with open(filename) as f:
		text = f.read()
		text_list = text.split(',')
		print(len(text_list))
	all_text_list = []
	for num in text_list:
		onehot_template = [0]*10
		onehot_template[int(num)] = 1
		all_text_list.append(onehot_template)
	text_array = np.array(all_text_list)
	print(np.shape((text_array)))
	return text_array


if __name__ == '__main__':
	onehot_transform('t10k-label.txt')