#!/usr/bin/env python
# encoding: utf-8
"""
@version: 1.0
@author: liaoliwei
@contact: levio@pku.edu.cn
@file: format_decoder.py
@time: 2018/5/15 14:18
"""
import struct
from PIL import Image
import numpy as np
import gzip
import os.path as osp


class Format_decoder():
	def __init__(self):
		pass

	def save_images_into_file(filename):
		g_file = gzip.GzipFile(filename)
		# 创建gzip对象
		buf = g_file.read()
		g_file.close()
		index = 0
		magic, images, rows, columns = struct.unpack_from('>IIII', buf, index)
		index += struct.calcsize('>IIII')
		for i in range(images):
			image = Image.new('L', (columns, rows))
			for x in range(rows):
				for y in range(columns):
					image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
					index += struct.calcsize('>B')
			print('save ' + str(i) + 'image')
			image.save('imgs/' + str(i) + '.png')


	def reform_data_into_npy(filename):
		saveFilename = filename.split('.')[0]+'.npy'
		if osp.exists(saveFilename):
			print(saveFilename+' has already existed')
			return
		g_file = gzip.GzipFile(filename)
		# 创建gzip对象
		buf = g_file.read()
		g_file.close()
		index = 0
		magic, images, rows, columns = struct.unpack_from('>IIII', buf, index)
		index += struct.calcsize('>IIII')
		all_array_list = []
		for i in range(images):
			if i%1000 == 0:
				percentage = round(i*100/images, 0)
				print('processing: %s %%'% percentage)
			image = Image.new('L', (columns, rows))
			for x in range(rows):
				for y in range(columns):
					image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
					index += struct.calcsize('>B')
			narray = np.array(image)
			all_array_list.append(narray)
		all_arrays = np.array(all_array_list)
		print('processing successfully!')
		print(np.shape(all_arrays))
		np.save(saveFilename, all_arrays)

	def read_label(filename):
		saveFilename = filename.split('-')[0]+'-label.txt'
		if osp.exists(saveFilename):
			print(saveFilename+' has already existed')
			return
		g_file = gzip.GzipFile(filename)
		# 创建gzip对象
		buf = g_file.read()
		g_file.close()
		index = 0
		magic, labels = struct.unpack_from('>II', buf, index)
		index += struct.calcsize('>II')
		labelArr = [0] * labels
		for x in range(labels):
			labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
			index += struct.calcsize('>B')
		save = open(saveFilename, 'w')
		save.write(','.join(map(lambda x: str(x), labelArr)))
		save.write('\n')
		save.close()
		print('save labels success')


if __name__ == '__main__':
	Format_decoder.reform_data_into_npy('t10k-images-idx3-ubyte.gz')
	Format_decoder.read_label('t10k-labels-idx1-ubyte.gz')
	Format_decoder.reform_data_into_npy('train-images-idx3-ubyte.gz')
	Format_decoder.read_label('train-labels-idx1-ubyte.gz')

