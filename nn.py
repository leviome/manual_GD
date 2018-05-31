#!/usr/bin/env python
# encoding: utf-8
"""
@version: 1.0
@author: liaoliwei
@contact: levio@pku.edu.cn
@file: fc_nn.py
@time: 2018/5/30 0:39
"""
import numpy as np
import matplotlib.pyplot as plt
import time


def ReLU(x):
	xc = np.copy(x)
	xc[xc < 0] = 0
	return xc


def d_ReLU(x):
	mask = (x > 0) * 1.0
	return mask


def MSE(x):
	length = len(x)
	return (x * x).sum() / length


def d_MSE(x):
	length = len(x)
	return 2 * x / length


def FullyConnect(last_out, weights, bias):
	nodes = len(weights)
	L = []
	for i in range(nodes):
		temp = np.dot(last_out, weights[i])
		L.append(temp)
	return np.array(L) + bias


np.random.seed(12314)
Neural_nodes = 500
w1 = (np.random.rand(Neural_nodes, 784) - 0.5) * 0.01
w2 = (np.random.rand(10, Neural_nodes) - 0.5) * 0.01
b1 = np.zeros(Neural_nodes)
b2 = np.zeros(10)


# w1 = np.zeros((Neural_nodes, 784))
# w2 = np.zeros((10, Neural_nodes))


def inference(img):
	raveled_img = img.ravel()
	node = []
	layer1 = FullyConnect(raveled_img, w1, b1)
	node.append(layer1)
	layer2 = FullyConnect(layer1, w2, b2)
	return layer2, node


def onehot_transform(filename):
	with open(filename) as f:
		text = f.read()
		text_list = text.split(',')
	all_text_list = []
	for num in text_list:
		onehot_template = [0] * 10
		onehot_template[int(num)] = 1
		all_text_list.append(onehot_template)
	text_array = np.array(all_text_list)
	return text_array


def arise_multiply(a, b):
	# L = []
	# for j in range(len(a)):
	# 	temp = a[j] * b
	# 	L.append(temp)
	# return np.array(L)
	ac = np.expand_dims(a.ravel(), 0).T
	bc = np.expand_dims(b.ravel(), 0)
	return bc * ac


def run():
	data_arr = np.load('datasets/t10k-images-idx3-ubyte.npy')
	inference(data_arr[0])


def read_txt(filename):
	with open(filename) as f:
		text = f.read()
		text_list = text.split(',')
		return text_list


lr_base = 0.00001
decay = 0.99
global_step = 0
decay_step = 10
# lr = lr_base

data_arr = np.load('datasets/train-images-idx3-ubyte.npy')
label_arr = onehot_transform('datasets/train-label.txt')
test_label_arr = read_txt('datasets/t10k-label.txt')
test_arr = np.load('datasets/t10k-images-idx3-ubyte.npy')


def train():
	t1 = time.time()
	global w1, w2
	global b1, b2
	x_axis = []
	y1_axis = []
	y2_axis = []
	loss_line = []
	cnt = 0
	# assert len(data_arr) == len(label_arr)
	for i in range(60000):
		decay_coex = int(i / 200)
		decay_rate = pow(decay, decay_coex)
		lr = lr_base * decay_rate
		raveled_img = data_arr[i].ravel()
		y_ = label_arr[i]
		y, node = inference(data_arr[i])
		loss = MSE(y - y_)
		grad2_part = d_MSE(y - y_)
		b2_grad = grad2_part
		grad2 = arise_multiply(grad2_part, node[0])

		b1_grad = np.dot(b2_grad, w2)
		grad1_part = np.dot(np.expand_dims(d_MSE(y - y_), 0), w2)

		grad1 = arise_multiply(grad1_part[0], raveled_img)
		b2 = b2 - lr * b2_grad
		b1 = b1 - lr * b1_grad
		w2 = w2 - lr * grad2
		w1 = w1 - lr * grad1
		tx = test_arr[i % 10000]
		layer_out, t_node = inference(tx)
		infer = np.argmax(layer_out)

		if infer == int(test_label_arr[i % 10000]):
			cnt += 1
		if i % 1000 == 0:
			# print(decay_rate)
			print('time costs:', time.time() - t1)
			print('epoch =', int(i / 1000), end='')
			print('  loss =', loss, end='')
			print('  acc =', cnt / 1000)
			x_axis.append(i)
			loss_line.append(loss)
			y1_axis.append(cnt / 1000)
			y2_axis.append(decay_rate)
			cnt = 0
	plt.plot(x_axis, y1_axis, 'ro-', label='accuracy')
	plt.plot(x_axis, y2_axis, 'go-', label='learning_rate')
	plt.plot(x_axis, loss_line, 'bo-', label='loss')
	plt.legend(loc='best')
	plt.show()


if __name__ == '__main__':
	train()
