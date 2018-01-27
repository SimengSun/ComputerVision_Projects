# Author: Xiao Zhang
import numpy as np
import pdb

class SGD_Optimizer(object):
	def __init__(self, lr_rate, weight_decay = 5e-4, momentum = 0.99):
		self.lr_rate = lr_rate
		self.weight_decay = weight_decay
		self.momentum = momentum
	
	def update(self, layer, hist):
		param_list, diff_list = layer.get_param_for_optimizer()
		new_param_list = []
		new_hist_list = None
		if hist is not None:
			hist_val = hist
			new_hist_list = [None]*len(hist)
		index = 0 
		for param,diff in zip(param_list,diff_list):
			if param is None:
				new_param_list.append(None)
			else:
				grad = self.do_update(param, diff, hist_val[index])
				new_param_list.append(param - self.lr_rate * grad.reshape(param.shape))
				new_hist_list[index] = grad
				index += 1
		layer.set_new_param(new_param_list)
		return new_hist_list

	def do_update(self, param, diff, hist_val = None):
		grad = diff.reshape(param.shape) + param * self.weight_decay
		if hist_val is not None:
			grad = grad + self.momentum * hist_val
		return grad

class Decay_learning_rate(object):
	def __init__(self, decay_step = 500, base = 0.96, staircase = True):
		self.decay_step = decay_step
		self.base = base
		self.staircase = staircase
		self.counter = 0

	def step(self):
		self.counter += 1
		if self.staircase:
			return self.base ** (self.counter / self.decay_step)
		return self.base ** (self.counter * 1.0 / self.decay_step)
