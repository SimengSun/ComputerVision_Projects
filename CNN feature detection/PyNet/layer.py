# Author: Xiao Zhang
import os
import numpy as np
import upsample_function
from math import *
import pdb
np.random.seed(0)
class Layer(object):
	
	def __init__(self , name = None):
		self.name = name
		self.train = True
		
	def forward(self, input_):
		raise NotImplementedError

	def backward(self, grad_input):
		raise NotImplementedError

	def get_name(self):
		return self.name

	def get_param(self):
		return self.layer_param

	def set_input_dim(self, input_dim):
		self.input_dim = input_dim
		self.init_param()
	
	def init_param():
		pass
	
	def get_output_dim(self):
		return self.output_dim

	def __dict__(self):
		raise NotImplementedError

	def get_param_for_optimizer(self):
		raise NotImplementedError

	def set_new_param(self, param):
		pass

class Linear(Layer):

	def __init__(self, input_dim, output_dim, name = None, bias = True):
		super(self.__class__, self).__init__(name)
		self.output_dim = output_dim
		self.bias = bias
		self.i_dim = input_dim

	def init_param(self):
		self.input_dim = self.i_dim
		self.w = (np.random.randn(self.input_dim,self.output_dim) * sqrt(2.0/(self.input_dim+self.output_dim))).astype(np.float32)
		self.b = None
		if self.bias:
			self.b = np.zeros((1,self.output_dim))
		self.layer_param = [self.w, self.b]

	def forward(self, input_):
		self.input = input_
		self.output = np.dot(input_, self.w)
		if self.bias:
			self.output += self.b
		return self.output

	def get_type(self):
		return 'linear'

	def backward(self, grad_input):
		self.grad_b = np.mean(grad_input, axis = 0)
		self.grad_w = self.input.T.dot(grad_input)
		self.grad_output = grad_input.dot(self.w.T)
		return self.grad_output

	def get_param_for_optimizer(self):
		return [self.w, self.b],[self.grad_w,self.grad_b]

	def set_new_param(self, param):
		if len(param) == 1:
			self.w = param[0]
		else:
			self.w = param[0]
			self.b = param[1]

class Upsample(Layer):
	def __init__(self, size = None, scale = None, name = None):
		super(self.__class__, self).__init__(name)
		self.size = size
		self.scale = scale
		if isinstance(self.size, tuple):
			self.size_h = self.size[0]
			self.size_w = self.size[1]
		elif self.size is not None:
			self.size_h = self.size
			self.size_w = self.size

		if isinstance(self.scale, tuple):
			self.scale_h = self.scale[0]
			self.scale_w = self.scale[1]
		elif self.scale is not None:
			self.scale_h = self.scale
			self.scale_w = self.scale

		if self.scale is None and self.size is None:
			raise Exception('The size and scale cant be both None')


	def init_param(self):
		self.output_dim = self.input_dim
		self.layer_param = None

	def forward(self, input_):
		self.input_h = input_.shape[2]
		self.input_w = input_.shape[3]
		if self.size is None:
			self.output_h = int(round(self.scale_h * self.input_h))
			self.output_w = int(round(self.scale_w * self.input_w))
		elif self.scale is None:
			self.output_h = int(round(self.size_h))
			self.output_w = int(round(self.size_w))
		# upsample_val = self.bilinear(input_, self.output_h, self.output_w)
		upsample_val = np.zeros((input_.shape[0], input_.shape[1], self.output_h, self.output_w)).astype(np.float32)
		upsample_val = upsample_function.bilinear_forward(input_.astype(np.float32), upsample_val)
		return upsample_val

	def backward(self, grad_input):
		grad_output = np.zeros((grad_input.shape[0], grad_input.shape[1], self.input_h, self.input_w)).astype(np.float32)
		grad_output = upsample_function.bilinear_backward(grad_input.astype(np.float32), grad_output)

		# grad_output = self.bilinear(grad_input, self.input_h, self.input_w)
		return grad_output

	def bilinear(self, input, output_h, output_w):
		
		input_h = input.shape[2]
		input_w = input.shape[3]
		w = input_w
		h = input_h
		input_dim = input.shape[1]
		input_batch = input.shape[0]

		mesh_x, mesh_y = np.meshgrid(np.arange(output_w), np.arange(output_h))
		scale_h = output_h * 1.0/input_h
		scale_w = output_w * 1.0/input_w
		mesh_x = mesh_x / scale_w
		mesh_y = mesh_y / scale_h
		xq = mesh_x.flatten()
		yq = mesh_y.flatten()

		x_floor = np.floor(xq).astype(np.int32)
		y_floor = np.floor(yq).astype(np.int32)
		x_ceil = np.ceil(xq).astype(np.int32)
		y_ceil = np.ceil(yq).astype(np.int32)

		x_floor[x_floor<0] = 0
		y_floor[y_floor<0] = 0
		x_ceil[x_ceil<0] = 0
		y_ceil[y_ceil<0] = 0

		x_floor[x_floor>=w-1] = w-1
		y_floor[y_floor>=h-1] = h-1
		x_ceil[x_ceil>=w-1] = w-1
		y_ceil[y_ceil>=h-1] = h-1

		v1 = input[:,:,y_floor, x_floor]
		v2 = input[:,:,y_floor, x_ceil]
		v3 = input[:,:,y_ceil, x_floor]
		v4 = input[:,:,y_ceil, x_ceil]

		lh = yq - y_floor
		lw = xq - x_floor
		hh = 1 - lh
		hw = 1 - lw

		w1 = hh * hw
		w2 = hh * lw
		w3 = lh * hw
		w4 = lh * lw

		interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4
		return interp_val.reshape(input_batch, input_dim, output_h, output_w)

	def get_type(self):
		return 'upsample'

	def get_param_for_optimizer(self):
		return [None],[None]


class BatchNorm1D(Layer):

	def __init__(self, momentum = 0.9, name = None):
		super(self.__class__, self).__init__(name)
		self.beta = beta
		self.gamma = gamma
		self.momentum = momentum
		self.train = True
		self.eps = 1e-5

	def init_param(self):
		self.output_dim = self.input_dim
		self.r_mean = np.zeros((1,self.output_dim)).astype(np.float32)
		self.r_var = np.ones((1,self.output_dim)).astype(np.float32)
		self.beta = np.zeros((1,self.output_dim)).astype(np.float32)
		self.gamma = (np.random.rand(1,self.output_dim) * sqrt(2.0/(self.output_dim))).astype(np.float32)
		self.layer_param = [self.beta, self.gamma]

	def forward(self, input_):
		self.input_ = input_
		if self.train:
			self.mu = np.mean(self.input_, axis =0, keepdims = True)
			self.var = np.var(self.input_, axis =0, keepdims = True)
			self.r_mean = self.r_mean * self.momentum + (1 - self.momentum) * self.mu
			self.r_var = self.r_var * self.momentum + (1 - self.momentum) * self.var
			self.input_norm = (self.input_ - self.mu) / np.sqrt(self.var + self.eps)
			output = (self.input_norm * self.gamma) + self.beta
		else:
			input_norm = (self.input_ - self.r_mean)/np.sqrt(self.r_var + self.eps) 
			output = (input_norm * self.gamma) + self.beta
		return output

	def backward(self, grad_input):
		N = grad_input.shape[0]
		output_term1 =  (1. / N) * self.gamma * (self.var + self.eps)**(-1. / 2.)
		output_term2 = N * grad_input
		output_term3 = np.sum(grad_input, axis=0)
		output_term4 = self.input_norm * np.sum(self.grad_input * (self.input_ - self.mu), axis=0)
		grad_output = output_term1 * (output_term2 - output_term3 - output_term4)
		self.grad_gamma = np.mean(grad_input * self.input_norm, axis = 0)
		self.grad_beta = np.mean(grad_input, axis = 0)
		return grad_output

	def get_type(self):
		return 'batchnorm1d'

	def get_param_for_optimizer(self):
		return [self.beta, self.gamma],[self.grad_beta,self.grad_gamma]

	def set_new_param(self, param):
		self.beta = param[0]
		self.gamma = param[1]

class BatchNorm2D(Layer):

	def __init__(self, momentum = 0.9, name = None):
		super(self.__class__, self).__init__(name)
		self.momentum = momentum
		self.train = True
		self.eps = 1e-5

	def init_param(self):
		self.output_dim = self.input_dim
		self.r_mean = np.zeros((1,self.output_dim)).astype(np.float32)
		self.r_var = np.ones((1,self.output_dim)).astype(np.float32)
		self.beta = np.zeros((1,self.output_dim)).astype(np.float32)
		self.gamma = (np.random.rand(1,self.output_dim) * sqrt(2.0/(self.output_dim))).astype(np.float32)
		self.layer_param = [self.beta, self.gamma]

	def forward(self, input_):
		self.input_shape = input_.shape
		self.input_ = input_.transpose(0,2,3,1).reshape(-1, self.input_shape[1])
		if self.train:
			self.mu = np.mean(self.input_, axis =0, keepdims = True)
			self.var = np.var(self.input_, axis =0, keepdims = True)
			self.r_mean = self.r_mean * self.momentum + (1 - self.momentum) * self.mu
			self.r_var = self.r_var * self.momentum + (1 - self.momentum) * self.var
			self.input_norm = (self.input_ - self.mu) / np.sqrt(self.var + self.eps)
			output = (self.input_norm * self.gamma) + self.beta
		else:
			input_norm = (self.input_ - self.r_mean)/np.sqrt(self.r_var + self.eps) 
			output = (input_norm * self.gamma) + self.beta
		return output.reshape(self.input_shape[0], self.input_shape[2],self.input_shape[3],self.input_shape[1]).transpose(0,3,1,2)

	def backward(self, grad_input):
		self.grad_input = grad_input.transpose(0,2,3,1).reshape(-1, self.input_shape[1])
		sum_val = np.sum(self.grad_input, axis = 0,keepdims = True)
		N = self.grad_input.shape[0]
		dotp = np.sum(self.grad_input * (self.input_ - self.mu), axis = 0, keepdims = True)
		invstd =   1/np.sqrt(self.var + self.eps)
		k = dotp * invstd * invstd / N
		repeat_gradinput = self.grad_input
		mean_gradinput = sum_val / N
		grad_output_k = k * (self.input_ - self.mu)
		grad_output = self.gamma * invstd * (repeat_gradinput - mean_gradinput - grad_output_k)
		self.grad_gamma = np.sum(dotp * invstd, axis = 0,keepdims = True)
		self.grad_beta = sum_val
		return grad_output.reshape(self.input_shape[0], self.input_shape[2],self.input_shape[3],self.input_shape[1]).transpose(0,3,1,2)


	def get_type(self):
		return 'batchnorm2d'

	def get_param_for_optimizer(self):
		return [self.beta, self.gamma],[self.grad_beta,self.grad_gamma]

	def set_new_param(self, param):
		self.beta = param[0]
		self.gamma = param[1]


class Relu(Layer):
	def __init__(self, name = None):
		super(self.__class__, self).__init__(name)

	def init_param(self):
		self.output_dim = self.input_dim
		self.layer_param = None

	def forward(self, input_):
		self.input_ = input_
		return np.maximum(input_, 0)

	def backward(self, grad_input):
		grad_input[self.input_<0] = 0
		return grad_input

	def get_type(self):
		return 'relu'

	def get_param_for_optimizer(self):
		return [None],[None]

class Sigmoid(Layer):
	def __init__(self, name = None):
		super(self.__class__, self).__init__(name)

	def init_param(self):
		self.output_dim = self.input_dim
		self.layer_param = None

	def sigmoid_func(self, x):
		return 1.0 / (1.0 + np.exp(-x));

	def forward(self, input_):
		self.sigmoid_val = self.sigmoid_func(input_)
		return self.sigmoid_val

	def backward(self, grad_input):
		return grad_input * self.sigmoid_val * (1 - self.sigmoid_val)

	def get_type(self):
		return 'sigmoid'

	def get_param_for_optimizer(self):
		return [None],[None]

class Flatten(Layer):
	def __init__(self, name = None):
		super(self.__class__, self).__init__(name)

	def init_param(self):
		self.output_dim = self.input_dim
		self.layer_param = None

	def forward(self, input_):
		# shift = input_ - np.max(input_)
		self.shape = input_.shape
		return input_.reshape(input_.shape[0],-1)

	def backward(self, grad_input):
		return grad_input.reshape(self.shape)

	def get_type(self):
		return 'flatten'

	def get_param_for_optimizer(self):
		return [None],[None]

class Softmax(Layer):
	def __init__(self, name = None):
		super(self.__class__, self).__init__(name)
	
	def init_param(self):
		self.output_dim = self.input_dim
		self.layer_param = None

	def forward(self, input_):
		# shift = input_ - np.max(input_)
		shift = input_
		exp_scores = np.exp(shift)
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
		self.probs = probs
		return probs

	def backward(self, grad_input):
		N = grad_input.shape[0]
		C = grad_input.shape[1]

		grad = -1*self.probs.reshape(N,C,1,-1)*self.probs.reshape(N,1,C,-1)
		if len(self.probs.shape) == 2:
			probs = self.probs[...,np.newaxis]
		grad[:,np.arange(C),np.arange(C),:] = probs*(1-probs)
		self.grad_output = np.sum(grad_input.reshape(N,C,1,-1)*grad, axis = 1)
		return self.grad_output.reshape(self.probs.shape)

	def get_type(self):
		return 'softmax'

	def get_param_for_optimizer(self):
		return [None],[None]
		
class L2_loss(Layer):
	def __init__(self, average = True, name = None):
		super(self.__class__, self).__init__(name)
		self.average = average

	def init_param(self):
		self.output_dim = 1
		self.layer_param = None 

	def forward(self, input_, label):
		if len(label.shape) == 1:
			label = label[:,np.newaxis]
		N = input_.shape[0]
		self.input = input_
		self.label = label
		loss = np.sum((input_ - label)**2)
		if self.average:
			loss /= N
		return loss

	def backward(self, grad_input):
		if self.average:
			norm = 2.0/(self.input.size)
		else:
			norm = 2.0
		grad_output = norm * (self.input - self.label)
		return grad_output

	def get_type(self):
		return 'l2_loss'

class Binary_cross_entropy_loss(Layer):
	def __init__(self, average = True, name = None):
		self.average = average
		self.eps = 1e-12
	def init_param(self):
		self.output_dim = 1
		self.layer_param = None

	def forward(self, input_, label):
		if len(label.shape) == 1:
			label = label[:,np.newaxis]
		if np.sum(np.logical_or(input_ < 0, input_ > 1)) > 0:
			raise Exception('The input to BCEloss should between 0~1, please to make sure you use sigmoid before BCELoss') 
		self.input = input_
		self.label = label
		if self.average:
			self.count = input_.size
		else:
			self.count = 1
		loss = -1*np.sum(label*np.log(input_+self.eps) + (1-label)*np.log(1-input_+self.eps))
		loss = loss / self.count 
		return loss

	def backward(self, grad_input):
		
		self.grad_output = -1.0/self.count * (self.label - self.input) / ((1. - self.input + self.eps) * (self.input + self.eps))
		return self.grad_output

class Cross_entropy_loss(Layer):
	def __init__(self, average = True, name = None):
		super(self.__class__, self).__init__(name)
		self.average = average

	def init_param(self):
		self.output_dim = 1
		self.layer_param = None

	def forward(self, input_, label):
		N = input_.shape[0]
		self.input = input_
		self.label = label.astype(np.int32)
		self.loss = np.sum(-1*np.log(input_[range(N),self.label]))
		if self.average:
			return  self.loss / N
		return self.loss

	def backward(self, grad_input):
		N = self.input.shape[0]
		grad_output = np.zeros_like(self.input)
		grad_output[range(N),self.label] = -1 / self.input[range(N),self.label]
		return grad_output

	def get_param_for_optimizer(self):
		return [None],[None]

	def get_type(self):
		return 'cross_entropy_loss'