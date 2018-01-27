# Author: Xiao Zhang
import numpy as np
from math import *
from layer import *
import pdb, time
from im2col import im2col_cython, col2im_cython
np.random.seed(0)

def get_im2col_indices(x_shape, layer_config):
	# First figure out what the size of the output should be
	kernel_h, kernel_w ,pad_h, pad_w, stride_h, stride_w = layer_config
	N, C, H, W = x_shape
	out_height = int((H + 2 * pad_h - kernel_h) / stride_h + 1)
	out_width = int((W + 2 * pad_w - kernel_w) / stride_w + 1)

	i0 = np.repeat(np.arange(kernel_h), kernel_w)
	i0 = np.tile(i0, C)
	i1 = stride_h * np.repeat(np.arange(out_height), out_width)
	j0 = np.tile(np.arange(kernel_w), kernel_h * C)
	j1 = stride_w * np.tile(np.arange(out_width), out_height)
	i = i0.reshape(-1, 1) + i1.reshape(1, -1)
	j = j0.reshape(-1, 1) + j1.reshape(1, -1)

	k = np.repeat(np.arange(C), kernel_h * kernel_w).reshape(-1, 1)

	return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, layer_config):
	""" An implementation of im2col based on some fancy indexing """
	# Zero-pad the input

	kernel_h, kernel_w ,pad_h, pad_w, stride_h, stride_w = layer_config
	x_padded = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_w), (pad_h, pad_w)), mode='constant')
	k, i, j = get_im2col_indices(x.shape, layer_config)
	cols = x_padded[:, k, i, j]
	C = x.shape[1]
	cols = cols.transpose(1, 2, 0).reshape(kernel_h * kernel_w * C, -1)
	return cols


def col2im_indices(cols, x_shape, layer_config):
	""" An implementation of col2im based on fancy indexing and np.add.at """
	kernel_h, kernel_w ,pad_h, pad_w, stride_h, stride_w = layer_config
	N, C, H, W = x_shape
	H_padded, W_padded = H + 2 * pad_h, W + 2 * pad_w
	x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
	
	k, i, j = get_im2col_indices(x_shape, layer_config)
	
	cols_reshaped = cols.reshape(C * kernel_h * kernel_w, -1, N)
	cols_reshaped = cols_reshaped.transpose(2, 0, 1)
	np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
	if pad_h == 0 and pad_h == 0:
		return x_padded
	if pad_h == 0:
		return x_padded[:, :, :, pad_h:-pad_h]
	return x_padded[:, :, pad_h:-pad_h, pad_w:-pad_w]


class Conv_Base(Layer):
	def __init__ (self, output_channel, kernel_size, padding = 0, stride = 1, name = None):
		super(self.__class__, self)
		self.kernel_size = kernel_size
		self.padding = padding
		self.stride = stride
		self.output_dim = output_channel
		self.name = name
		self.i = None
		self.j = None
		self.k = None
		self.param_checking()
		self.layer_config = np.array((self.kernel_h, self.kernel_w, self.pad_h, self.pad_w, self.stride_h, self.stride_w)).flatten().astype(np.int32)
	
	def param_checking(self):
		if isinstance(self.padding, int):
			self.pad_h = self.padding
			self.pad_w = self.padding
		elif isinstance(self.padding, tuple):
			if len(self.padding) != 2:
				raise 'invalid input padding at ' + self.name
			self.pad_h = self.padding[0]
			self.pad_w = self.padding[1]
		elif isinstance(self.padding, np.ndarray):
			if len(self.padding.shape) != 2:
				raise 'invalid input padding at '  + self.name
			self.pad_h = self.padding[0]
			self.pad_w = self.padding[1]
		else:
			raise 'Input of padding should be only integer, tuple or numpy array'
		self.pad_w = int(self.pad_w)
		self.pad_h = int(self.pad_h)
		if isinstance(self.stride, int):
			self.stride_h = self.stride
			self.stride_w = self.stride
		elif isinstance(self.stride, tuple):
			if len(self.stride) != 2:
				raise Exception('invalid input stride at ' + self.name)
			self.stride_h = self.stride[0]
			self.stride_w = self.stride[1]
		elif isinstance(self.stride, np.ndarray):
			if len(self.stride.shape) != 2:
				raise Exception('invalid input stride at ' + self.name)
			self.stride_h = self.stride[0]
			self.stride_w = self.stride[1]
		else:
			raise Exception('Input of stride should be only integer, tuple or numpy array')

		if isinstance(self.kernel_size, int):
			self.kernel_h = self.kernel_size
			self.kernel_w = self.kernel_size
		elif isinstance(self.kernel_size, tuple):
			if len(self.kernel_size) != 2:
				raise Exception('invalid input stride at ' + self.name)
			self.kernel_h = self.kernel_size[0]
			self.kernel_w = self.kernel_size[1]
		elif isinstance(self.kernel_size, np.ndarray):
			if len(self.kernel_size.shape) != 2:
				raise Exception('invalid input stride at ' + self.name)
			self.kernel_h = self.kernel_size[0]
			self.kernel_w = self.kernel_size[1]
		else:
			raise Exception('Input of kernel_size should be only integer, tuple or numpy array')

	def input_checking(self, input_):

		if len(input_.shape) != 4:
			raise Exception('Convolution invalid input shape at ' + self.name, 'The input should be 4 dimensional array')
		self.height = input_.shape[2]
		self.width = input_.shape[3]
		if input_.shape[1] != self.input_dim:
			raise Exception('Convolution input channel number dismatch at ' + self.name)

		self.output_h = int(round((self.height + self.pad_h * 2 - self.kernel_h)/self.stride_h + 1))
		self.output_w = int(round((self.width + self.pad_w * 2 - self.kernel_w)/self.stride_w + 1))

class Conv2d(Conv_Base):
	def __init__(self, output_channel, kernel_size, padding = 0, stride = 1, bias = True, name = None):
		super(self.__class__, self).__init__(output_channel, kernel_size, padding, stride, name)
		self.kernel_size = kernel_size
		self.padding = padding
		self.stride = stride
		self.bias = bias
		self.output_dim = output_channel


	def init_param(self):
		self.w = (np.random.randn(self.output_dim, self.input_dim, self.kernel_size, self.kernel_size) * sqrt(2.0/(self.input_dim + self.output_dim))).astype(np.float32)
		self.b = np.zeros((1,self.output_dim, 1,1)).astype(np.float32)
		self.layer_param = [self.w, self.b]

	def forward(self, input_):
		self.input_ = input_
		N = self.input_.shape[0]
		self.input_checking(input_)
		input_col = im2col_cython(input_, self.layer_config)
		w_col = self.w.reshape(self.output_dim,self.kernel_h*self.kernel_w*self.input_dim)
		self.col_output = np.dot(w_col, input_col).reshape(self.output_dim,self.output_h,self.output_w,N).transpose(3,0,1,2)
		self.input_col = input_col
		if self.bias:
			self.col_output += self.b.reshape(1,-1,1,1)
		return self.col_output

	def backward(self, grad_input):
		self.grad_input = grad_input
		N = grad_input.shape[0]
		self.grad_b = np.sum(grad_input,axis = (0,2,3)).reshape(self.output_dim,-1)
		grad_input_reshaped = grad_input.transpose(1,2,3,0).reshape(self.output_dim,-1)
		self.grad_w = (np.dot(grad_input_reshaped,self.input_col.T)).reshape(self.w.shape)
		w_reshape = self.w.reshape(self.output_dim,-1)
		grad_col = np.dot(w_reshape.T, grad_input_reshaped)
		grad_ouput = col2im_cython(grad_col, np.array(self.input_.shape), self.layer_config)
		return grad_ouput

	def get_type(self):
		return 'conv2d'

	def get_param_for_optimizer(self):
		return [self.w, self.b],[self.grad_w,self.grad_b]

	def set_new_param(self, param):
		if len(param) == 1:
			self.w = param[0]
		else:
			self.w = param[0]
			self.b = param[1]

class MaxPool2d(Conv_Base):
	def __init__(self, kernel_size, padding = 0, stride = 1, name = None):
		super(self.__class__, self).__init__(None, kernel_size, padding, stride, name)
		self.kernel_size = kernel_size
		self.padding = padding
		self.stride = stride
		self.name = name
		self.output_dim = None
		self.i = None
		self.j = None
		self.k = None
		self.bi = None
		self.bj = None
		self.bk = None
		self.param_checking()

	def init_param(self):
		self.output_dim = self.input_dim
		self.layer_param = None

	def forward(self, input_):
		if self.output_dim is None:
			self.output_dim = self.input_dim
		self.input_checking(input_)
		self.input_ = input_
		N = self.input_.shape[0]
		input_reshaped = input_.reshape(N * self.output_dim, 1, self.height, self.width)
		input_col = im2col_cython(input_reshaped, self.layer_config)
		self.argmax_coor = np.argmax(input_col, axis = 0)
		max_pool_feat = input_col[self.argmax_coor,np.arange(self.argmax_coor.size)].reshape(self.output_h,self.output_w,N,self.input_dim)
		max_pool_feat = max_pool_feat.transpose(2,3,0,1)
		self.input_col = input_col
		return max_pool_feat

	def backward(self, grad_input):
		grad_col = np.zeros_like(self.input_col)
		grad_col[self.argmax_coor, np.arange(self.argmax_coor.size)] = grad_input.transpose(2,3,0,1).ravel()
		shape = np.array((self.input_.shape[0]*self.input_dim, 1, self.height, self.width)).astype(np.int64)
		grad_output = col2im_cython(grad_col, shape, self.layer_config)
		return grad_output.reshape(self.input_.shape)

	def get_type(self):
		return 'maxpool2d'

	def get_param_for_optimizer(self):
		return [None, None],[None,None]

