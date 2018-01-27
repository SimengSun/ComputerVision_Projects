# Author: Xiao Zhang
import numpy as np
import pdb,time
import pickle

class Model(object):
	def __init__(self, input_layers, loss_layer, optimizer = None, lr_decay = None):
		self.num_layer = len(input_layers)
		self.input_layers = input_layers
		self.optimizer = optimizer
		self.loss_layer = loss_layer	
		self.lr_decay = lr_decay
		self.base_lr = None

	def set_input_channel(self, dim):
		self.input_dim = dim
		self.layer_name_list, self.layer_name_dict, self.param_dict = self.layer_init()
		self.show_layer_name()

	def forward(self,input_, label = None):
		self.layer_forward = {}
		input_pointer = input_
		for layer_name in self.layer_name_list:
			self.layer_forward[layer_name] = self.layer_name_dict[layer_name].forward(input_pointer)
			input_pointer = self.layer_forward[layer_name].copy()
		if label is None:
			return input_pointer
		loss = self.loss_layer.forward(input_pointer,label)
		return loss, input_pointer
	
	def show_layer_name(self):
		for layer_name in self.layer_name_list:
			print(('layer name: ' + layer_name).ljust(25),  ('layer type: ' + self.layer_name_dict[layer_name].get_type()).ljust(25))

	def get_layer_output(self, layer_name):
		if layer_name in self.layer_forward.keys():
			return self.layer_forward[layer_name]
		else:
			raise Exception('The input layer name '+ layer_name +' is not existed, please check again!')

	def get_layer_grad(self, layer_name):
		if layer_name in self.layer_backward.keys():
			return self.layer_backward[layer_name]
		else:
			raise Exception('The input layer name '+ layer_name +' is not existed, please check again!')

	def backward(self, loss):
		self.layer_backward ={}
		back_pointer = self.loss_layer.backward(loss)
		for layer_name in reversed(self.layer_name_list):
			self.layer_backward[layer_name] = self.layer_name_dict[layer_name].backward(back_pointer)
			back_pointer = self.layer_backward[layer_name].copy()

	def train(self, is_train):
		for layer_name in self.layer_name_list:
			self.layer_name_dict[layer_name].train = is_train

	def update_param(self):
		if self.optimizer is None:
			raise 'optimizer is not defined yet'
		if self.base_lr is None:
			self.base_lr = self.optimizer.lr_rate
		for layer_name in self.layer_name_list:
			new_hist = self.optimizer.update(self.layer_name_dict[layer_name], self.grad_history[layer_name])
			self.grad_history[layer_name] = new_hist
		if self.lr_decay is not None:
			self.optimizer.lr_rate = self.base_lr * self.lr_decay.step()

	def save_model(self, path):
		layer_weight_dict={}
		for key in self.layer_name_dict.keys():
			if self.layer_name_dict[key].get_type() == 'linear':
				layer_weight_dict[key] = [self.layer_name_dict[key].w, self.layer_name_dict[key].b]
			elif 'batchnorm' in self.layer_name_dict[key].get_type():
				layer_weight_dict[key] = [self.layer_name_dict[key].beta, self.layer_name_dict[key].gamma, self.layer_name_dict[key].r_mean, self.layer_name_dict[key].r_var]
			elif self.layer_name_dict[key].get_type() == 'conv2d':
				layer_weight_dict[key] = [self.layer_name_dict[key].w, self.layer_name_dict[key].b]
		save_dict = {'grad_hist':self.grad_history, 'layer_name':self.layer_name_dict.keys(), 'layer_weight':layer_weight_dict}
		with open(str(path), 'wb') as handle:
			pickle.dump(save_dict, handle)
		print('model saved at :' + str(path))

	def load_model(self, path):
		with open(str(path), 'rb') as handle:
			 save_dict = pickle.load(handle)
		grad_hist = save_dict['grad_hist']
		layer_name = save_dict['layer_name']
		layer_weight = save_dict['layer_weight']
		current_layer_name = self.layer_name_dict.keys()
		if len(current_layer_name) != len(layer_name):
			raise Exception('The saved model and current model is not consistent!')
		for name in layer_name:
			if name not in current_layer_name:
				raise Exception('The saved layer name: ' + name+ ' doesnot match current model')
		for key in layer_weight.keys():
			val = layer_weight[key]
			if self.layer_name_dict[key].get_type() == 'linear':
				self.layer_name_dict[key].w = val[0]
				self.layer_name_dict[key].b = val[1]
			elif 'batchnorm' in self.layer_name_dict[key].get_type():
			 	self.layer_name_dict[key].beta = val[0]
			 	self.layer_name_dict[key].gamma = val[1]
			 	self.layer_name_dict[key].r_mean = val[2]
			 	self.layer_name_dict[key].r_var = val[3]
			elif self.layer_name_dict[key].get_type() == 'conv2d':
				self.layer_name_dict[key].w = val[0]
				self.layer_name_dict[key].b = val[1]
		self.grad_history = grad_hist
		print('model is successfully loaded!')
		
	def layer_init(self):
		layer_name_list = []
		name_dict = {}
		param_dict = {}
		layer_type_counter = {}
		self.grad_history = {}
		input_dim = self.input_dim
		for layer in self.input_layers:
			layer.set_input_dim(input_dim)
			input_dim = layer.get_output_dim()
			current_type = layer.get_type()
			if current_type in layer_type_counter:
				if layer.get_name() is None:
					current_layer_name = current_type +'_'+str(layer_type_counter[current_type])
					layer.name = current_layer_name
				else:
					current_layer_name = layer.get_name()
				layer_name_list.append(current_layer_name)
				name_dict[current_layer_name] = layer
				param_dict[current_layer_name] = layer.get_param()
				layer_type_counter[current_type] += 1
			else:
				if layer.get_name() is None:
					current_layer_name = current_type +'_0'
					layer.name = current_layer_name
				else:
					current_layer_name = layer.get_name()
				layer_name_list.append(current_layer_name)
				name_dict[current_layer_name] = layer
				param_dict[current_layer_name] = layer.get_param()
				layer_type_counter[current_type] = 1
			if layer.get_param() is not None:
				self.grad_history[current_layer_name] = [None]*len(layer.get_param())
			else:
				self.grad_history[current_layer_name] = None
		return layer_name_list, name_dict, param_dict
