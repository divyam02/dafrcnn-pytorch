import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np

def flatten(x):
	N = list(x.size())[0]
	return x.view(N, -1)

class GRL(Function):
	@staticmethod
	def forward(self, x, beta):
		self.beta = beta
		return x.view_as(x)

	@staticmethod
	def backward(self, grad_output):
		grad_output = grad_output*(-1)*self.beta
		return grad_output, None

class domain_image_classifier(nn.Module):
	def __init__(self):
		super(domain_image_classifier, self).__init__()
		self.da_conv_ss_6 = 	nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0, stride=1)
		self.da_relu_ss_6 =  	nn.ReLU()
		self.da_conv_score =	nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1, padding=0, stride=1)
		# label resize layer??
		# uses Softmax with Loss	
		
	def forward(self, img_lvl_features, beta=1.0):
		x = GRL.apply(img_lvl_features, beta)
		x = self.da_conv_ss_6(x)
		x = self.da_relu_ss_6(x)
		x = self.da_conv_score(x)
		x = x.view(-1)
		return x


class domain_instance_classifier(nn.Module):
	def __init__(self):
		super(domain_instance_classifier, self).__init__()
		self.dc_ip_1 = 		nn.Linear(2048, 1024)
		self.dc_relu_1 =  	nn.ReLU()
		self.dc_drop_1 = 	nn.Dropout()
		self.dc_ip_2 = 		nn.Linear(1024, 1024)
		self.dc_relu_2 = 	nn.ReLU()
		self.dc_drop_2 = 	nn.Dropout()
		self.dc_ip_3 = 		nn.Linear(1024, 1)
		# label resize layer??
		# uses sigmoid cross entropy loss: BCELossWithLogits()

	def forward(self, inst_lvl_features, beta):
		x = GRL.apply(inst_lvl_features, beta)
		x = self.dc_ip_1(x)
		x = self.dc_relu_1(x)
		x = self.dc_drop_1(x)
		x = self.dc_ip_2(x)
		x = self.dc_relu_2(x)
		x = self.dc_ip_3(x)
		x = x.view(-1)
		return x

class domain_consistency_regularizer(nn.Module):
	def __init__(self):
		super(domain_consistency_regularizer, self).__init__()
		self.sigmoid_d_probs = nn.Sigmoid()
		
	def forward(self, logits):
		logits = self.sigmoid_d_probs(logits)
		return logits

