import torch
import torch.nn as nn
from torch.autograd import Function, Variable
import numpy as np

class GRL(Function):
	def __init__(self, beta=1):
		self.beta = beta

	def forward(self, x):
		return x.view_as(x)

	def backward(self, grad_output):
		output = grad_output*(-1)*self.beta
		return output

def grad_reverse(x, beta=1):
	return GRL(beta)(x)


class domain_img_cls(nn.Module):
	def __init__(self, net):
		super(domain_img_cls, self).__init__()
		if net=="res101":
			in_channels = 1024
		else:
			in_channels = 512
		self.conv_1= nn.Conv2d(in_channels=in_channels, out_channels=512, kernel_size=1, padding=0, stride=1)
		self.relu = nn.ReLU()
		self.conv_2 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, padding=0, stride=1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x, beta=1):
		x = grad_reverse(x, beta)
		x = self.conv_1(x)
		x = self.relu(x)
		x = self.conv_2(x)
		x = self.sigmoid(x)
		x = x.view(-1)

		return x


class domain_inst_cls(nn.Module):
	def __init__(self, net):
		super(domain_inst_cls, self).__init__()
		if net=="res101":
			in_channels = 2048
		else:
			in_channels = 4096
		self.fc_1 = nn.Linear(in_channels, 1024)
		self.fc_2 = nn.Linear(1024, 1024)
		self.fc_3 = nn.Linear(1024, 1)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x, beta=1):
		x = grad_reverse(x, beta)
		x = self.fc_1(x)
		x = self.relu(x)
		x = self.dropout(x)

		x = self.fc_2(x)
		x = self.relu(x)
		x = self.dropout(x)

		x = self.fc_3(x)
		x = self.sigmoid(x)
		x = x.view(-1)

		return x


def domain_loss(logits, labels):
	#print('domain loss:', logits.size())
	if labels==0:
		# For source
		labels = torch.from_numpy(np.zeros(list(logits.size())[0])).float().cuda()
	else:
		# For target
		labels = torch.from_numpy(np.ones(list(logits.size())[0])).float().cuda()

	loss = nn.BCELoss()
	return loss(logits, labels)


def consistency_loss(source_logits, target_logits):
	target = torch.from_numpy(np.zeros(list(target_logits.size())[0])).float().cuda()
	source_logits = torch.sum(source_logits)/list(source_logits.size())[0]
	source_logits = torch.ones(target_logits.size()).cuda() * source_logits
	#source_logits = source_logits.view(1, list(source_logits.size())[0])
	#target_logits = target_logits.view(1, list(target_logits.size())[0])
	loss = nn.L1Loss()

	return loss(source_logits - target_logits, target)