import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
# Mini-batch train loop to take sum over every image
# Take target for cross entropy loss as torch.ones_like(x)
# take softmax scores before calculating loss
# ADD LOGSOFTMAX(x) FOR CONSISTENCY REGEX
# take in d_inst_y as n x 2 where n is the number of roi samples for the given image in train loop
# in consistency_reg, N is image/feature map dimensions.
def consistency_reg(N, d_image_y, d_inst_y):	
    y = d_image_y.sum(dim=0)
    L_cst = 0
    l2dist = nn.PairwiseDist()
    for i in range(d_inst_y.shape[0]):
		L_cst += l2dist((y/N - d_inst_y[i][1]))
    return L_cst
    
def flatten(x):
    N = x.shape[0]
    return x.view(N, -1)

def grad_reverse(x, beta):
    return GradReverse(beta)(x)

class GradReverse(Function):    
	def __init__(self, beta):
		self.beta = beta

	def set_beta(self, beta):
		self.beta = beta

    def forward(self, x):
	return x.view_as(x)

    def backward(self, grad_output):
	return (grad_output*(-self.beta))

# base_feat dim: 1 x 1024 x 38 x 75, atleast for city.
# EDIT - take LogSoftmax of 1 x 1024*w*h
# Taking feat_map output as label score.

class d_cls_image(nn.Module):
    def __init__(self, ch_in=1024, ch_out=1024, W=38, H=75, stride=1, padding=1, kernel=3, beta=1):
    	super(d_cls_image, self).__init__()
	self.conv_image = nn.Conv2d(ch_in, ch_out, stride=stride, padding=padding, kernel_size=kernel)
	self.bn_image = nn.BatchNorm2d(ch_out)
	self.fc_1_image = nn.Linear(1, 2)
	self.beta = beta
	#self.fc_1_image = nn.Linear(ch_out*W*H, 100)
	#self.fc_2_image = nn.Linear(100, 2)	

    def forward(self, x):
	x = grad_reverse(x, self.beta)
	x = nn.bn_image(F.ReLU(self.conv_image(x)))
	x = nn.MaxPool2d(x, (2, 2))
	x = nn.BatchNorm1d(flatten(x))
	# convert to 1024*W*H x 1.
	x = torch.transpose(x, 0, 1)
	x = self.fc_1_image(x)
	#x = self.fc_2_image(x)
	# 1 x lala vector
	y = nn.Softmax(x)
	x = nn.LogSoftmax(x) 
	return x, y

# pool_feat dim: N x 2048, where N may be 300.

class d_cls_inst(nn.Module):
    def __init__(self, fc_size=2048, beta=1):
	super(d_cls_instance, self).__init__()
	self.fc_1_inst = nn.Linear(fc_size, 100)
	self.fc_2_inst = nn.Linear(100, 2)
	self.beta = beta

    def forward(self, x):
	x = grad_reverse(x, self.beta)
	x = F.ReLU(self.fc_1_inst(x))
	x = F.ReLU(nn.BatchNorm1d(self.fc_2_inst(x)))
	y = nn.Softmax(x)
	x = nn.LogSoftmax(x)
	return x, y
