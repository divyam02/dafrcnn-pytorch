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
def consistency_reg(N, d_image_y, d_inst_y, domain):
  y = d_image_y.sum(dim=0)
  L_cst = 0
  if domain=='src':
    r = 0
    f = 1
  else:
    r = 1
    f = 1
    #f = 15

  y = y[r]
  #size = list(d_inst_y.size())[0]
  size = min(list(d_inst_y.size())[0], 128)
  for i in range(size):
  #for i in range(size/f)
    # L_cst += torch.norm((y/N - d_inst_y[f*i][r]), p=2)
    L_cst += torch.norm((y/N - d_inst_y[i][r]), p=2)
  #print(i)
  return L_cst

def flatten(x):
    N = list(x.size())[0]
    #print('dim 0', N, 1024*19*37)
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
    return (grad_output*(-1*self.beta))

# base_feat dim: 1 x 1024 x 38 x 75, atleast for city.
# EDIT - take LogSoftmax of 1 x 1024*w*h
# Taking feat_map output as label score.

class d_cls_image(nn.Module):
  def __init__(self, beta=1, ch_in=1024, ch_out=1024, W=38, H=75, stride_1=1, padding_1=1, kernel=3):
    super(d_cls_image, self).__init__()
    self.conv_image = nn.Conv2d(ch_in, ch_out, stride=stride_1, padding=padding_1, kernel_size=kernel)
    self.bn_image = nn.BatchNorm2d(ch_out)
    self.fc_1_image = nn.Linear(1, 2)
    self.ch_out = ch_out
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=2)
    self.bn_2 = nn.BatchNorm1d(1024)
    #self.softmax = nn.Softmax()
    #self.logsoftmax = nn.LogSoftmax()
    self.beta = beta

  def forward(self, x):
    x = grad_reverse(x, self.beta)
    x = self.conv_image(x)
    x = self.relu(x)
    x = self.bn_image(x)
    x = self.maxpool(x)
    x = self.bn_2(x)
    # convert to 1024*W*H x 1.
    x = flatten(x)
    x = torch.transpose(x, 0, 1)
    x = self.fc_1_image(x)
    # 1 x n vector
    #y = self.softmax(x)
    #x = self.logsoftmax(x)
    #return x, y
    return x

  def set_beta(self, beta):
    self.beta = beta

# pool_feat dim: N x 2048, where N may be 300.

class d_cls_inst(nn.Module):
  def __init__(self, beta=1, fc_size=2048):
    super(d_cls_inst, self).__init__()
    self.fc_1_inst = nn.Linear(fc_size, 100)
    self.fc_2_inst = nn.Linear(100, 2)
    self.relu = nn.ReLU(inplace=True)
    self.beta = beta
    #self.softmax = nn.Softmax()
    #self.logsoftmax = nn.LogSoftmax()
    self.bn = nn.BatchNorm1d(2)

  def forward(self, x):
    x = grad_reverse(x, self.beta)
    x = self.relu(self.fc_1_inst(x))
    x = self.relu(self.bn(self.fc_2_inst(x)))
    #y = self.softmax(x)
    #x = self.logsoftmax(x)
    #return x, y
    return x

  def set_beta(self, beta):
    self.beta = beta
