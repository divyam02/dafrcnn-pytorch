# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

from domain_adapt import *
import math

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  #########################################################
  """
  Add target dataset...
  """
  parser.add_argument('--target_dataset', dest='target_dataset',
                      help='target dataset', default='fcity',
                      type=str)
  #########################################################
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of workers to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether to perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is iterations',
                      default=50000, type=int)
  """
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  """
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)
  parser.add_argument('--stop_iter', dest='stop_iter',
                      help='Iteration at which training stops',
                      default=70000, type=int)
  
# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and display
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

  args = parser.parse_args()
  return args

class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "city":
      args.imdb_name = "city_2007_trainval"
      args.imdbval_name = "city_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "fcity":
      args.imdb_name = "fcity_2007_trainval"
      args.imdbval_name = "fcity_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "kitti-voc":
      args.imdb_name = "kitti-voc_2007_trainval"
      args.imdbval_name = "kitti-voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "caltech-voc":
      args.imdb_name = "caltech-voc_2007_trainval"
      args.imdbval_name = "caltech-voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal-voc":
      args.imdb_name = "pascal-voc-2007_2007_trainval+pascal-voc-2012_2012_trainval"
      args.imdbval_name = "..."
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "sim10k":
      args.imdb_name = "sim10k_2012_trainval"
      args.imdbval_name = "sim10k_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']      
  elif args.dataset == "species":
      args.imdb_name = "species_2007_trainval"
      args.imdbval_name = "sim10k_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

  """
  Get target dataset names
  """
  if args.target_dataset == "fcity":
      args.target_imdb_name = "fcity_2007_trainval"
      args.target_imdbval_name = "fcity_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.target_dataset == "city":
      args.target_imdb_name = "city_2007_trainval"
      args.target_imdbval_name = "city_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.target_dataset == "kitti-voc":
      args.target_imdb_name = "kitti-voc_2007_trainval"
      args.target_imdbval_name = "kitti-voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.target_dataset == "caltech-voc":
      args.target_imdb_name = "caltech-voc_2007_test"
      args.target_imdbval_name = ""
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.target_dataset == "clipart":
      args.target_imdb_name = "clipart_2007_trainval"
      args.target_imdbval_name = "clipart_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.target_dataset == "comic":
      args.target_imdb_name = "comic_2007_trainval"
      args.target_imdbval_name = "comic_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.target_dataset == "watercolor":
      args.target_imdb_name = "watercolor_2007_trainval"
      args.target_imdbval_name = "watercolor_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.target_dataset == "species_2018":
      args.target_imdb_name = "species_2018_2007_trainval"
      args.target_imdbval_name = "watercolor_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  train_size = len(roidb)


  """
  Get imdb, roidb for target dataset...
  """
  print("getting target values...\n")
  target_imdb, target_roidb, target_ratio_list, target_ratio_index = combined_roidb(args.target_imdb_name, False)
  target_size = len(target_roidb)


  print('{:d} roidb entries'.format(len(roidb)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  sampler_batch = sampler(train_size, args.batch_size)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

###########################################################
  """
  Build target dataset, dataloaders...
  @Note; Fix num classes for target dataset. Must be binary (0,1).
  """
  target_sampler_batch = sampler(target_size, args.batch_size)

  target_dataset = roibatchLoader(target_roidb, target_ratio_list, target_ratio_index,
                                  args.batch_size, target_imdb.classes, training=False, normalize=True)

  target_dataloader = torch.utils.data.DataLoader(target_dataset, batch_size=args.batch_size,
                                    sampler=target_sampler_batch, num_workers=args.num_workers)
###########################################################

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)


  """
  @Add target img holders...
  """
  target_im_data = torch.FloatTensor(1)
  target_im_info = torch.FloatTensor(1)
  target_num_boxes = torch.LongTensor(1)
  target_gt_boxes = torch.FloatTensor(1)

  if args.cuda:
    target_im_data = target_im_data.cuda()
    target_im_info = target_im_info.cuda()
    target_num_boxes = target_num_boxes.cuda()
    target_gt_boxes = target_num_boxes.cuda()

  target_im_data = Variable(target_im_data)
  target_im_info = Variable(target_im_info)
  target_num_boxes = Variable(target_num_boxes)
  target_gt_boxes = Variable(target_gt_boxes)


  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum


  """
  @Add domain adaptation components.
  """
  img_domain_classifier = domain_img_cls(args.net)
  inst_domain_classifier = domain_inst_cls(args.net)
  img_domain_classifier.cuda()
  inst_domain_classifier.cuda()


  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.cuda:
    fasterRCNN.cuda()


  """
  @Add optimizers for the same.
  """
  img_domain_optimizer = torch.optim.SGD(img_domain_classifier.parameters(), lr=lr, momentum=cfg.TRAIN.MOMENTUM)
  inst_domain_optimizer = torch.optim.SGD(inst_domain_classifier.parameters(), lr=lr, momentum=cfg.TRAIN.MOMENTUM)


  if args.resume:
    load_name = os.path.join(output_dir,
      'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    img_domain_classifier.load_state_dict(checkpoint['img_domain_classifier'])
    inst_domain_classifier.load_state_dict(checkpoint['inst_domain_classifier'])
    img_domain_optimizer.load_state_dict(checkpoint['img_domain_optimizer'])
    inst_domain_optimizer.load_state_dict(checkpoint['inst_domain_optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

###########################################################
  """
  @Add resuming capability for domain classifiers.
  """
###########################################################

  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  iters_per_epoch = int(train_size / args.batch_size)

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")

  total_steps = 0
  total_train_size = args.max_epochs * train_size

  target_data_iter = iter(target_dataloader)

  for epoch in range(args.start_epoch, args.max_epochs + 1):
    # setting to train mode
    fasterRCNN.train()
    loss_temp = 0
    start = time.time()


    """
    @Add classifiers to training. Adjust lr with decay
    """
    img_domain_classifier.train()
    inst_domain_classifier.train()
    print(img_domain_classifier)
    print(inst_domain_classifier)
    """
    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        adjust_learning_rate(img_domain_optimizer, args.lr_decay_gamma)
        adjust_learning_rate(inst_domain_optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma
    """
    data_iter = iter(dataloader)

    for step in range(iters_per_epoch):
      
      data = next(data_iter)

      if total_steps==args.lr_decay_step:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        adjust_learning_rate(img_domain_optimizer, args.lr_decay_gamma)
        adjust_learning_rate(inst_domain_optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma 

      """
      @Add iter for target dataset...
      """
      total_steps+=1
      if total_steps%target_size==0:
          target_data_iter = iter(target_dataloader)


      target_data = next(target_data_iter)


      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])


      """
      @Initialize target img holders...
      """
      target_im_data.data.resize_(target_data[0].size()).copy_(target_data[0])
      target_im_info.data.resize_(target_data[1].size()).copy_(target_data[1])
      target_gt_boxes.data.resize_(target_data[2].size()).copy_(target_data[2])
      target_num_boxes.data.resize_(target_data[3].size()).copy_(target_data[3])


      fasterRCNN.zero_grad()

      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label, base_feat, pooled_feat = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)


      """
      @Add target values...
      """
      target_base_feat, target_pooled_feat = fasterRCNN(target_im_data, target_im_info, target_gt_boxes, target_num_boxes, target=True)


      """
      @Add domain classifier losses and outputs.
      domain_classifiers.zero_grad()
      source_loss_1 = dc_1(base_feat, pooled_feat)
      target_loss_1 = dc_1(target_base_feat, target_pooled_feat)
      """
      img_domain_classifier.zero_grad()
      inst_domain_classifier.zero_grad()

      """
      @Set up beta hyperparameter for GRL...
      """
      p = total_steps / total_train_size
      beta = (2./(1.+np.exp(-10 * p))) - 1.


      img_feat = img_domain_classifier(base_feat, beta)
      target_img_feat = img_domain_classifier(target_base_feat, beta)
      inst_feat = inst_domain_classifier(pooled_feat, beta)
      target_inst_feat = inst_domain_classifier(target_pooled_feat, beta)

      src_img_loss = domain_loss(img_feat, 0) 
      tar_img_loss = domain_loss(target_img_feat, 1)
      src_inst_loss = domain_loss(inst_feat, 0)
      tar_inst_loss = domain_loss(target_inst_feat, 1)


      """
      @Add consistency loss...
      """
      src_consistency_loss = consistency_loss(img_feat, inst_feat)
      tar_consistency_loss = consistency_loss(target_img_feat, target_inst_feat)


      loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
           + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() \
           + 0.1*(src_img_loss.mean() + src_inst_loss.mean()) \
           + 0.1*(tar_img_loss.mean() + tar_inst_loss.mean()) \
           + 0.1*(src_consistency_loss.mean() + tar_consistency_loss.mean())
      loss_temp += loss.item()

      # backward
      optimizer.zero_grad()

      """
      @Add classifier optimizers
      """
      img_domain_optimizer.zero_grad()
      inst_domain_optimizer.zero_grad()

      loss.backward()
      if args.net == "vgg16":
          clip_gradient(fasterRCNN, 10.)
      optimizer.step()

      """
      @Add optimizer update
      """
      img_domain_optimizer.step()
      inst_domain_optimizer.step()

      """
      Debug statements...
      """
      if math.isnan(rpn_loss_cls) or math.isnan(RCNN_loss_cls) or math.isnan(RCNN_loss_bbox) or math.isnan(rpn_loss_box):
        print(im_data, im_info, gt_boxes, num_boxes)
        assert 1<0, print("encountered nan!", rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox)

      if rpn_loss_box>5 or rpn_loss_cls>5 or RCNN_loss_cls>5 or RCNN_loss_bbox>5 or src_img_loss>5 or src_inst_loss>5 or tar_img_loss>5 or tar_inst_loss>5:
        print("WARNING: unstable losses!")
        print(im_data, im_info, gt_boxes, num_boxes)
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, RCNN_cls: %.4f, RCNN_box %.4f" \
                      % (rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox))
        print("\t\t\tsrc_img_loss: %.4f, src_inst_loss: %.4f, src_consistency_loss: %.4f" \
                                % (src_img_loss, src_inst_loss, src_consistency_loss))
        print("\t\t\ttar_img_loss: %.4f, tar_inst_loss: %.4f, tar_consistency_loss: %.4f" \
                                % (tar_img_loss, tar_inst_loss, tar_consistency_loss))

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (args.disp_interval + 1)

        if args.mGPUs:
          loss_rpn_cls = rpn_loss_cls.mean().item()
          loss_rpn_box = rpn_loss_box.mean().item()
          loss_rcnn_cls = RCNN_loss_cls.mean().item()
          loss_rcnn_box = RCNN_loss_bbox.mean().item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
          src_img_loss = src_img_loss.mean().item()
          src_inst_loss = src_inst_loss.mean().item()
          src_consistency_loss = src_consistency_loss.mean().item()
          tar_img_loss = tar_img_loss.mean().item()
          tar_inst_loss = tar_inst_loss.mean().item()
          tar_consistency_loss = tar_consistency_loss.mean().item()
        else:
          loss_rpn_cls = rpn_loss_cls.item()
          loss_rpn_box = rpn_loss_box.item()
          loss_rcnn_cls = RCNN_loss_cls.item()
          loss_rcnn_box = RCNN_loss_bbox.item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
          src_img_loss = src_img_loss.item()
          src_inst_loss = src_inst_loss.item()
          src_consistency_loss = src_consistency_loss.item()
          tar_img_loss = tar_img_loss.item()
          tar_inst_loss = tar_inst_loss.item()
          tar_consistency_loss = tar_consistency_loss.item()

        print("[session %d][epoch %2d][iter %4d/%4d][steps %4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, total_steps,loss_temp, lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
        print("\t\t\tsrc_img_loss: %.4f, src_inst_loss: %.4f, src_consistency_loss: %.4f" \
                                % (src_img_loss, src_inst_loss, src_consistency_loss))
        print("\t\t\ttar_img_loss: %.4f, tar_inst_loss: %.4f, tar_consistency_loss: %.4f" \
                                % (tar_img_loss, tar_inst_loss, tar_consistency_loss))
        #print("source outputs:", img_feat, inst_feat)
        #print("target outputs:", target_img_feat, target_inst_feat)
        print()
        if args.use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_box': loss_rcnn_box
          }
          logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

        loss_temp = 0
        start = time.time()
        if total_steps>=args.stop_iter:
          break
    
    save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
    save_checkpoint({
      'session': args.session,
      'epoch': epoch + 1,
      'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
      'optimizer': optimizer.state_dict(),
      'pooling_mode': cfg.POOLING_MODE,
      'class_agnostic': args.class_agnostic,
      'img_domain_classifier': img_domain_classifier.state_dict(),
      'inst_domain_classifier': inst_domain_classifier.state_dict(),
      'img_domain_optimizer': img_domain_optimizer.state_dict(),
      'inst_domain_optimizer': inst_domain_optimizer.state_dict(),
    }, save_name)
    print('save model: {}'.format(save_name))
    if total_steps>=args.stop_iter:
      break

  if args.use_tfboard:
    logger.close()
