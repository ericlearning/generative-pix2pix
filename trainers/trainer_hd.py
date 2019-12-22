import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from utils import *
from losses.losses import *
from itertools import chain

class Trainer_HD():
	def __init__(self, loss_type, netDs, netG, device, train_dl, val_dl, lr_D = 0.0002, lr_G = 0.0002, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 50, image_interval = 50, save_img_dir = 'saved_images/'):
		self.loss_type, self.device = loss_type, device
		self.require_type = get_require_type(self.loss_type)
		self.loss = get_gan_loss(self.device, self.loss_type)

		self.netDs = netDs
		self.netG = netG
		self.train_dl = train_dl
		self.val_dl = val_dl
		self.lr_D = lr_D
		self.lr_G = lr_G
		self.train_iteration_per_epoch = len(self.train_dl)
		self.device = device
		self.resample = resample
		self.weight_clip = weight_clip
		self.use_gradient_penalty = use_gradient_penalty
		self.special = None

		self.optimizerD = optim.Adam(list(self.netDs[0].parameters())+list(self.netDs[1].parameters())+list(self.netDs[2].parameters()), lr = self.lr_D, betas = (0, 0.9))
		self.optimizerG = optim.Adam(self.netG.parameters(), lr = self.lr_G, betas = (0, 0.9))

		self.real_label = 1
		self.fake_label = 0

		self.loss_interval = loss_interval
		self.image_interval = image_interval

		self.errD_records = []
		self.errG_records = []

		self.save_cnt = 0
		self.save_img_dir = save_img_dir
		if(not os.path.exists(self.save_img_dir)):
			os.makedirs(self.save_img_dir)

	def gradient_penalty(self, x, real_image, fake_image):
		NotImplementedError
		# // TOTO : one day ... 

	def resize(self, x, scales):
		return [F.adaptive_avg_pool2d(x, (x.shape[2] // s, x.shape[3] // s)) if(s != 1) else x for s in scales]

	def resize_input(self, stage, x, y, fake_y):
		m = 1 if(stage) else 2
		scales = [2**s*m for s in range(len(self.netDs))]
		xs = self.resize(x, scales)
		ys = self.resize(y, scales)
		fake_ys = self.resize(fake_y, scales)

		return xs, ys, fake_ys

	def train(self, num_epochs):
		for stage, num_epoch in enumerate(num_epochs):
			for epoch in range(num_epoch):
				if(self.resample):
					train_dl_iter = iter(self.train_dl)
				for i, (x, y) in enumerate(tqdm(self.train_dl)):
					x = x.to(self.device)
					y = y.to(self.device)
					bs = x.size(0)
					fake_y = self.netG(x, stage)
					xs, ys, fake_ys = self.resize_input(stage, x, y, fake_y)

					self.netDs.zero_grad()

					# calculate the discriminator results for both real & fake
					c_xrs, c_xfs = [], []
					for netD_i, x_i, y_i, fake_y_i in zip(self.netDs, xs, ys, fake_ys):
						c_xr = netD_i(x_i, y_i).view(-1)
						c_xf = netD_i(x_i, fake_y_i.detach()).view(-1)
						c_xrs.append(c_xr)
						c_xfs.append(c_xf)

					errDs = nn.ModuleList([])
					if(self.require_type == 0 or self.require_type == 1):
						errDs = [self.loss.d_loss(c_xr_i, c_xf_i) for c_xr_i, c_xf_i in zip(c_xrs, c_xfs)]
					elif(self.require_type == 2):
						errDs = [self.loss.d_loss(c_xr_i, c_xf_i, y_i, fake_y_i)\
								 for c_xr_i, c_xf_i, y_i, fake_y_i in zip(c_xrs, c_xfs, ys, fake_ys)]

					if(self.use_gradient_penalty != False):
						NotImplementedError

					errD = torch.mean(errDs)
					errD.backward()

					# update D using the gradients calculated previously
					self.optimizerD.step()

					#if(self.weight_clip != None):
					#	for param in self.netDs.parameters():
					#		param.data.clamp_(-self.weight_clip, self.weight_clip)

					self.netG.zero_grad()
					if(self.resample):
						x, y = next(train_dl_iter)
						x = x.to(self.device)
						y = y.to(self.device)
						fake_y = self.netG(x, stage)
						xs, ys, fake_ys = self.resize_input(stage, x, y, fake_y)

					# calculate the discriminator results for both real & fake
					c_xrs, c_xfs = nn.ModuleList([]), nn.ModuleList([])
					features_a, features_b = nn.ModuleList([]), nn.ModuleList([])
					for netD_i, x_i, y_i, fake_y_i in zip(self.netDs, xs, ys, fake_ys):
						c_xr_i, feature_i_a = self.netD(x_i, y_i, return_feature = True).view(-1)
						c_xf_i, feature_i_b = self.netD(x_i, fake_y_i, return_feature = True).view(-1)
						c_xrs.append(c_xr_i)
						c_xfs.append(c_xf_i)
						features_a.append(feature_i_a)
						features_b.append(feature_i_b)

					errGs_a, errGs_b = nn.ModuleList([]), nn.ModuleList([])
					if(self.require_type == 0):
						errGs_a = [self.loss.g_loss(c_xf_i) for c_xf_i in c_xfs]
					if(self.require_type == 1 or self.require_type == 2):
						errGs_a = [self.loss.g_loss(c_xr_i, c_xf_i) for c_xr_i, c_xf_i in zip(c_xrs, c_xfs)]

					errG_a = torch.mean(errGs_a)

					errG_bs = nn.ModuleList([])
					for feature_i_a, feature_i_b in zip(features_a, features_b):
						errG_b_i = 0
						for f1, f2 in zip(feature_i_a, feature_i_b):
							errG_b_i += (f1 - f2).abs().mean()
						errG_b_i /= len(feature_i_a)
						errG_bs.append(errG_b_i)
					errG_b = torch.mean(errG_bs)

					errG = errG_a + errG_b * 10
					errG.backward()
					#update G using the gradients calculated previously
					self.optimizerG.step()

					self.errD_records.append(float(errD))
					self.errG_records.append(float(errG))

					if(i % self.loss_interval == 0):
						print('[%d/%d] [%d/%d] errD : %.4f, errG : %.4f'
							  %(epoch+1, num_epoch, i+1, self.train_iteration_per_epoch, errD, errG))

					if(i % self.image_interval == 0):
						if(self.special == None):
							sample_images_list = get_sample_images_list('Pix2pixHD_Normal', (self.val_dl, self.netG, stage, self.device))
							plot_img = get_display_samples(sample_images_list, 3, 3)
							cur_file_name = os.path.join(self.save_img_dir, str(self.save_cnt)+' : '+str(epoch)+'-'+str(i)+'.jpg')
							self.save_cnt += 1
							cv2.imwrite(cur_file_name, plot_img)
