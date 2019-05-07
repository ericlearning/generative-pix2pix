import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from utils import set_lr, get_lr, generate_noise, plot_multiple_images, save_fig, save, lab_to_rgb, get_sample_images_list

class Trainer_RALSGAN_Pix2PixHD():
	def __init__(self, netD, netG, device, train_dl, val_dl, lr_D = 0.0002, lr_G = 0.0002, loss_interval = 50, image_interval = 50, snapshot_interval = None, save_img_dir = 'saved_images/', save_snapshot_dir = 'saved_snapshots', resample = False):
		self.netD = netD
		self.netG = netG
		self.train_dl = train_dl
		self.val_dl = val_dl
		self.lr_D = lr_D
		self.lr_G = lr_G
		self.train_iteration_per_epoch = len(self.train_dl)
		self.device = device
		self.resample = resample
		self.special = None

		self.optimizerD = optim.Adam(self.netD.parameters(), lr = self.lr_D, betas = (0, 0.9))
		self.optimizerG = optim.Adam(self.netG.parameters(), lr = self.lr_G, betas = (0, 0.9))

		self.real_label = 1
		self.fake_label = 0

		self.loss_interval = loss_interval
		self.image_interval = image_interval
		self.snapshot_interval = snapshot_interval

		self.errD_records = []
		self.errG_records = []

		self.save_cnt = 0
		self.save_img_dir = save_img_dir
		self.save_snapshot_dir = save_snapshot_dir
		if(not os.path.exists(self.save_img_dir)):
			os.makedirs(self.save_img_dir)
		if(not os.path.exists(self.save_snapshot_dir)):
			os.makedirs(self.save_snapshot_dir)

	def resize_input(self, stage, x, y, fake_y):
		if(stage == 0):
			x1 = F.adaptive_avg_pool2d(x, (x.shape[2] // 2, x.shape[3] // 2))								# (sz/2, sz/2)
			x2 = F.adaptive_avg_pool2d(x, (x.shape[2] // 4, x.shape[3] // 4))								# (sz/4, sz/4)
			x3 = F.adaptive_avg_pool2d(x, (x.shape[3] // 8, x.shape[3] // 8))								# (sz/8, sz/8)

			y1 = F.adaptive_avg_pool2d(y, (y.shape[2] // 2, y.shape[3] // 2))								# (sz/2, sz/2)
			y2 = F.adaptive_avg_pool2d(y, (y.shape[2] // 4, y.shape[3] // 4))								# (sz/4, sz/4)
			y3 = F.adaptive_avg_pool2d(y, (y.shape[2] // 8, y.shape[3] // 8))								# (sz/8, sz/8)

			fake_y_1 = fake_y 																				# (sz/2, sz/2)
			fake_y_2 = F.adaptive_avg_pool2d(fake_y, (fake_y.shape[2] // 2, fake_y.shape[3] // 2))			# (sz/4, sz/4)
			fake_y_3 = F.adaptive_avg_pool2d(fake_y, (fake_y.shape[2] // 4, fake_y.shape[3] // 4))			# (sz/8, sz/8)

		else:
			x1 = x																							# (sz, sz)
			x2 = F.adaptive_avg_pool2d(x, (x.shape[2] // 2, x.shape[3] // 2))								# (sz/2, sz/2)
			x3 = F.adaptive_avg_pool2d(x, (x.shape[3] // 4, x.shape[3] // 4))								# (sz/4, sz/4)

			y1 = y																							# (sz, sz)
			y2 = F.adaptive_avg_pool2d(y, (y.shape[2] // 2, y.shape[3] // 2))								# (sz/2, sz/2)
			y3 = F.adaptive_avg_pool2d(y, (y.shape[2] // 4, y.shape[3] // 4))								# (sz/4, sz/4)

			fake_y_1 = fake_y 																				# (sz, sz)
			fake_y_2 = F.adaptive_avg_pool2d(fake_y, (fake_y.shape[2] // 2, fake_y.shape[3] // 2))			# (sz/2, sz/2)
			fake_y_3 = F.adaptive_avg_pool2d(fake_y, (fake_y.shape[2] // 4, fake_y.shape[3] // 4))			# (sz/4, sz/4)

		return x1, x2, x3, y1, y2, y3, fake_y_1, fake_y_2, fake_y_3

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
					x1, x2, x3, y1, y2, y3, fake_y_1, fake_y_2, fake_y_3 = self.resize_input(stage, x, y, fake_y)

					self.netD.zero_grad()
					
					# calculate the discriminator results for both real & fake
					c_xr_1 = self.netD(x1, y1)
					c_xr_1 = c_xr_1.view(-1)
					c_xf_1 = self.netD(x1, fake_y_1.detach())
					c_xf_1 = c_xf_1.view(-1)

					c_xr_2 = self.netD(x2, y2)
					c_xr_2 = c_xr_2.view(-1)
					c_xf_2 = self.netD(x2, fake_y_2.detach())
					c_xf_2 = c_xf_2.view(-1)

					c_xr_3 = self.netD(x3, y3)
					c_xr_3 = c_xr_3.view(-1)
					c_xf_3 = self.netD(x3, fake_y_3.detach())
					c_xf_3 = c_xf_3.view(-1)
					
					# calculate the discriminator loss
					real_label_1 = torch.ones(c_xr_1.size()).to(self.device)
					real_label_2 = torch.ones(c_xr_2.size()).to(self.device)
					real_label_3 = torch.ones(c_xr_3.size()).to(self.device)
					errD_1 = (torch.mean((c_xr_1 - torch.mean(c_xf_1) - real_label_1)**2) + torch.mean((c_xf_1 - torch.mean(c_xr_1) + real_label_1)**2)) / 2.0
					errD_2 = (torch.mean((c_xr_2 - torch.mean(c_xf_2) - real_label_2)**2) + torch.mean((c_xf_2 - torch.mean(c_xr_2) + real_label_2)**2)) / 2.0
					errD_3 = (torch.mean((c_xr_3 - torch.mean(c_xf_3) - real_label_3)**2) + torch.mean((c_xf_3 - torch.mean(c_xr_3) + real_label_3)**2)) / 2.0
					errD = errD_1 + errD_2 + errD_3
					errD.backward()
					# update D using the gradients calculated previously
					self.optimizerD.step()

					# -log(D(G(x), y)) + L1(G(x), y)
					self.netG.zero_grad()
					if(self.resample):
						x, y = next(train_dl_iter)
						x = x.to(self.device)
						y = y.to(self.device)
						fake_y = self.netG(x, stage)
						x1, x2, x3, y1, y2, y3, fake_y_1, fake_y_2, fake_y_3 = self.resize_input(stage, x, y, fake_y)

					# calculate the discriminator results for both real & fake
					c_xr_1, feature_1_a = self.netD(x1, y1, return_feature = True)
					c_xr_1 = c_xr_1.view(-1)
					c_xf_1, feature_1_b = self.netD(x1, fake_y_1, return_feature = True)
					c_xf_1 = c_xf_1.view(-1)

					c_xr_2, feature_2_a = self.netD(x2, y2, return_feature = True)
					c_xr_2 = c_xr_2.view(-1)
					c_xf_2, feature_2_b = self.netD(x2, fake_y_2, return_feature = True)
					c_xf_2 = c_xf_2.view(-1)

					c_xr_3, feature_3_a = self.netD(x3, y3, return_feature = True)
					c_xr_3 = c_xr_3.view(-1)
					c_xf_3, feature_3_b = self.netD(x3, fake_y_3, return_feature = True)
					c_xf_3 = c_xf_3.view(-1)

					# calculate the Generator loss
					errG_a_1 = (torch.mean((c_xf_1 - torch.mean(c_xr_1) - real_label_1)**2) + torch.mean((c_xr_1 - torch.mean(c_xf_1) + real_label_1)**2)) / 2.0
					errG_a_2 = (torch.mean((c_xf_2 - torch.mean(c_xr_2) - real_label_2)**2) + torch.mean((c_xr_2 - torch.mean(c_xf_2) + real_label_2)**2)) / 2.0
					errG_a_3 = (torch.mean((c_xf_3 - torch.mean(c_xr_3) - real_label_3)**2) + torch.mean((c_xr_3 - torch.mean(c_xf_3) + real_label_3)**2)) / 2.0
					errG_a = errG_a_1 + errG_a_2 + errG_a_3

					errG_b_1, errG_b_2, errG_b_3 = 0, 0, 0
					for f1, f2 in zip(feature_1_a, feature_1_b):
						errG_b_1 += (f1 - f2).abs().mean()
					errG_b_1 /= len(feature_1_a)
					for f1, f2 in zip(feature_2_a, feature_2_b):
						errG_b_2 += (f1 - f2).abs().mean()
					errG_b_2 /= len(feature_2_a)
					for f1, f2 in zip(feature_3_a, feature_3_b):
						errG_b_3 += (f1 - f2).abs().mean()
					errG_b_3 /= len(feature_3_a)
					errG_b = 10 * (errG_b_1 + errG_b_2 + errG_b_3)

					errG = errG_a + errG_b
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
							plot_fig = plot_multiple_images(sample_images_list, 3, 3)
							cur_file_name = os.path.join(self.save_img_dir, str(self.save_cnt)+' : '+str(epoch)+'-'+str(i)+'.jpg')
							self.save_cnt += 1
							save_fig(cur_file_name, plot_fig)
							plot_fig.clf()

					if(self.snapshot_interval is not None):
						if(i % self.snapshot_interval == 0):
							save(os.path.join(self.save_snapshot_dir, 'Stage' + str(stage) + 'Epoch' + str(epoch) + '_' + str(i) + '.state'), self.netD, self.netG, self.optimizerD, self.optimizerG)
