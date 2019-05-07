import os
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from utils import set_lr, get_lr, generate_noise, plot_multiple_images, save_fig, save, lab_to_rgb, get_sample_images_list

class Trainer_LSGAN_Pix2Pix():
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

		self.optimizerD = optim.RMSprop(self.netD.parameters(), lr = self.lr_D)
		self.optimizerG = optim.RMSprop(self.netG.parameters(), lr = self.lr_G)

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

	def train(self, num_epoch):
		l1 = nn.L1Loss()
		for epoch in range(num_epoch):
			if(self.resample):
				train_dl_iter = iter(self.train_dl)
			for i, (x, y) in enumerate(tqdm(self.train_dl)):
				x = x.to(self.device)
				y = y.to(self.device)
				bs = x.size(0)
				fake_y = self.netG(x)

				self.netD.zero_grad()

				# calculate the discriminator results for both real & fake

				c_xr = self.netD(x, y)				# (bs, 1, 1, 1)
				c_xr = c_xr.view(-1)						# (bs)
				c_xf = self.netD(x, fake_y.detach())		# (bs, 1, 1, 1)
				c_xf = c_xf.view(-1)						# (bs)

				# calculate the discriminator loss
				errD_real = torch.mean((c_xr - torch.ones(c_xr.size()).to(self.device)) ** 2)
				errD_fake = torch.mean((c_xf - torch.zeros(c_xf.size()).to(self.device)) ** 2)
				errD = errD_real + errD_fake
				errD.backward()
				# update D using the gradients calculated previously
				self.optimizerD.step()

				# -log(D(G(x), y)) + L1(G(x), y)
				self.netG.zero_grad()
				if(self.resample):
					x, y = next(train_dl_iter)
					x = x.to(self.device)
					y = y.to(self.device)
					fake_y = self.netG(x)
				# we updated the discriminator once, therefore recalculate c_xf
				c_xf = self.netD(x, fake_y)		# (bs, 1, 1, 1)
				c_xf = c_xf.view(-1)						# (bs)
				# calculate the Generator loss
				errG_1 = torch.mean((c_xf - torch.ones(c_xf.size()).to(self.device)) ** 2)
				errG_2 = l1(fake_y, y)
				lambd = 100
				errG = errG_1 + errG_2 * lambd
				errG.backward()
				#update G using the gradients calculated previously
				self.optimizerG.step()

				self.errD_records.append(float(errD))
				self.errG_records.append(float(errG))

				if(i % self.loss_interval == 0):
					print('[%d/%d] [%d/%d] errD : %.4f, errG : %.4f'
						  %(epoch+1, num_epoch, i+1, self.train_iteration_per_epoch, errD, errG))

				if(i % self.image_interval == 0):
					if(self.special == 'Colorization'):
						sample_images_list = get_sample_images_list('Pix2pix_Colorization', (self.val_dl, self.netG, self.device))
						plot_fig = plot_multiple_images(sample_images_list, 2, 3)
						cur_file_name = os.path.join(self.save_img_dir, str(self.save_cnt)+' : '+str(epoch)+'-'+str(i)+'.jpg')
						self.save_cnt += 1
						save_fig(cur_file_name, plot_fig)
						plot_fig.clf()

					else:
						sample_images_list = get_sample_images_list('Pix2pix_Normal', (self.val_dl, self.netG, self.device))
						plot_fig = plot_multiple_images(sample_images_list, 3, 3)
						cur_file_name = os.path.join(self.save_img_dir, str(self.save_cnt)+' : '+str(epoch)+'-'+str(i)+'.jpg')
						self.save_cnt += 1
						save_fig(cur_file_name, plot_fig)
						plot_fig.clf()


				if(self.snapshot_interval is not None):
					if(i % self.snapshot_interval == 0):
						save(os.path.join(self.save_snapshot_dir, 'Epoch' + str(epoch) + '_' + str(i) + '.state'), self.netD, self.netG, self.optimizerD, self.optimizerG)