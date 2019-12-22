import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
import torchvision.models as models

class Nothing(nn.Module):
	def __init__(self):
		super(Nothing, self).__init__()
		
	def forward(self, x):
		return x

def get_norm(norm_type, size):
	if(norm_type == 'batchnorm'):
		return nn.BatchNorm2d(size)
	elif(norm_type == 'instancenorm'):
		return nn.InstanceNorm2d(size)

def get_activation(activation_type):
	if(activation_type == 'relu'):
		return nn.ReLU(inplace = True)
	elif(activation_type == 'leakyrelu'):
		return nn.LeakyReLU(0.2, inplace = True)
	elif(activation_type == 'elu'):
		return nn.ELU(inplace = True)
	elif(activation_type == 'selu'):
		return nn.SELU(inplace = True)
	elif(activation_type == 'prelu'):
		return nn.PReLU()
	elif(activation_type == 'tanh'):
		return nn.Tanh()
	elif(activation_type == None):
		return Nothing()

class ConvBlock(nn.Module):
	def __init__(self, ni, no, ks, stride, pad = None, use_bn = True, use_sn = False, use_pixelshuffle = False, norm_type = 'batchnorm', activation_type = 'leakyrelu', pad_type = 'Zero'):
		super(ConvBlock, self).__init__()
		self.use_bn = use_bn
		self.use_sn = use_sn
		self.use_pixelshuffle = use_pixelshuffle
		self.norm_type = norm_type
		self.pad_type = pad_type

		if(pad == None):
			pad = ks // 2 // stride

		ni_ = ni
		if(use_pixelshuffle):
			self.pixelshuffle = nn.PixelShuffle(2)
			ni_ = ni // 4
		
		if(pad_type == 'Zero'):
			self.conv = nn.Conv2d(ni_, no, ks, stride, pad, bias = False)
		else:
			self.conv = nn.Sequential(*[
				nn.ReflectionPad2d(pad),
				nn.Conv2d(ni_, no, ks, stride, 0, bias = False)
			])

		if(self.use_bn):
			self.bn = get_norm(norm_type, no)
		if(self.use_sn):
			self.conv = SpectralNorm(self.conv)

		self.act = get_activation(activation_type)

	def forward(self, x):
		out = x
		if(self.use_pixelshuffle):
			out = self.pixelshuffle(out)
		out = self.conv(out)
		if(self.use_bn):
			out = self.bn(out)
		out = self.act(out)
		return out

class DeConvBlock(nn.Module):
	def __init__(self, ni, no, ks, stride, pad = None, output_pad = 0, use_bn = True, use_sn = False, norm_type = 'batchnorm', activation_type = 'leakyrelu', pad_type = 'Zero'):
		super(DeConvBlock, self).__init__()
		self.use_bn = use_bn
		self.use_sn = use_sn
		self.norm_type = norm_type
		self.pad_type = pad_type

		if(pad is None):
			pad = ks // 2 // stride

		if(pad_type == 'Zero'):
			self.deconv = nn.ConvTranspose2d(ni, no, ks, stride, pad, output_padding = output_pad, bias = False)
		else:
			self.deconv = nn.Sequential(*[
				nn.ReflectionPad2d(pad),
				nn.ConvTranspose2d(ni, no, ks, stride, 0, output_padding = output_pad, bias = False)
			])

		if(self.use_bn):
			self.bn = get_norm(norm_type, no)
		if(self.use_sn):
			self.deconv = SpectralNorm(self.deconv)

		self.act = get_activation(activation_type)

	def forward(self, x):
		out = self.deconv(x)
		if(self.use_bn):
			out = self.bn(out)
		out = self.act(out)
		return out

# Residual Block
class ResBlock(nn.Module):
	def __init__(self, ic, oc, use_bn = True, use_sn = False, norm_type = 'instancenorm'):
		super(ResBlock, self).__init__()
		self.ic = ic
		self.oc = oc
		self.norm_type = norm_type

		self.relu = nn.ReLU(inplace = True)
		self.reflection_pad1 = nn.ReflectionPad2d(1)
		self.reflection_pad2 = nn.ReflectionPad2d(1)

		self.conv1 = nn.Conv2d(ic, oc, 3, 1, 0, bias = False)
		self.conv2 = nn.Conv2d(oc, oc, 3, 1, 0, bias = False)

		if(use_sn):
			self.conv1 = SpectralNorm(self.conv1)
			self.conv2 = SpectralNorm(self.conv2)

		if(use_bn):
			if(self.norm_type == 'batchnorm'):
				self.bn1 = nn.BatchNorm2d(oc)
				self.bn2 = nn.BatchNorm2d(oc)

			elif(self.norm_type == 'instancenorm'):
				self.bn1 = nn.InstanceNorm2d(oc)
				self.bn2 = nn.InstanceNorm2d(oc)

		else:
			self.bn1 = Nothing()
			self.bn2 = Nothing()

	def forward(self, x):
		out = self.reflection_pad1(x)
		out = self.relu(self.bn1(self.conv1(out)))
		out = self.reflection_pad2(out)
		out = self.bn2(self.conv2(out))
		out = out + x
		return out

class PatchGan_D_70x70(nn.Module):
	def __init__(self, ic_1, ic_2, use_sigmoid = True, norm_type = 'batchnorm', return_feature = False):
		super(PatchGan_D_70x70, self).__init__()
		self.ic_1 = ic_1
		self.ic_2 = ic_2
		self.use_sigmoid = use_sigmoid
		self.conv1 = ConvBlock(self.ic_1 + self.ic_2, 64, 4, 2, 1, use_bn = False, activation_type = 'leakyrelu')
		self.conv2 = ConvBlock(64, 128, 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv3 = ConvBlock(128, 256, 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv4 = ConvBlock(256, 512, 4, 1, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv5 = nn.Conv2d(512, 1, 4, 1, 1, bias = False)
		self.sigmoid = nn.Sigmoid()
		self.nothing = Nothing()

		for m in self.modules():
			if(isinstance(m, nn.Conv2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x1, x2, return_feature = False):
		out = torch.cat([x1, x2], 1)
		# (bs, ic_1+ic_2, 256, 256)
		out1 = self.conv1(out)
		# (bs, 64, 128, 128)
		out2 = self.conv2(out1)
		# (bs, 128, 64, 64)
		out3 = self.conv3(out2)
		# (bs, 256, 32, 32)
		out4 = self.conv4(out3)
		# (bs, 512, 31, 31)
		out5 = self.conv5(out4)
		# (bs, 1, 30, 30)
		if(self.use_sigmoid == True):
			out = self.sigmoid(out5)
		else:
			out = self.nothing(out5)

		if(return_feature):
			return out, [out1, out2, out3, out4, out5]
		else:
			return out


# PatchGan 256x256 - OneInput
class PatchGan_D_70x70_One_Input(nn.Module):
	def __init__(self, ic, use_sigmoid = True, norm_type = 'batchnorm'):
		super(PatchGan_D_70x70_One_Input, self).__init__()
		self.ic = ic
		self.use_sigmoid = use_sigmoid
		self.conv1 = ConvBlock(self.ic, 64, 4, 2, 1, use_bn = False, activation_type = 'leakyrelu')
		self.conv2 = ConvBlock(64, 128, 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv3 = ConvBlock(128, 256, 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv4 = ConvBlock(256, 512, 4, 1, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv5 = nn.Conv2d(512, 1, 4, 1, 1, bias = False)
		self.sigmoid = nn.Sigmoid()
		self.nothing = Nothing()

		for m in self.modules():
			if(isinstance(m, nn.Conv2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x, return_feature = False):
		out = x
		# (bs, ic, 256, 256)
		out1 = self.conv1(out)
		# (bs, 64, 128, 128)
		out2 = self.conv2(out1)
		# (bs, 128, 64, 64)
		out3 = self.conv3(out2)
		# (bs, 256, 32, 32)
		out4 = self.conv4(out3)
		# (bs, 512, 31, 31)
		out5 = self.conv5(out4)
		# (bs, 1, 30, 30)
		if(self.use_sigmoid == True):
			out = self.sigmoid(out5)
		else:
			out = self.nothing(out5)

		if(return_feature):
			return out, [out1, out2, out3, out4, out5]
		else:
			return out



class PatchGan_D_286x286(nn.Module):
	def __init__(self, ic_1, ic_2, use_sigmoid = True, norm_type = 'batchnorm'):
		super(PatchGan_D_286x286, self).__init__()
		self.ic_1 = ic_1
		self.ic_2 = ic_2
		self.use_sigmoid = use_sigmoid
		self.conv1 = ConvBlock(self.ic_1 + self.ic_2, 64, 4, 2, 1, use_bn = False, activation_type = 'leakyrelu')
		self.conv2 = ConvBlock(64, 128, 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv3 = ConvBlock(128, 256, 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv4 = ConvBlock(256, 512, 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv5 = ConvBlock(512, 512, 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv6 = ConvBlock(512, 512, 4, 1, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv7 = nn.Conv2d(512, 1, 4, 1, 1, bias = False)
		self.sigmoid = nn.Sigmoid()
		self.nothing = Nothing()

		for m in self.modules():
			if(isinstance(m, nn.Conv2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x1, x2, return_feature = False):
		out = torch.cat([x1, x2], 1)
		# (bs, ic_1+ic_2, 256, 256)
		out1 = self.conv1(out)
		# (bs, 64, 128, 128)
		out2 = self.conv2(out1)
		# (bs, 128, 64, 64)
		out3 = self.conv3(out2)
		# (bs, 256, 32, 32)
		out4 = self.conv4(out3)
		# (bs, 256, 16, 16)
		out5 = self.conv5(out4)
		# (bs, 256, 8, 8)
		out6 = self.conv6(out5)
		# (bs, 512, 7, 7)
		out7 = self.conv7(out6)
		# (bs, 1, 6, 6)
		if(self.use_sigmoid == True):
			out = self.sigmoid(out7)
		else:
			out = self.nothing(out7)

		if(return_feature):
			return out, [out1, out2, out3, out4, out5, out6, out7]
		else:
			return out


class PatchGan_D_286x286_One_Input(nn.Module):
	def __init__(self, ic, use_sigmoid = True, norm_type = 'batchnorm'):
		super(PatchGan_D_286x286_One_Input, self).__init__()
		self.ic = ic
		self.use_sigmoid = use_sigmoid
		self.conv1 = ConvBlock(self.ic, 64, 4, 2, 1, use_bn = False, activation_type = 'leakyrelu')
		self.conv2 = ConvBlock(64, 128, 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv3 = ConvBlock(128, 256, 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv4 = ConvBlock(256, 512, 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv5 = ConvBlock(512, 512, 4, 2, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv6 = ConvBlock(512, 512, 4, 1, 1, use_bn = True, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv7 = nn.Conv2d(512, 1, 4, 1, 1, bias = False)
		self.sigmoid = nn.Sigmoid()
		self.nothing = Nothing()

		for m in self.modules():
			if(isinstance(m, nn.Conv2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x, return_feature = False):
		out = x
		# (bs, ic, 256, 256)
		out1 = self.conv1(out)
		# (bs, 64, 128, 128)
		out2 = self.conv2(out1)
		# (bs, 128, 64, 64)
		out3 = self.conv3(out2)
		# (bs, 256, 32, 32)
		out4 = self.conv4(out3)
		# (bs, 256, 16, 16)
		out5 = self.conv5(out4)
		# (bs, 256, 8, 8)
		out6 = self.conv6(out5)
		# (bs, 512, 7, 7)
		out7 = self.conv7(out6)
		# (bs, 1, 6, 6)
		if(self.use_sigmoid == True):
			out = self.sigmoid(out7)
		else:
			out = self.nothing(out7)

		if(return_feature):
			return out, [out1, out2, out3, out4, out5, out6, out7]
		else:
			return out
		
def receptive_calculator(input_size, ks, stride, pad):
	return int((input_size - ks + 2 * pad) / stride + 1)

def inverse_receptive_calculator(output_size, ks, stride, pad):
	return ((output_size - 1) * stride) + ks

class Pix2PixHD_G(nn.Module):
	def __init__(self, ic, oc):
		super(Pix2PixHD_G, self).__init__()
		self.ic = ic
		self.oc = oc

		self.global_G = Global(ic, oc)
		self.local_G1 = Local(ic, oc, 0)
		self.local_G2 = Local(ic, oc, 1)

		self.cur_stage = -1
		for m in self.modules():
			if(isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x, stage = -1):
		# 0:global / 1:local / 2:finetune / -1:normal pass
		if(stage == 0):
			if(self.cur_stage != stage):
				self.freeze(self.local_G1, True)
				self.freeze(self.local_G2, True)
				self.freeze(self.global_G, False)
				self.cur_stage = stage

			x_half = F.adaptive_avg_pool2d(x, (x.shape[2] // 2, x.shape[3] // 2))
			out = self.global_G(x_half, use_last_conv = True)
			# outputs half of desired resolution

		elif(stage == 1):
			if(self.cur_stage != stage):
				self.freeze(self.local_G1, False)
				self.freeze(self.local_G2, False)
				self.freeze(self.global_G, True)
				self.cur_stage = stage

			x_half = F.adaptive_avg_pool2d(x, (x.shape[2] // 2, x.shape[3] // 2))
			out1 = self.local_G1(x)
			out2 = self.global_G(x_half, use_last_conv = False)
			out = self.local_G2(out1 + out2)
			# outputs full desired resolution

		elif(stage == 2):
			if(self.cur_stage != stage):
				self.freeze(self.local_G1, False)
				self.freeze(self.local_G2, False)
				self.freeze(self.global_G, False)
				self.cur_stage = stage

			x_half = F.adaptive_avg_pool2d(x, (x.shape[2] // 2, x.shape[3] // 2))
			out1 = self.local_G1(x)
			out2 = self.global_G(x_half, use_last_conv = False)
			out = self.local_G2(out1 + out2)
			# outputs full desired resolution

		elif(stage == -1):
			x_half = F.adaptive_avg_pool2d(x, (x.shape[2] // 2, x.shape[3] // 2))
			out1 = self.local_G1(x)
			out2 = self.global_G(x_half, use_last_conv = False)
			out = self.local_G2(out1 + out2)
			# outputs full desired resolution

		return out

	def freeze(self, model, choice):
		for child in model.children():
			for param in child.parameters():
				param.requires_grad = not choice


class Global(nn.Module):
	def __init__(self, ic, oc):
		super(Global, self).__init__()
		self.ic = ic
		self.oc = oc

		self.conv1 = ConvBlock(ic, 64, 7, 1, 3, pad_type = 'Reflection', use_bn = False, activation_type = None)
		self.blocks = nn.Sequential(
			ConvBlock(64, 128, 3, 2, 1, pad_type = 'Reflection', use_bn = True, norm_type = 'instancenorm', activation_type = 'relu'),
			ConvBlock(128, 256, 3, 2, 1, pad_type = 'Reflection', use_bn = True, norm_type = 'instancenorm', activation_type = 'relu'),
			ConvBlock(256, 512, 3, 2, 1, pad_type = 'Reflection', use_bn = True, norm_type = 'instancenorm', activation_type = 'relu'),
			ConvBlock(512, 1024, 3, 2, 1, pad_type = 'Reflection', use_bn = True, norm_type = 'instancenorm', activation_type = 'relu')
		)
		self.res = [ResBlock(1024, 1024)] * 9
		self.res = nn.Sequential(*self.res)

		self.blocks2 = nn.Sequential(
			DeConvBlock(1024, 512, 3, 2, 1, output_pad = 1, use_bn = True, norm_type = 'instancenorm', activation_type = 'relu'),
			DeConvBlock(512, 256, 3, 2, 1, output_pad = 1, use_bn = True, norm_type = 'instancenorm', activation_type = 'relu'),
			DeConvBlock(256, 128, 3, 2, 1, output_pad = 1, use_bn = True, norm_type = 'instancenorm', activation_type = 'relu'),
			DeConvBlock(128, 64, 3, 2, 1, output_pad = 1, use_bn = True, norm_type = 'instancenorm', activation_type = 'relu'),
		)
		self.conv2 = ConvBlock(64, oc, 7, 1, 3, pad_type = 'Reflection', use_bn = False, activation_type = None)
		self.tanh = nn.Tanh()

	def forward(self, x, use_last_conv = True):
		out = self.conv1(x)
		out = self.blocks(out)
		out = self.res(out)
		out = self.blocks2(out)
		if(use_last_conv):
			out = self.conv2(out)
			out = self.tanh(out)
		return out

class Local(nn.Module):
	def __init__(self, ic, oc, part):
		super(Local, self).__init__()
		self.ic = ic
		self.oc = oc
		self.part = part

		if(self.part == 0):
			self.conv = ConvBlock(ic, 32, 7, 1, 3, pad_type = 'Reflection', use_bn = False, activation_type = None)
			self.block = ConvBlock(32, 64, 3, 2, 1, pad_type = 'Reflection', use_bn = True, norm_type = 'instancenorm', activation_type = 'relu')
		elif(self.part == 1):
			self.res = [ResBlock(64, 64)] * 3
			self.res = nn.Sequential(*self.res)
			self.block = DeConvBlock(64, 32, 3, 2, 1, output_pad = 1, use_bn = True, norm_type = 'instancenorm', activation_type = 'relu')
			self.conv = ConvBlock(32, oc, 7, 1, 3, pad_type = 'Reflection', use_bn = False, activation_type = None)
			self.tanh = nn.Tanh()

	def forward(self, x):
		if(self.part == 0):
			out = self.conv(x)
			out = self.block(out)
		elif(self.part == 1):
			out = self.res(x)
			out = self.block(out)
			out = self.conv(out)
			out = self.tanh(out)

		return out

class VGG():
	def __init__(self):
		super(VGG, self).__init__()
		self.f = model.vgg19(pretrained = True).features
		self.split = [0, 2, 7, 12, 21, 30]

	def forward(self, x):
		outs = []
		out = x
		for i in range(len(self.split) - 1):
			out = self.f[self.split[i]:self.split[i+1]](x)
			outs.append(out)
		return outs
