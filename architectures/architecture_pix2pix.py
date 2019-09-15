import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm

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

		if(self.use_sn):
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

# ResNet Generator
class ResNet_G_256x256(nn.Module):
	def __init__(self, ic, oc, use_bn = True, use_sn = False, norm_type = 'instancenorm'):
		super(ResNet_G_256x256, self).__init__()
		self.ic = ic
		self.oc = oc

		self.relu = nn.ReLU(inplace = True)

		self.conv = ConvBlock(ic, 64, 7, 1, 3, use_bn = False, use_sn = use_sn, activation_type = None, pad_type = 'Reflection')
		self.conv_block1 = ConvBlock(64, 128, 3, 2, 1, use_bn = use_bn, use_sn = use_sn, norm_type = norm_type)
		self.conv_block2 = ConvBlock(128, 256, 3, 2, 1, use_bn = use_bn, use_sn = use_sn, norm_type = norm_type)

		self.resblock1 = ResBlock(256, 256, use_bn, use_sn, norm_type)
		self.resblock2 = ResBlock(256, 256, use_bn, use_sn, norm_type)
		self.resblock3 = ResBlock(256, 256, use_bn, use_sn, norm_type)
		self.resblock4 = ResBlock(256, 256, use_bn, use_sn, norm_type)
		self.resblock5 = ResBlock(256, 256, use_bn, use_sn, norm_type)
		self.resblock6 = ResBlock(256, 256, use_bn, use_sn, norm_type)

		self.deconv_block1 = DeConvBlock(256, 128, 3, 2, 1, output_pad = 1, use_bn = use_bn, use_sn = use_sn, norm_type = norm_type)
		self.deconv_block2 = DeConvBlock(128, 64, 3, 2, 1, output_pad = 1, use_bn = use_bn, use_sn = use_sn, norm_type = norm_type)
		self.deconv = ConvBlock(64, oc, 7, 1, 3, use_bn = False, use_sn = use_sn, activation_type = None, pad_type = 'Reflection')

		self.tanh = nn.Tanh()

		for m in self.modules():
			if(isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x):
		# (bs, ic, sz, sz)
		out = self.conv(out)
		# (bs, 64, sz, sz)
		out = self.conv_block1(out)
		# (bs, 128, sz / 2, sz / 2)
		out = self.conv_block2(out)
		# (bs, 256, sz / 4, sz / 4)
		out = self.resblock1(out)
		out = self.resblock2(out)
		out = self.resblock3(out)
		out = self.resblock4(out)
		out = self.resblock5(out)
		out = self.resblock6(out)
		# (bs, 256, sz / 4, sz / 4)
		out = self.deconv_block1(out)
		# (bs, 128, sz / 2, sz / 2)
		out = self.deconv_block2(out)
		# (bs, 64, sz, sz)
		out = self.deconv(out)
		# (bs, oc, sz, sz)
		out = self.tanh(out)
		# (bs, oc, sz, sz)
		return out

class PatchGan_D_70x70(nn.Module):
	def __init__(self, ic_1, ic_2, use_sigmoid = True, use_bn = True, use_sn = False, norm_type = 'batchnorm'):
		super(PatchGan_D_70x70, self).__init__()
		self.ic_1 = ic_1
		self.ic_2 = ic_2
		self.use_sigmoid = use_sigmoid
		self.conv1 = ConvBlock(self.ic_1 + self.ic_2, 64, 4, 2, 1, use_bn = False, use_sn = use_sn, activation_type = 'leakyrelu')
		self.conv2 = ConvBlock(64, 128, 4, 2, 1, use_bn = use_bn, use_sn = use_sn, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv3 = ConvBlock(128, 256, 4, 2, 1, use_bn = use_bn, use_sn = use_sn, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv4 = ConvBlock(256, 512, 4, 1, 1, use_bn = use_bn, use_sn = use_sn, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv5 = ConvBlock(512, 1, 4, 1, 1, use_bn = False, use_sn = use_sn, activation_type = None)
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


class PatchGan_D_70x70_One_Input(nn.Module):
	def __init__(self, ic, use_sigmoid = True, use_bn = True, use_sn = False, norm_type = 'batchnorm'):
		super(PatchGan_D_70x70_One_Input, self).__init__()
		self.ic = ic
		self.use_sigmoid = use_sigmoid
		self.conv1 = ConvBlock(self.ic, 64, 4, 2, 1, use_bn = False, use_sn = use_sn, activation_type = 'leakyrelu')
		self.conv2 = ConvBlock(64, 128, 4, 2, 1, use_bn = use_bn, use_sn = use_sn, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv3 = ConvBlock(128, 256, 4, 2, 1, use_bn = use_bn, use_sn = use_sn, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv4 = ConvBlock(256, 512, 4, 1, 1, use_bn = use_bn, use_sn = use_sn, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv5 = ConvBlock(512, 1, 4, 1, 1, use_bn = False, use_sn = use_sn, activation_type = None)
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
	def __init__(self, ic_1, ic_2, use_sigmoid = True, use_bn = True, use_sn = False, norm_type = 'batchnorm'):
		super(PatchGan_D_286x286, self).__init__()
		self.ic_1 = ic_1
		self.ic_2 = ic_2
		self.use_sigmoid = use_sigmoid
		self.conv1 = ConvBlock(self.ic_1 + self.ic_2, 64, 4, 2, 1, use_bn = False, use_sn = use_sn, activation_type = 'leakyrelu')
		self.conv2 = ConvBlock(64, 128, 4, 2, 1, use_bn = use_bn, use_sn = use_sn, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv3 = ConvBlock(128, 256, 4, 2, 1, use_bn = use_bn, use_sn = use_sn, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv4 = ConvBlock(256, 512, 4, 1, 1, use_bn = use_bn, use_sn = use_sn, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv5 = ConvBlock(512, 512, 4, 1, 1, use_bn = use_bn, use_sn = use_sn, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv6 = ConvBlock(512, 512, 4, 1, 1, use_bn = use_bn, use_sn = use_sn, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv7 = ConvBlock(512, 1, 4, 1, 1, use_bn = False, use_sn = use_sn, activation_type = None)
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
	def __init__(self, ic, use_sigmoid = True, use_bn = True, use_sn = False, norm_type = 'batchnorm'):
		super(PatchGan_D_286x286_One_Input, self).__init__()
		self.ic = ic
		self.use_sigmoid = use_sigmoid
		self.conv1 = ConvBlock(self.ic, 64, 4, 2, 1, use_bn = False, use_sn = use_sn, activation_type = 'leakyrelu')
		self.conv2 = ConvBlock(64, 128, 4, 2, 1, use_bn = use_bn, use_sn = use_sn, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv3 = ConvBlock(128, 256, 4, 2, 1, use_bn = use_bn, use_sn = use_sn, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv4 = ConvBlock(256, 512, 4, 1, 1, use_bn = use_bn, use_sn = use_sn, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv5 = ConvBlock(512, 512, 4, 1, 1, use_bn = use_bn, use_sn = use_sn, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv6 = ConvBlock(512, 512, 4, 1, 1, use_bn = use_bn, use_sn = use_sn, norm_type = norm_type, activation_type = 'leakyrelu')
		self.conv7 = ConvBlock(512, 1, 4, 1, 1, use_bn = False, use_sn = use_sn, activation_type = None)
		self.sigmoid = nn.Sigmoid()
		self.nothing = Nothing()

		for m in self.modules():
			if(isinstance(m, nn.Conv2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x1, x2, return_feature = False):
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

class UNet_G(nn.Module):
	def __init__(self, ic, oc, sz, use_f = True, use_bn = True, use_sn = False, norm_type = 'batchnorm', dropout_num = 3):
		super(UNet_G, self).__init__()
		self.ic = ic
		self.oc = oc
		self.use_f = use_f
		self.sz = sz
		self.dims = {
			'16' : [64, 128, 256, 512],
			'32' : [64, 128, 256, 512, 512],
			'64' : [64, 128, 256, 512, 512, 512],
			'128' : [64, 128, 256, 512, 512, 512, 512],
			'256' : [64, 128, 256, 512, 512, 512, 512, 512],
			'512' : [64, 128, 256, 512, 512, 512, 512, 512, 512]
		}
		self.cur_dim = self.dims[str(sz)]
		self.num_convs = len(self.cur_dim)
		self.dropout_num = dropout_num

		self.leaky_relu = nn.LeakyReLU(0.2, inplace = True)
		self.relu = nn.ReLU(inplace = True)

		self.enc_convs = nn.ModuleList([])
		cur_block_ic = self.ic
		for i, dim in enumerate(self.cur_dim):
			if(i == 0 or i == len(self.cur_dim) - 1):
				self.enc_convs.append(ConvBlock(cur_block_ic, dim, 4, 2, 1, use_bn = False, use_sn = use_sn, activation_type = None))
			else:
				self.enc_convs.append(ConvBlock(cur_block_ic, dim, 4, 2, 1, use_bn = use_bn, use_sn = use_sn, norm_type = norm_type, activation_type = None))
			cur_block_ic = dim

		self.dec_convs = nn.ModuleList([])
		cur_block_ic = self.cur_dim[-1]
		for i, dim in enumerate(list(reversed(self.cur_dim))[1:] + [self.oc]):
			if(i == 0):
				self.dec_convs.append(DeConvBlock(cur_block_ic, dim, 4, 2, 1, use_bn = False, use_sn = use_sn, activation_type = None))
			elif(i == len(self.cur_dim) - 1):
				self.dec_convs.append(DeConvBlock(cur_block_ic*2, self.oc, 4, 2, 1, use_bn = False, use_sn = use_sn, activation_type = None))
			else:
				self.dec_convs.append(DeConvBlock(cur_block_ic*2, dim, 4, 2, 1, use_bn = use_bn, use_sn = use_sn, norm_type = norm_type, activation_type = None))
			cur_block_ic = dim

		self.tanh = nn.Tanh()
		self.dropout = nn.Dropout()

		for m in self.modules():
			if(isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()
	
	def forward(self, x):
		ens = []
		for i, cur_enc in enumerate(self.enc_convs):
			if(i == 0):
				out = cur_enc(x)
			else:
				out = cur_enc(self.leaky_relu(out))
			ens.append(out)

		for i, cur_dec in enumerate(self.dec_convs):
			cur_enc = ens[self.num_convs - 1 - i]
			if(i < self.dropout_num):
				if(self.use_f):
					if(i == 0):
						out = F.dropout(cur_dec(self.relu(cur_enc)))
					else:
						out = F.dropout(cur_dec(self.relu(torch.cat([out, cur_enc], 1))))
				else:
					if(i == 0):
						out = self.dropout(cur_dec(self.relu(cur_enc)))
					else:
						out = self.dropout(cur_dec(self.relu(torch.cat([out, cur_enc], 1))))
			else:
				if(i == 0):
					out = cur_dec(self.relu(cur_enc))
				else:
					out = cur_dec(self.relu(torch.cat([out, cur_enc], 1)))
		del ens
		out = self.tanh(out)
		return out
