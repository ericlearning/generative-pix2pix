import os
import torch
import torch.nn as nn
from torchvision import transforms
from dataset import Dataset
from architectures.architecture_pix2pix import UNet_G, ResNet_G_256x256, PatchGan_D_70x70, PatchGan_D_286x286
from architectures.architecture_pix2pixhd import Pix2PixHD_G
from trainers.trainer import Trainer
from trainers_hd.trainer_hd import Trainer_HD
from utils import save, load

train_dir_name = ['data/file/train/input', 'data/file/train/target']
val_dir_name = ['data/file/val/input', 'data/file/val/target']

lr_D, lr_G, bs = 0.0002, 0.0002, 128
sz, ic, oc, use_sigmoid = 256, 3, 3, False
norm_type = 'instancenorm'

dt = {
	'input' : transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.ToTensor(),
		transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
	]),
	'target' : transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.ToTensor(),
		transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
	])
}

train_data = Dataset(train_dir_name, basic_types = 'Pix2Pix', shuffle = True)
val_data = Dataset(val_dir_name, basic_types = 'Pix2Pix', shuffle = False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
netD = PatchGan_D_70x70(ic, oc, use_sigmoid, norm_type).to(device)
netG = UNet_G(ic, oc, sz, True, norm_type, dropout_num = 3).to(device)
# netG = Pix2PixHD_G(ic, oc).to(device)

trn_dl = train_data.get_loader(256, bs, data_transform = dt)
val_dl = list(val_data.get_loader(256, 3, data_transform = dt))[0]

trainer = Trainer('SGAN', netD, netG, device, trn_dl, val_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('LSGAN', netD, netG, device, trn_dl, val_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('HINGEGAN', netD, netG, device, trn_dl, val_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('WGAN', netD, netG, device, trn_dl, val_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = 0.01, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('WGAN', netD, netG, device, trn_dl, val_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = 10, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('RASGAN', netD, netG, device, trn_dl, val_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('RALSGAN', netD, netG, device, trn_dl, val_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('RAHINGEGAN', netD, netG, device, trn_dl, val_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('QPGAN', netD, netG, device, trn_dl, val_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')

# trainer = Trainer_HD('SGAN', netD, netG, device, trn_dl, val_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
# trainer = Trainer_HD('LSGAN', netD, netG, device, trn_dl, val_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
# trainer = Trainer_HD('HINGEGAN', netD, netG, device, trn_dl, val_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
# trainer = Trainer_HD('WGAN', netD, netG, device, trn_dl, val_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = 0.01, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
# trainer = Trainer_HD('WGAN', netD, netG, device, trn_dl, val_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = 10, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
# trainer = Trainer_HD('RASGAN', netD, netG, device, trn_dl, val_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
# trainer = Trainer_HD('RALSGAN', netD, netG, device, trn_dl, val_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
# trainer = Trainer_HD('RAHINGEGAN', netD, netG, device, trn_dl, val_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
# trainer = Trainer_HD('QPGAN', netD, netG, device, trn_dl, val_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')

trainer.train(5)
# trainer.train([5, 5, 5])
save('saved/cur_state.state', netD, netG, trainer.optimizerD, trainer.optimizerG)