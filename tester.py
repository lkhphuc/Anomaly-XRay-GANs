
import os
import time
import torch
import datetime
import itertools

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from sagan_models import Generator, Discriminator, Encoder
from utils import *

import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class Tester(object):
    def __init__(self, data_loader, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.data_loader = data_loader

        # exact model and loss
        self.model = config.model
        self.adv_loss = config.adv_loss

        # Model hyper-parameters
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel

        self.lambda_gp = config.lambda_gp
        self.total_step = config.total_step
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.ge_lr = config.ge_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.dataset = config.dataset
        self.mura_class = config.mura_class
        self.mura_type = config.mura_type
        self.use_tensorboard = config.use_tensorboard
        self.image_path = config.image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        # Build tensorboard for debugiing
        self.build_tensorboard()

        # Build model
        self.build_model()

        # Load models
        self.load_pretrained_model()

    def test(self):
        data_iter = iter(self.data_loader)
        self.D.eval()
        self.E.eval()
        self.G.eval()

        with torch.no_grad():
            for i, data in enumerate(data_iter):

                val_images, val_labels = data
                val_images = tensor2var(val_images)
                
                # Run val images through models X -> E(X) -> G(E(X))
                z, ef1, ef2 = self.E(val_images)
                re_images, gf1, gf2 = self.G(z)
                
                dv, dv5, dv4, dv3, dvz, dva2, dva1 = self.D(val_images, z)
                dr, dr5, dr4, dr3, drz, dra2, dra1 = self.D(re_images, z)

                # Compute residual loss 
                l1 = (re_images - val_images).abs()
                l2 = (re_images- val_images).pow(2).sqrt()
                # Computer feature matching loss
                ld = (dv - dr).abs().view((self.batch_size, -1)).mean(dim=1)
                ld5 = (dv5 - dr5).abs().view((self.batch_size, -1)).mean(dim=1)
                ld4 = (dv4 - dr4).abs().view((self.batch_size, -1)).mean(dim=1)
                ld3 = (dv3 - dr3).abs().view((self.batch_size, -1)).mean(dim=1)

                import ipdb; ipdb.set_trace()
                
                plt.scatter(range(1, self.batch_size+1), l1, c=val_labels)
                
                

    def build_tensorboard(self):
        '''Initialize tensorboard writer'''
        self.writer = SummaryWriter(self.log_path)

    def build_model(self):
        self.G = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).to(self.device)
        self.E = Encoder(self.batch_size, self.imsize, self.z_dim, self.d_conv_dim).to(self.device)
        self.D = Discriminator(self.batch_size,self.imsize, self.z_dim, self.d_conv_dim).to(self.device)
        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.E = nn.DataParallel(self.E)
            self.D = nn.DataParallel(self.D)

        # Loss and optimizer
        self.ge_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
            itertools.chain(self.G.parameters(), self.E.parameters())), self.ge_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.c_loss = torch.nn.CrossEntropyLoss()
        # print networks
        print(self.G)
        print(self.E)
        print(self.D)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.E.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_E.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.ge_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
