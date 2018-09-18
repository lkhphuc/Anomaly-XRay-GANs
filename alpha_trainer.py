
import os
import time
import torch
import datetime
import itertools

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from alphagan_models import Generator, Discriminator, Encoder, Codecriminator
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def log(x):
    return torch.log(x + 1e-10)

class alpha_Trainer(object):
    def __init__(self, data_loader, config):
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

        if self.use_tensorboard:
            self.build_tensorboard()

        self.build_model()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def train(self):
        '''Training loop'''

        # Data iterator
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)
        model_save_step = int(self.model_save_step * step_per_epoch)

        # Fixed input for debugging
        fixed_img, _ = next(data_iter)
        fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))
        if self.use_tensorboard:
            self.writer.add_image('img/fixed_img', denorm(fixed_img.data), 0)
        else:
            save_image(denorm(fixed_img.data),
                        os.path.join(self.sample_path, 'fixed_img.png'))

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        self.D.train()
        self.E.train()
        self.G.train()
        self.C.train()

        # Start time
        start_time = time.time()
        for step in range(start, self.total_step):
            # Sample from data and prior
            try:
                real_images, _ = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                real_images, _ = next(data_iter)

            # Create labels for training
            real_labels = tensor2var(torch.ones((real_images.size(0))))
            fake_labels = tensor2var(torch.zeros((real_images.size(0))))

            x_real = tensor2var(real_images)
            # Sample from condition
            z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            
            # Encoder
            z_mean, z_logvar, _, _ = self.E(x_real)
            z_hat = z_mean + z_logvar * torch.randn(z_mean.size()).cuda()
            # Decoder (generator)
            x_rec, x_rec4, x_rec3 = self.G(z_hat)
            x_gen, x_gen4, x_gen3 = self.G(z)

            # Discriminator
            d_real, d_real4, d_real3  = self.D(x_real)
            d_rec, d_rec4, d_rec3 = self.D(x_rec)
            d_gen, d_gen4, d_gen3 = self.D(x_gen)

            # Codecriminator
            c_z_hat = self.C(z_hat)
            c_z = self.C(z)

            # # ================== Train G and E ================== #
            # self.reset_grad()
            # l1_loss = 0.01 * self.criterion_l1(x_real, x_rec)
            # c_hat_loss = self.criterion_bce(c_z_hat, real_labels)
            # # c_loss = self.criterion_bce(z, real_labels)
            # # d_real_loss = self.criterion_bce(x_real, fake_labels)
            # d_rec_loss = self.criterion_bce(d_rec, real_labels)
            # d_gen_loss = self.criterion_bce(d_gen, real_labels)
            # ge_loss = l1_loss + c_hat_loss + d_rec_loss + d_gen_loss
            # ge_loss.backward(retain_graph=True)
            # self.ge_optimizer.step()

            # ================== Train E ================== #
            self.e_optimizer.zero_grad()
            l1_loss = 0.01 * self.l1(x_real, x_rec)
            c_hat_loss = self.bce(c_z_hat, real_labels) - self.bce(c_z_hat, fake_labels)
            e_loss = l1_loss + c_hat_loss
            e_loss.backward(retain_graph=True)
            self.e_optimizer.step()
            
            # ================== Train G ================== #
            self.g_optimizer.zero_grad()
            g_rec_loss = self.bce(d_rec, real_labels) - self.bce(d_rec, fake_labels)
            g_gen_loss = self.bce(d_gen, real_labels) - self.bce(d_gen, fake_labels)
            g_loss = l1_loss + g_rec_loss + g_gen_loss
            g_loss.backward(retain_graph=True)
            self.g_optimizer.step()

            # ================== Train D ================== #
            self.d_optimizer.zero_grad()
            d_real_loss = self.bce(d_real, real_labels)
            d_rec_loss = self.bce(d_rec, fake_labels)
            d_gen_loss = self.bce(d_gen, fake_labels)
            d_loss = d_real_loss + d_rec_loss + d_gen_loss 
            d_loss.backward(retain_graph=True)
            self.d_optimizer.step()

            # ================== Train C ================== #
            self.c_optimizer.zero_grad()
            c_hat_loss = self.bce(c_z_hat, fake_labels)
            c_z_loss = self.bce(c_z, real_labels)
            c_loss = c_hat_loss + c_z_loss
            c_loss.backward()
            self.c_optimizer.step()

            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(f"Elapsed: [{elapsed}], step: [{step+1}/{self.total_step}], e_loss: {e_loss}, g_loss: {g_loss}, d_loss: {d_loss}, c_loss: {c_loss}")
                
                if self.use_tensorboard:
                    self.writer.add_scalar('loss/e', e_loss.data, step+1)
                    self.writer.add_scalar('loss/g', g_loss.data, step+1)
                    self.writer.add_scalar('loss/d', d_loss.data, step+1)
                    self.writer.add_scalar('loss/c', c_loss.data, step+1)
                    
            # Sample images
            if (step + 1) % self.sample_step == 0:
                img_from_z, _, _ = self.G(fixed_z)
                z_mean, z_log_var, _, _ = self.E(tensor2var(fixed_img))
                z_from_img = z_mean + z_log_var * torch.randn(z_mean.size()).cuda()
                rec_from_z, _, _ = self.G(z_from_img)

                if self.use_tensorboard:
                    self.writer.add_image('img/rec_from_z', denorm(rec_from_z.data), step + 1)
                    self.writer.add_image('img/img_from_z', denorm(img_from_z.data), step + 1)
                else:
                    save_image(denorm(img_from_z.data),
                            os.path.join(self.sample_path, '{}_img_from_z.png'.format(step + 1)))
                    save_image(denorm(reimg_from_z.data),
                            os.path.join(self.sample_path, '{}_reimg_from_z.png'.format(step + 1)))

            if (step+1) % model_save_step==0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                torch.save(self.E.state_dict(),
                           os.path.join(self.model_save_path, '{}_E.pth'.format(step + 1)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))
                torch.save(self.C.state_dict(),
                           os.path.join(self.model_save_path, '{}_C.pth'.format(step + 1)))

    def build_model(self):
        self.G = Generator(self.batch_size, self.imsize, self.z_dim, self.g_conv_dim).cuda()
        self.E = Encoder(self.batch_size, self.imsize, self.z_dim, self.d_conv_dim).cuda()
        self.D = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).cuda()
        self.C = Codecriminator(self.batch_size, self.z_dim).cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.E = nn.DataParallel(self.E)
            self.D = nn.DataParallel(self.D)
            self.C = nn.DataParallel(self.C)

        # Loss and optimizer
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.ge_lr, [self.beta1, self.beta2])
        self.e_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.E.parameters()), self.ge_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])
        self.c_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.C.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.l1 = nn.L1Loss()
        self.bce = nn.BCELoss()

        # print networks
        print(self.G)
        print(self.E)
        print(self.D)
        print(self.C)

    def build_tensorboard(self):
        '''Initialize tensorboard writeri'''
        self.writer = SummaryWriter(self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.E.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_E.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        self.C.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_C.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
