
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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def log(x):
    return torch.log(x + 1e-10)

class Trainer(object):
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

        # Start time
        start_time = time.time()
        for step in range(start, self.total_step):
            self.reset_grad()
            # Sample from data and prior
            try:
                real_images, _ = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                real_images, _ = next(data_iter)

            real_images = tensor2var(real_images)
            fake_z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            
            noise1 = torch.Tensor(real_images.size()).normal_(0, 0.01 * (step + 1 - self.total_step) / (step+1)).cuda()

            noise2 = torch.Tensor(real_images.size()).normal_(0, 0.01 * (step +1 - self.total_step) / (step+1)).cuda()
            # Sample from condition
            real_z, _, _ = self.E(real_images)
            fake_images, gf1, gf2 = self.G(fake_z)

            dr, dr5, dr4, dr3, drz, dra2, dra1 = self.D(real_images+noise1, real_z)
            df, df5, df4, df3, dfz, dfa2, dfa1 = self.D(fake_images+noise2, fake_z)

            # Compute loss with real and fake images
            # dr1, dr2, df1, df2, gf1, gf2 are attention scores
            if self.adv_loss == 'wgan-gp':
                d_loss_real = - torch.mean(dr)
                d_loss_fake = df.mean()
                g_loss_fake = - df.mean()
                e_loss_real = - dr.mean() 
            elif self.adv_loss == 'hinge1':
                d_loss_real = torch.nn.ReLU()(1.0 - dr).mean()
                d_loss_fake = torch.nn.ReLU()(1.0 + df).mean()
                g_loss_fake = - df.mean()
                e_loss_real = - dr.mean() 
            elif self.adv_loss == 'hinge':
                d_loss_real = - log(dr).mean()
                d_loss_fake = - log(1.0 - df).mean()
                g_loss_fake = - log(df).mean()
                e_loss_real = - log(1.0 - dr).mean() 
            elif self.adv_loss == 'inverse':
                d_loss_real = - log(1.0 - dr).mean()
                d_loss_fake = - log(df).mean()
                g_loss_fake = - log(1.0 - df).mean()
                e_loss_real = - log(dr).mean() 

            # ================== Train D ================== #
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward(retain_graph=True)
            self.d_optimizer.step()

            if self.adv_loss == 'wgan-gp':
                # Compute gradient penalty
                alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
                interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
                out,_,_ = self.D(interpolated)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp

                d_loss.backward()
                self.d_optimizer.step()

            # ================== Train G and E ================== #
            ge_loss = g_loss_fake + e_loss_real
            ge_loss.backward()
            self.ge_optimizer.step()

            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(f"Elapsed: [{elapsed}], step: [{step+1}/{self.total_step}], d_loss: {d_loss}, ge_loss: {ge_loss}")
                
                if self.use_tensorboard:
                    self.writer.add_scalar('d/loss_real', d_loss_real.data, step+1)
                    self.writer.add_scalar('d/loss_fake', d_loss_fake.data, step+1)
                    self.writer.add_scalar('d/loss', d_loss.data, step+1)
                    self.writer.add_scalar('ge/loss_real', e_loss_real.data, step+1)
                    self.writer.add_scalar('ge/loss_fake', g_loss_fake.data, step+1)
                    self.writer.add_scalar('ge/loss', ge_loss.data, step+1)
                    self.writer.add_scalar('ave_gamma/l3', self.G.attn1.gamma.mean().data, step+1)
                    self.writer.add_scalar('ave_gamma/l4', self.G.attn2.gamma.mean().data, step+1)
                    
            # Sample images
            if (step + 1) % self.sample_step == 0:
                img_from_z, _, _ = self.G(fixed_z)
                z_from_img, _, _ = self.E(tensor2var(fixed_img))
                reimg_from_z, _, _ = self.G(z_from_img)

                if self.use_tensorboard:
                    self.writer.add_image('img/reimg_from_z', denorm(reimg_from_z.data), step + 1)
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

    def build_model(self):
        self.G = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).cuda()
        self.E = Encoder(self.batch_size, self.imsize, self.z_dim, self.d_conv_dim).cuda()
        self.D = Discriminator(self.batch_size,self.imsize, self.z_dim, self.d_conv_dim).cuda()
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
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.ge_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
