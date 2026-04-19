"""
基于卷积神经网络的生成对抗网络
来自https://pytorch.ac.cn/tutorials/beginner/dcgan_faces_tutorial.html
"""
from collections import namedtuple
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from common.lion_pytorch import Lion

class Generator(nn.Module):
    latent_len:int
    channel_scale: int
    
    def __init__(self, latent_len:int, channel_scale: int):
        """
        :param latent_len: 潜空间向量长度 
        :param channel_scale: 卷积通道数控制
        """
        super(Generator, self).__init__()
        
        self.latent_len = latent_len
        self.channel_scale = channel_scale
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( latent_len, channel_scale * 8, 4, 1, 0, bias=True),
            nn.BatchNorm2d(channel_scale * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(channel_scale * 8, channel_scale * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channel_scale * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( channel_scale * 4, channel_scale * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channel_scale * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( channel_scale * 2, channel_scale, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channel_scale),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( channel_scale, 3, 4, 2, 1, bias=False),
            # state size. ``(nc) x 64 x 64``
            nn.Tanh()
        )

    def forward(self, input) -> Tensor:
        return self.main(input)
    
    def generate(self, batch_size: int) -> Tensor:
        noise = torch.randn(batch_size, self.latent_len, 1, 1, device=next(self.parameters()).device, requires_grad=False)
        return self(noise)
    
    
class Discriminator(nn.Module):
    def __init__(self, channel_scale: int):
        """ 
        :param channel_scale: 卷积通道数控制
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is ``3 x 64 x 64``
            nn.Conv2d(3, channel_scale, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(channel_scale) x 32 x 32``
            nn.Conv2d(channel_scale, channel_scale * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channel_scale * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(channel_scale*2) x 16 x 16``
            nn.Conv2d(channel_scale * 2, channel_scale * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channel_scale * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(channel_scale*4) x 8 x 8``
            nn.Conv2d(channel_scale * 4, channel_scale * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channel_scale * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(channel_scale*8) x 4 x 4``
            nn.Conv2d(channel_scale * 8, 1, 4, 1, 0, bias=False),
            nn.Flatten(0)
        )

    def forward(self, input) -> Tensor:
        return self.main(input)
    
    def loss(self, input_image: Tensor, label: Tensor) -> Tensor:
        logit = self(input_image)
        loss= torch.nn.functional.binary_cross_entropy_with_logits(logit, label)
        return loss

class Trainer:
    generator: Generator
    discriminator: Discriminator
    dataloader: DataLoader
    
    save_path: Path
    device: torch.device
    
    batch_size: int
    num_steps: int
    
    def __init__(self, generator: Generator, dataset: Dataset, save_path: Path, device: torch.device, batch_size: int, num_steps: int):
        self.generator = generator
        self.discriminator = Discriminator(generator.channel_scale)
        self.dataloader = DataLoader(dataset, batch_size, shuffle=True, pin_memory=True, drop_last=True)
        
        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        self.batch_size = batch_size
        self.num_steps = num_steps
        pass
    
    def train(self):
        self.generator.to(self.device, non_blocking=True).train()
        self.discriminator.to(self.device, non_blocking=True).train()
        
        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        label_real = torch.ones(self.batch_size, device=self.device, requires_grad=False)
        label_fake= torch.zeros(self.batch_size, device=self.device, requires_grad=False)
        
        epoch = 0
        step = 0
        
        def train_batch(data_batch: Tensor):
            data_batch = data_batch.to(self.device, non_blocking=True) / 255
            
            opt_d.zero_grad()
            
            # 使用真实图像训练判别器
            loss_real= self.discriminator.loss(data_batch, label_real)
            loss_real.backward()
            
            # 生成虚构图像训练判别器
            fake_image = self.generator.generate(self.batch_size)
            loss_fake = self.discriminator.loss(fake_image.detach(), label_fake) 
            loss_fake.backward()
            
            opt_d.step()
            
            loss_d = loss_real + loss_fake
            
            # 生成虚构图像训练生成器
            opt_g.zero_grad()
            
            # fake_image = self.generator.generate(self.batch_size)
            loss_g = self.discriminator.loss(fake_image, label_real)
            loss_g.backward()
            
            opt_g.step()
                    
            print(f"[{step}/{self.num_steps}, epoch {epoch}]: loss_g={loss_g.item()} loss_d={loss_d.item()} loss_real={loss_real.item()} loss_fake={loss_fake.item()}")
            pass
        
        while True:
            for data_batch in self.dataloader:
                if step >= self.num_steps:
                    break
                    
                train_batch(data_batch)
                
                step += self.batch_size
                pass
            
            epoch += 1
            if step >= self.num_steps:
                break
                
            
        
            
            
            