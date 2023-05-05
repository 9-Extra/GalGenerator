# 扩散模型
import argparse

import torch
from torch.nn import Module
from common.common_layers import Conv2, Conv1, MultiHeadSelfAttention, MultiHeadSelfAttentionCV, ResBlock, \
    TimeEmbedResBlock
import torchsummary

import torch.nn as nn

# 超参数
IMAGE_SIZE = (160, 160)
T = 1000  # 总层数
BETA = torch.linspace(0.0001, 0.02, T, dtype=torch.float32)  # 每一层增加的噪声水平 forward process variances （实际上还要开方）
ALPHA = 1.0 - BETA  # 相当于每层剩余的原图片比率（实际上还要开方）


# 将Diffusion模型输入的t信息编码到输入的图像tensor中，借用了transformer的位置编码方法
def cal_position_encoding() -> torch.Tensor:
    d = IMAGE_SIZE[0] * 3
    encoding = torch.empty((T, d), dtype=torch.float32)
    omega = 10000 * torch.exp(torch.arange(0, d, 2, dtype=torch.float32) / -d)
    for t in range(T):
        encoding[t, 0::2] = torch.sin(omega * (t + 1))  # 偶数位
        encoding[t, 1::2] = torch.sin(omega * (t + 1))  # 奇数位

    return encoding


# 预计算一些常用值
alpha_over_line = torch.cumprod(ALPHA, dim=0)  # alpha累乘的结果
alpha_over_line_sqrt = torch.sqrt(alpha_over_line)
one_sub_alpha_over_line_sqrt = torch.sqrt(1 - alpha_over_line)
alpha_sqrt = torch.sqrt(ALPHA)  # alpha开方的结果
beta_sqrt = torch.sqrt(BETA)
position_encoding = cal_position_encoding()
position_encoding_size = position_encoding.shape[1]  # 位置编码的长度


def to_device(device):
    global ALPHA, BETA, alpha_over_line, alpha_over_line_sqrt, one_sub_alpha_over_line_sqrt, alpha_sqrt, beta_sqrt, position_encoding
    ALPHA = ALPHA.to(device)
    BETA = BETA.to(device)
    alpha_over_line_sqrt = alpha_over_line_sqrt.to(device)
    one_sub_alpha_over_line_sqrt = one_sub_alpha_over_line_sqrt.to(device)
    alpha_over_line = alpha_over_line.to(device)
    alpha_sqrt = alpha_sqrt.to(device)
    beta_sqrt = beta_sqrt.to(device)
    position_encoding = position_encoding.to(device)


class Down(Module):
    """Downscaling with maxpool then double conv"""
    res1: Module
    res2: Module
    down_sample: Module

    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        self.res1 = TimeEmbedResBlock(in_channels, out_channels, time_embed_dim)
        self.res2 = TimeEmbedResBlock(out_channels, out_channels, time_embed_dim)

        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x, embed):
        x1 = self.res1(x, embed)
        x2 = self.res2(x1, embed)
        return x1, x2, self.down_sample(x2)


class Up(Module):
    """Upscaling then double conv"""
    res1: Module
    res2: Module
    up_sample: Module

    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        self.res1 = TimeEmbedResBlock(in_channels + out_channels, out_channels, time_embed_dim)
        self.res2 = TimeEmbedResBlock(in_channels + out_channels, out_channels, time_embed_dim)
        self.up_sample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x, x1, x2, embed):
        x = self.up_sample(x)
        x = self.res1(torch.cat([x, x1], dim=1), embed)
        x = self.res2(torch.cat([x, x2], dim=1), embed)
        return x


class UNet(Module):
    middle_res1: Module
    middle_res2: Module
    middle_attention: Module
    middle_norm: Module

    def __init__(self, n_channels, time_embed_dim):
        super(UNet, self).__init__()
        self.inc = Conv2(n_channels, 32)
        self.down1 = Down(32, 64, time_embed_dim)
        self.down2 = Down(64, 128, time_embed_dim)
        self.down3 = Down(128, 256, time_embed_dim)
        self.down4 = Down(256, 512, time_embed_dim)

        self.middle_res1 = TimeEmbedResBlock(512, 512, time_embed_dim)
        self.middle_norm = torch.nn.LayerNorm([512, IMAGE_SIZE[0] // 16, IMAGE_SIZE[1] // 16])
        self.middle_attention = MultiHeadSelfAttentionCV(512, 512)
        self.middle_res2 = TimeEmbedResBlock(512, 512, time_embed_dim)

        self.up1 = Up(512, 256, time_embed_dim)
        self.up2 = Up(256, 128, time_embed_dim)
        self.up3 = Up(128, 64, time_embed_dim)
        self.up4 = Up(64, 32, time_embed_dim)

    def forward(self, x, embed):
        connect = []  # 传递的连接输出

        x = self.inc(x)

        x1, x2, x = self.down1(x, embed)
        connect.extend([x1, x2])
        x1, x2, x = self.down2(x, embed)
        connect.extend([x1, x2])
        x1, x2, x = self.down3(x, embed)
        connect.extend([x1, x2])
        x1, x2, x = self.down4(x, embed)
        connect.extend([x1, x2])

        x = self.middle_norm(self.middle_res1(x, embed))
        x = self.middle_attention(x)
        x = self.middle_res2(x, embed)
        
        x2, x1 = connect.pop(), connect.pop()
        x = self.up1(x, x1, x2, embed)
        x2, x1 = connect.pop(), connect.pop()
        x = self.up2(x, x1, x2, embed)
        x2, x1 = connect.pop(), connect.pop()
        x = self.up3(x, x1, x2, embed)
        x2, x1 = connect.pop(), connect.pop()
        x = self.up4(x, x1, x2, embed)
        return x


class Diffusion(Module):
    unet: Module
    mlp: Module

    def __init__(self):
        super().__init__()
        time_embed_dim = position_encoding_size
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(position_encoding_size, time_embed_dim),
            torch.nn.SELU(),
            torch.nn.Linear(time_embed_dim, time_embed_dim),
            torch.nn.SELU(),
        )
        # 内部就是个UNet
        self.unet = UNet(3, time_embed_dim)

        self.conv = torch.nn.Conv2d(32, 3, 1, bias=True)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        embed = self.mlp(t)
        x = self.unet(x, embed)
        return self.conv(x)


def run(
        device: str
):
    batch_size = 16
    device = torch.device(device)

    auto_encoder = Diffusion()
    auto_encoder.to(device).train()
    to_device(device)

    torchsummary.summary(auto_encoder, [[3, *IMAGE_SIZE], [position_encoding.shape[1]]], batch_size=batch_size)

    pass


if __name__ == '__main__':
    # 仅仅加载模型
    opt = argparse.ArgumentParser()
    opt.add_argument('--device', type=str, default="cuda")
    run(**vars(opt.parse_args()))
