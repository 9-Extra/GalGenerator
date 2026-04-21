# 扩散模型共用的 UNet 架构
import math

import torch
from torch.nn import Module

from .common_layers import (
    MultiHeadSelfAttentionCV,
    LinearAttentionCV,
    TimeEmbedResBlock,
)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = torch.nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=1) * self.g * self.scale


class SinusoidalPosEmb(Module):
    def __init__(self, dim: int, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Down(Module):
    res1: Module
    res2: Module
    down_sample: Module

    def __init__(
        self,
        in_channels,
        out_channels,
        time_embed_dim
    ):
        super().__init__()
        self.res1 = TimeEmbedResBlock(in_channels, in_channels, time_embed_dim)
        self.res2 = TimeEmbedResBlock(in_channels, out_channels, time_embed_dim)

        self.down_sample = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, out_channels // 4, 3, 1, 1, bias=False),
            torch.nn.PixelUnshuffle(2),
        )

    def forward(self, x, embed) -> tuple[torch.Tensor, torch.Tensor]:
        """返回下采样前的特征用于跨层连接，下采用后的特征用于继续前向传播"""
        x = self.res1(x, embed)
        x = self.res2(x, embed)
        return x, self.down_sample(x)


class Up(Module):
    """Upscaling then double conv"""

    res1: Module
    res2: Module
    up_sample: Module

    def __init__(
        self,
        in_channels,
        out_channels,
        time_embed_dim,
    ):
        super().__init__()

        self.up_sample = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels * 4, 3, padding=1, bias=False),
            torch.nn.PixelShuffle(2),
        )

        self.res1 = TimeEmbedResBlock(
            in_channels + in_channels, out_channels, time_embed_dim
        )
        self.res2 = TimeEmbedResBlock(out_channels, out_channels, time_embed_dim)

    def forward(self, x, x1, embed):
        x = self.up_sample(x)
        x = self.res1(torch.cat([x, x1], dim=1), embed)
        x = self.res2(x, embed)
        return x


class UNet(Module):
    middle_res1: Module
    middle_res2: Module
    middle_attention1: Module
    middle_attention2: Module
    middle_norm1: Module
    middle_norm2: Module

    def __init__(self, image_size: int, n_channels: int, time_embed_dim: int):
        super(UNet, self).__init__()
        self.inc = torch.nn.Conv2d(n_channels, 64, 3, padding=1)
        self.down1 = Down(64, 64, time_embed_dim)
        self.down2 = Down(64, 128, time_embed_dim)
        self.down3 = Down(128, 256, time_embed_dim)
        self.down4 = Down(256, 512, time_embed_dim)

        self.middle_res1 = TimeEmbedResBlock(512, 512, time_embed_dim)
        self.middle_attn_norm1 = torch.nn.GroupNorm(32, 512)
        self.middle_attention1 = MultiHeadSelfAttentionCV(512, 512, head_dim=64)
        self.middle_res2 = TimeEmbedResBlock(512, 512, time_embed_dim)
        self.middle_attn_norm2 = torch.nn.GroupNorm(32, 512)
        self.middle_attention2 = MultiHeadSelfAttentionCV(512, 512, head_dim=64)

        self.up1 = Up(512, 256, time_embed_dim)
        self.up2 = Up(256, 128, time_embed_dim)
        self.up3 = Up(128, 64, time_embed_dim)
        self.up4 = Up(64, 64, time_embed_dim)

        self.final_res_block = TimeEmbedResBlock(128, 64, time_embed_dim)

        self.final_conv = torch.nn.Conv2d(64, 3, 1)

    def forward(self, x, embed):
        connect = []  # 传递的连接输出

        x = self.inc(x)
        connect.append(x)

        x1, x = self.down1(x, embed)
        connect.append(x1)
        x1, x = self.down2(x, embed)
        connect.append(x1)
        x1, x = self.down3(x, embed)
        connect.append(x1)
        x1, x = self.down4(x, embed)
        connect.append(x1)

        x = self.middle_res1(x, embed)
        x = x + self.middle_attention1(self.middle_attn_norm1(x))
        x = self.middle_res2(x, embed)
        x = x + self.middle_attention2(self.middle_attn_norm2(x))

        x = self.up1(x, connect.pop(), embed)
        x = self.up2(x, connect.pop(), embed)
        x = self.up3(x, connect.pop(), embed)
        x = self.up4(x, connect.pop(), embed)

        x = self.final_res_block(torch.cat((x, connect.pop()), dim=1), embed)

        return self.final_conv(x)
