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
    attn: Module | None
    attn_norm: Module | None

    def __init__(
        self,
        in_channels,
        out_channels,
        time_embed_dim,
        use_attn: bool = False,
        use_linear_attn: bool = False,
    ):
        super().__init__()
        assert not (
            use_attn and use_linear_attn
        ), "use_attn 和 use_linear_attn 不能同时为 True"
        self.res1 = TimeEmbedResBlock(in_channels, in_channels, time_embed_dim)
        self.res2 = TimeEmbedResBlock(in_channels, in_channels, time_embed_dim)

        self.down_sample = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, 2, 1), torch.nn.SiLU()
        )

        if use_attn:
            self.attn_norm = torch.nn.GroupNorm(32, in_channels)
            self.attn = MultiHeadSelfAttentionCV(in_channels, in_channels)
        elif use_linear_attn:
            self.attn_norm = torch.nn.GroupNorm(32, in_channels)
            self.attn = LinearAttentionCV(in_channels, in_channels)
        else:
            self.attn_norm = None
            self.attn = None

    def forward(self, x, embed):
        x1 = self.res1(x, embed)
        x = self.res2(x1, embed)
        if self.attn is not None:
            x = x + self.attn(self.attn_norm(x))
        x2 = x
        return x1, x2, self.down_sample(x)


class Up(Module):
    """Upscaling then double conv"""

    res1: Module
    res2: Module
    up_sample: Module
    attn: Module | None
    attn_norm: Module | None

    def __init__(
        self,
        in_channels,
        out_channels,
        time_embed_dim,
        use_attn: bool = False,
        use_linear_attn: bool = False,
    ):
        super().__init__()
        assert not (
            use_attn and use_linear_attn
        ), "use_attn 和 use_linear_attn 不能同时为 True"

        self.up_sample = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )

        self.res1 = TimeEmbedResBlock(
            out_channels + out_channels, out_channels, time_embed_dim
        )
        self.res2 = TimeEmbedResBlock(
            out_channels + out_channels, out_channels, time_embed_dim
        )

        if use_attn:
            self.attn_norm = torch.nn.GroupNorm(32, out_channels)
            self.attn = MultiHeadSelfAttentionCV(out_channels, out_channels)
        elif use_linear_attn:
            self.attn_norm = torch.nn.GroupNorm(32, out_channels)
            self.attn = LinearAttentionCV(out_channels, out_channels)
        else:
            self.attn_norm = None
            self.attn = None

    def forward(self, x, x1, x2, embed):
        x = self.up_sample(x)
        x = self.res1(torch.cat([x, x1], dim=1), embed)
        x = self.res2(torch.cat([x, x2], dim=1), embed)
        if self.attn is not None:
            x = x + self.attn(self.attn_norm(x))
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
        self.inc = torch.nn.Conv2d(n_channels, 64, 7, padding=3)
        self.down1 = Down(64, 64, time_embed_dim)
        self.down2 = Down(64, 128, time_embed_dim)
        # 32x32 尺度用 LinearAttention（计算量小，能处理长序列）
        self.down3 = Down(128, 256, time_embed_dim, use_linear_attn=True)
        # 16x16 尺度用标准 Attention
        self.down4 = Down(256, 512, time_embed_dim, use_attn=True)

        self.middle_res1 = TimeEmbedResBlock(512, 512, time_embed_dim)
        self.middle_attn_norm1 = torch.nn.GroupNorm(32, 512)
        self.middle_attention1 = MultiHeadSelfAttentionCV(512, 512)
        self.middle_res2 = TimeEmbedResBlock(512, 512, time_embed_dim)
        self.middle_attn_norm2 = torch.nn.GroupNorm(32, 512)
        self.middle_attention2 = MultiHeadSelfAttentionCV(512, 512)

        # 对称地在上采样路径也加 Attention
        self.up1 = Up(512, 256, time_embed_dim, use_attn=True)
        self.up2 = Up(256, 128, time_embed_dim, use_linear_attn=True)
        self.up3 = Up(128, 64, time_embed_dim)
        self.up4 = Up(64, 64, time_embed_dim)

        self.final_res_block = TimeEmbedResBlock(128, 64, time_embed_dim)

        self.final_conv = torch.nn.Conv2d(64, 3, 1)

    def forward(self, x, embed):
        connect = []  # 传递的连接输出

        x = self.inc(x)
        r = x.clone()

        x1, x2, x = self.down1(x, embed)
        connect.extend([x1, x2])
        x1, x2, x = self.down2(x, embed)
        connect.extend([x1, x2])
        x1, x2, x = self.down3(x, embed)
        connect.extend([x1, x2])
        x1, x2, x = self.down4(x, embed)
        connect.extend([x1, x2])

        x = self.middle_res1(x, embed)
        x = x + self.middle_attention1(self.middle_attn_norm1(x))
        x = self.middle_res2(x, embed)
        x = x + self.middle_attention2(self.middle_attn_norm2(x))

        x2, x1 = connect.pop(), connect.pop()
        x = self.up1(x, x1, x2, embed)
        x2, x1 = connect.pop(), connect.pop()
        x = self.up2(x, x1, x2, embed)
        x2, x1 = connect.pop(), connect.pop()
        x = self.up3(x, x1, x2, embed)
        x2, x1 = connect.pop(), connect.pop()
        x = self.up4(x, x1, x2, embed)

        x = self.final_res_block(torch.cat([x, r], dim=1), embed)

        return self.final_conv(x)
