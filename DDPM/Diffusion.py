# 扩散模型
import argparse

import torch
from torch.nn import Module
from common.common_layers import MultiHeadSelfAttentionCV, TimeEmbedResBlock, ChannelDownSample, Conv1
import torchsummary


# 将Diffusion模型输入的t信息编码到输入的图像tensor中，借用了transformer的位置编码方法
def cal_position_encoding(time_step: int, image_size: int) -> torch.Tensor:
    d = image_size * 3
    encoding = torch.empty((time_step, d), dtype=torch.float32)
    omega = 10000 * torch.exp(torch.arange(0, d, 2, dtype=torch.float32) / -d)
    for t in range(time_step):
        encoding[t, 0::2] = torch.sin(omega * (t + 1))  # 偶数位
        encoding[t, 1::2] = torch.sin(omega * (t + 1))  # 奇数位

    return encoding


class Down(Module):
    """Downscaling with maxpool then double conv"""
    res1: Module
    res2: Module
    down_sample: Module

    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        self.res1 = TimeEmbedResBlock(in_channels, out_channels, time_embed_dim)
        self.res2 = TimeEmbedResBlock(out_channels, out_channels, time_embed_dim)

        self.down_sample = ChannelDownSample(out_channels, out_channels)

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
        self.up_sample = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.res1 = TimeEmbedResBlock(in_channels + out_channels, out_channels, time_embed_dim)
        self.res2 = TimeEmbedResBlock(in_channels + out_channels, out_channels, time_embed_dim)

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

    def __init__(self, image_size: int, n_channels: int, time_embed_dim: int):
        super(UNet, self).__init__()
        self.inc = torch.nn.Conv2d(n_channels, 32, 1)
        self.down1 = Down(32, 64, time_embed_dim)
        self.down2 = Down(64, 128, time_embed_dim)
        self.down3 = Down(128, 256, time_embed_dim)
        self.down4 = Down(256, 512, time_embed_dim)

        self.middle_res1 = TimeEmbedResBlock(512, 512, time_embed_dim)
        self.middle_norm = torch.nn.LayerNorm([512, image_size // 16, image_size // 16])
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
    image_size: int
    total_timestep: int

    beta: torch.Tensor
    alpha: torch.Tensor
    beta_sqrt: torch.Tensor
    alpha_sqrt: torch.Tensor
    alpha_cumprod: torch.Tensor
    alpha_cumprod_sqrt: torch.Tensor
    one_minus_alpha_cumprod_sqrt: torch.Tensor
    position_encoding: torch.Tensor

    unet: Module
    mlp: Module

    def __init__(
            self,
            image_size: int,
            total_timestep=1000,
            beta_schedule="linear",
    ):
        super().__init__()
        self.image_size = image_size
        self.total_timestep = total_timestep

        beta = self._cal_beta(total_timestep, beta_schedule)

        self.register_buffer("beta", beta)
        self.register_buffer("beta_sqrt", torch.sqrt(beta))
        self.register_buffer("alpha", 1.0 - beta)
        self.register_buffer("alpha_sqrt", torch.sqrt(self.alpha))
        alpha_cumprod = torch.cumprod(self.alpha, dim=0)  # alpha累乘的结果
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_sqrt", torch.sqrt(alpha_cumprod))
        self.register_buffer("one_minus_alpha_cumprod_sqrt", 1.0 - self.alpha_cumprod_sqrt)
        self.register_buffer("position_encoding", cal_position_encoding(total_timestep, image_size))
        position_encoding_size = self.position_encoding.shape[1]  # 位置编码的长度

        time_embed_dim = position_encoding_size * 4
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(position_encoding_size, time_embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_embed_dim, time_embed_dim),
            torch.nn.SiLU(),
        )
        # 内部就是个UNet
        self.unet = UNet(image_size, 3, time_embed_dim)

        self.conv = torch.nn.Conv2d(32, 3, 1, bias=True)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        embed = self.mlp(torch.index_select(self.position_encoding, 0, t))
        x = self.unet(x, embed)
        return self.conv(x)

    def forward_process(self, x_0: torch.Tensor, t: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        前向过程，为图像加上t次噪声，返回加上噪声后的图像和加上的噪声（训练用）
        :param x_0: 原始图像
        :param t: 前向的时间
        :return: 加噪后图像和噪声
        """
        noise = torch.randn(x_0.size(), dtype=x_0.dtype, device=x_0.device)  # 生成高斯噪声
        alpha_over_line_sqrt_t = torch.index_select(self.alpha_cumprod_sqrt, 0, t).view((-1, 1, 1, 1))
        one_sub_alpha_over_line_sqrt_t = torch.index_select(self.one_minus_alpha_cumprod_sqrt, 0, t).view(
            (-1, 1, 1, 1))
        x_t = alpha_over_line_sqrt_t * x_0 + one_sub_alpha_over_line_sqrt_t * noise

        return x_t, noise

    @staticmethod
    def _cal_beta(timestep: int, beta_schedule) -> torch.Tensor:
        if beta_schedule == "linear":
            return torch.linspace(0.0001, 0.02, timestep, dtype=torch.float32)
        else:
            raise RuntimeError("No!")


def run(
        device: str
):
    batch_size = 16
    image_size = 160
    device = torch.device(device)

    model = Diffusion(
        image_size,
        total_timestep=1000,
        beta_schedule="linear"
    )
    model.to(device).train()

    torchsummary.summary(model, [[3, image_size, image_size], [model.position_encoding.shape[1]]],
                         batch_size=batch_size)

    pass


if __name__ == '__main__':
    # 仅仅加载模型
    opt = argparse.ArgumentParser()
    opt.add_argument('--device', type=str, default="cuda")
    run(**vars(opt.parse_args()))
