# 扩散模型
import argparse
import math

import torch
from torch.nn import Module
import tqdm

from common.common_layers import LinearAttentionCV, MultiHeadSelfAttentionCV, TimeEmbedResBlock


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
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
    
    
# def cal_position_encoding(time_step: int, emb_dim: int) -> torch.Tensor:
#     """
#     将Diffusion模型输入的t信息编码到输入的图像tensor中，借用了transformer的位置编码方法
#     返回的位置编码形状为[time, emb_dim]，对每个时间都有一个长度为emb_dim的向量
#     """
#     # encoding = torch.empty((time_step, emb_dim), dtype=torch.float32)
#     # e = 2 * math.log(10000) / emb_dim # omega的指数部分
#     # omega = torch.exp(torch.arange(emb_dim // 2, dtype=torch.float32) * -e)
#     # t_series = torch.arange(time_step, device=omega.device)
#     # omega_t = t_series[:, None] * omega[None, :]
#     # encoding[:, 0::2] = torch.sin(omega_t)  # 偶数位
#     # encoding[:, 1::2] = torch.cos(omega_t)  # 奇数位    
#     emb = SinusoidalPosEmb(emb_dim)
#     encoding = emb(torch.arange(time_step))
#     return encoding

def cal_position_encoding(time_step: int, emb_dim: int) -> torch.Tensor:
    encoding = torch.empty((time_step, emb_dim), dtype=torch.float32)
    omega = 10000 * torch.exp(torch.arange(0, emb_dim, 2, dtype=torch.float32) / -emb_dim)
    for t in range(time_step):
        encoding[t, 0::2] = torch.sin(omega * (t + 1))  # 偶数位
        encoding[t, 1::2] = torch.cos(omega * (t + 1))  # 奇数位

    return encoding


class Down(Module):
    """Downscaling with maxpool then double conv"""
    res1: Module
    res2: Module
    down_sample: Module

    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        self.res1 = TimeEmbedResBlock(in_channels, in_channels, time_embed_dim)
        self.res2 = TimeEmbedResBlock(in_channels, in_channels, time_embed_dim)

        self.attention = torch.nn.Sequential(
            torch.nn.GroupNorm(32, in_channels),
            LinearAttentionCV(in_channels, in_channels),
            torch.nn.GroupNorm(32, in_channels),
        )

        self.down_sample = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            torch.nn.SiLU()
        )

    def forward(self, x, embed):
        x1 = self.res1(x, embed)
        x = self.res2(x1, embed)
        x2 = x + self.attention(x)
        return x1, x2, self.down_sample(x)


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

        self.res1 = TimeEmbedResBlock(out_channels + out_channels, out_channels, time_embed_dim)
        self.res2 = TimeEmbedResBlock(out_channels + out_channels, out_channels, time_embed_dim)

        self.attention = torch.nn.Sequential(
            torch.nn.GroupNorm(32, out_channels),
            LinearAttentionCV(out_channels, out_channels),
            torch.nn.GroupNorm(32, out_channels),
        )

    def forward(self, x, x1, x2, embed):
        x = self.up_sample(x)
        x = self.res1(torch.cat([x, x1], dim=1), embed)
        x = self.res2(torch.cat([x, x2], dim=1), embed)
        x = x + self.attention(x)
        return x


class UNet(Module):
    middle_res1: Module
    middle_res2: Module
    middle_attention: Module
    middle_norm: Module

    def __init__(self, image_size: int, n_channels: int, time_embed_dim: int):
        super(UNet, self).__init__()
        self.inc = torch.nn.Conv2d(n_channels, 64, 7, padding=3)
        self.down1 = Down(64, 64, time_embed_dim)
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

        x = self.final_res_block(torch.cat([x, r], dim=1), embed)

        return self.final_conv(x)


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
        self.register_buffer("position_encoding", cal_position_encoding(total_timestep, image_size * 8))
        position_encoding_size = self.position_encoding.shape[1]  # 位置编码的长度

        time_embed_dim = position_encoding_size * 2
        # 先对位置编码使用mlp进行映射
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(position_encoding_size, time_embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_embed_dim, time_embed_dim),
            torch.nn.SiLU(),
        )
        # 内部就是个UNet
        self.unet = UNet(image_size, 3, time_embed_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        embed = self.mlp(torch.index_select(self.position_encoding, 0, t))
        x = self.unet(x, embed)
        return x

    def forward_process(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向过程，为图像加上t次噪声，返回加上噪声后的图像和加上的噪声（训练用）
        :param x_0: 原始图像
        :param t: 前向的时间
        :return: 加噪后图像和噪声
        """
        noise = torch.randn(x_0.shape, dtype=x_0.dtype, device=x_0.device)  # 生成高斯噪声
        alpha_over_line_sqrt_t = torch.index_select(self.alpha_cumprod_sqrt, 0, t).view((-1, 1, 1, 1))
        one_sub_alpha_over_line_sqrt_t = torch.index_select(self.one_minus_alpha_cumprod_sqrt, 0, t).view(
            (-1, 1, 1, 1))
        x_t = alpha_over_line_sqrt_t * x_0 + one_sub_alpha_over_line_sqrt_t * noise

        return x_t, noise
    
    @torch.no_grad()
    def sample(self, batch_size: int) -> torch.Tensor:
        """
        采样图像，一次生成batch_size个，得到的图像像素取值范围为[-1, 1]，还需要再映射一下
        """
        device = next(self.parameters()).device
        
        alphas_cumprod_prev = torch.nn.functional.pad(self.alpha_cumprod[:-1], (1, 0), value=1.)
        alpha_cumprod_sqrt_rev = 1.0 / self.alpha_cumprod_sqrt
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alpha_cumprod - 1.0)
        # 后验高斯分布的方差
        posterior_variance = (self.beta * (1 - alphas_cumprod_prev) / (1 - self.alpha_cumprod)).clamp(
            min=1e-20)
        # 后验高斯分布均值的两个系数
        posterior_mean_coef1 = self.beta * torch.sqrt(alphas_cumprod_prev) / (1 - self.alpha_cumprod)
        posterior_mean_coef2 = (1 - alphas_cumprod_prev) * self.alpha_sqrt / (1 - self.alpha_cumprod)

        input_size = (batch_size, 3, self.image_size, self.image_size)
        x = torch.randn(input_size, device=device)  # 从高斯噪声开始
        for t in tqdm.tqdm(reversed(range(0, self.total_timestep)), total=self.total_timestep, desc="采样图像"):
            # 前向过程
            z_noise = torch.randn(input_size, device=device) if t != 0 else torch.zeros(input_size, device=device)
            batch_t: torch.Tensor = torch.full((batch_size,), t, device=device)
            noise_predicted = self.forward(x, batch_t)  # 预测出的噪声

            x_start = alpha_cumprod_sqrt_rev[t] * x - sqrt_recipm1_alphas_cumprod[t] * noise_predicted
            x_start = x_start.clip_(-1.0, 1.0)
            posterior_mean = posterior_mean_coef1[t] * x_start + posterior_mean_coef2[t] * x
    
            pred_img = posterior_mean + torch.sqrt(input=posterior_variance[t]) * z_noise
            x = pred_img

        pass

        return x.clip_(-1.0, 1.0)

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

    pass


if __name__ == '__main__':
    # 仅仅加载模型
    opt = argparse.ArgumentParser()
    opt.add_argument('--device', type=str, default="cuda")
    run(**vars(opt.parse_args()))
