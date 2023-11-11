import argparse
import os

import cv2
import numpy
import torch
import tqdm

import Diffusion
from common import utils


# 生成图片

def _tensor_to_image(t: torch.Tensor) -> numpy.ndarray:
    # 从[-1, 1]映射到[0, 255]
    image: numpy.ndarray = (t.numpy(force=True) + 1.0) * (255 / 2)
    image = image.transpose((0, 2, 3, 1))
    return image.astype(numpy.uint8)


def sample(diffusion: Diffusion.Diffusion, image_size: int, count: int, device: torch.device) -> numpy.ndarray:
    alphas_cumprod_prev = torch.nn.functional.pad(diffusion.alpha_cumprod[:-1], (1, 0), value=1.)
    alpha_cumprod_sqrt_rev = 1.0 / diffusion.alpha_cumprod_sqrt
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / diffusion.alpha_cumprod - 1.0)
    posterior_variance = (diffusion.beta * (1. - alphas_cumprod_prev) / (1. - diffusion.alpha_cumprod)).clamp(
        min=1e-20)
    posterior_mean_coef1 = diffusion.beta * torch.sqrt(alphas_cumprod_prev) / (1. - diffusion.alpha_cumprod)
    posterior_mean_coef2 = (1. - alphas_cumprod_prev) * diffusion.alpha_sqrt / (1. - diffusion.alpha_cumprod)

    input_size = [count, 3, image_size, image_size]
    noise = torch.randn(input_size, device=device)  # 生成高斯噪声
    x = noise  # x_t
    for t in tqdm.tqdm(reversed(range(0, diffusion.total_timestep)), total=diffusion.total_timestep):
        # 前向过程
        z_noise = torch.randn(input_size, device=device) if t != 0 else torch.zeros(input_size, device=device)
        batch_t: torch.Tensor = torch.full([count], t, device=device)
        noise_predicted = diffusion.forward(x, batch_t)

        # cv2.imshow("x", ((x + 1.0) / 2).numpy(force=True).transpose((0, 2, 3, 1))[0, :, :, ::1])
        # cv2.imshow("noise", noise_predicted.numpy(force=True).transpose((0, 2, 3, 1))[0, :, :, ::1])
        # cv2.waitKey()

        x_start = alpha_cumprod_sqrt_rev[t] * x - sqrt_recipm1_alphas_cumprod[t] * noise_predicted
        x_start = x_start.clip(-1.0, 1.0)
        posterior_mean = posterior_mean_coef1[t] * x_start + posterior_mean_coef2[t] * x
        # print(posterior_mean_coef1[t].item(), posterior_mean_coef2[t].item(), posterior_variance[t].item())
        pred_img = posterior_mean + torch.sqrt(posterior_variance[t]) * z_noise
        x = pred_img

    pass

    x = x.clip(-1.0, 1.0)

    return _tensor_to_image(x)


def run(
        weight: str,
        save_dir: str,
        count: int,
        device: str
):
    image_size = 32

    save_dir = utils.auto_increase_dir(save_dir)
    device = torch.device(device)

    diffusion = Diffusion.Diffusion(
        image_size
    )
    diffusion.load_state_dict(torch.load(weight))
    diffusion.to(device)

    diffusion.eval()

    # generate
    with torch.no_grad():
        images = sample(diffusion, diffusion.image_size, count, device)
        for i in range(count):
            cv2.imwrite(os.path.join(save_dir, f"gen{i}.png"), images[i, ...])
    pass


if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('weight', type=str)
    opt.add_argument('--save_dir', type=str, default='./result')
    opt.add_argument('--count', type=int, default=4)
    opt.add_argument('--device', type=str, default="cuda")
    run(**vars(opt.parse_args()))
