import os

import cv2
import numpy
import torch


import ddpm.diffusion as diffusion
from common import utils


# 生成图片

def _tensor_to_image(t: torch.Tensor) -> numpy.ndarray:
    # 从[-1, 1]映射到[0, 255]
    image: numpy.ndarray = (t.numpy(force=True) + 1.0) * (255 / 2)
    image = image.transpose((0, 2, 3, 1))
    return image.astype(numpy.uint8)


def run(
        weight: str,
        image_size: int,
        save_dir: str,
        count: int,
        device: str
):
    torch.set_float32_matmul_precision('high')
    device = torch.device(device)

    model = diffusion.Diffusion(
        image_size
    )
    model: diffusion.Diffusion = torch.compile(model)  # typing: ignore
    model.load_state_dict(torch.load(weight, weights_only=True))
    model.to(device).eval()

    # generate
    images = _tensor_to_image(model.sample(count))
    save_dir = utils.auto_increase_dir(save_dir)
    for i in range(count):
        cv2.imwrite(os.path.join(save_dir, f"gen{i}.png"), images[i, ...])

pass
