import os
from typing import Optional

import cv2
import torch
import tqdm
from torch.utils.data import DataLoader

from common import DataSet, utils
from ddpm_ori.denoising_diffusion_pytorch import GaussianDiffusion, Unet


def guass_model(image_size: int, weight: Optional[str] = None) -> GaussianDiffusion:
    model = GaussianDiffusion(
        Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8)
        ),
        image_size=image_size,
        timesteps=1000,  # number of steps
    )

    model: GaussianDiffusion = torch.compile(model, fullgraph=True, disable=False) # noqa
    
    if weight is not None:
        model.load_state_dict(torch.load(weight, weights_only=True))

    return model

def _train(
        model: GaussianDiffusion,
        data: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        save_dir: str
):
    model.train()
    device = next(model.parameters()).device
    print("Start Training")
    for e in range(1, epoch + 1):
        running_loss = []
        for batch in tqdm.tqdm(data):
            batch = batch.to(device, non_blocking=True) / 255

            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()

            running_loss.append(loss)  # 保存引用，不获取loss的值以降低传输延迟

        pass  # 一个epoch结束

        mean_loss = sum(l.item() for l in running_loss) / len(data)
        print(f"Finished epoch {e}/{epoch}, mean-loss: {mean_loss}")

        if e % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"guass_diffusion_{e}.pth"))

    torch.save(model.state_dict(), os.path.join(save_dir, "guass_diffusion.pth"))
    
    pass

def train(
        dataset: str,
        weight: Optional[str],
        save: str,
        epoch: int
):
    device = torch.device("cuda")

    batch_size = 16
    # 加载数据集
    data_set = DataSet.H5Dataset(dataset)
    image_shape = data_set[0].shape
    assert image_shape[0] == 3 and image_shape[1] == image_shape[2]
    image_size = image_shape[1]
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             drop_last=True)
    
    # 创建文件夹
    dir_model = os.path.join(save, "gauss_ddpm_weights")
    os.makedirs(dir_model, exist_ok=True)

    print(f"图像大小为{image_size}")

    model = guass_model(image_size, weight)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    # 训练
    _train(model, data_loader, optimizer, epoch, dir_model)
    
    model = model.to(device).eval()
    count = 16
    images = (model.sample(count) * 255).permute((0, 2, 3, 1)).numpy(force=True)
    save = utils.auto_increase_dir(os.path.join(save, "result/guass"))
    for i in range(count):
        cv2.imwrite(os.path.join(save, f"gen{i}.png"), images[i, ...])

    pass

def sample(image_size: int, weight: str, count: int=16):
    device = torch.device("cuda")
    model = guass_model(image_size, weight).to(device).eval()
    
    images = (model.sample(count) * 255).permute((0, 2, 3, 1)).numpy(force=True)
    save = utils.auto_increase_dir("run/result/guass")
    for i in range(count):
        cv2.imwrite(os.path.join(save, f"gen{i}.png"), images[i, ...])

    

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    
    train("./run/datasets/anime_faces64.h5", None, save="./run", epoch=5)
    sample(64, "./run/gauss_ddpm_weights/guass_diffusion.pth")
