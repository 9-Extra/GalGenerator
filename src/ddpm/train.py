import argparse
import os
from typing import Optional
import tqdm

import torch
from torch.utils.data import DataLoader

from . import network

from common import DataSet


def _train(
        model: network.Diffusion,
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
            batch = batch.to(device, non_blocking=True) / (255 / 2) - 1
            t = torch.randint(0, model.total_timestep, [data.batch_size], device=device,
                              dtype=torch.int)  # 随机选一层[0..T)，比论文中的t小1

            x_t, noise = model.forward_process(batch, t)

            optimizer.zero_grad()
            output = model(x_t, t)

            # loss = torch.nn.functional.l1_loss(noise, output)  # l1 损失
            loss = torch.nn.functional.mse_loss(noise, output)  # l2 损失
            loss.backward()
            optimizer.step()

            running_loss.append(loss)  # 保存引用，不获取loss的值以降低传输延迟

        pass  # 一个epoch结束

        mean_loss = sum(l.item() for l in running_loss) / len(data)
        print(f"Finished epoch {e}/{epoch}, mean-loss: {mean_loss}")

        if e % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"diffusion_{e}.pth"))

    pass


def run(
        dataset: str,
        weight: Optional[str],
        save: str,
        epoch: int
):
    device = torch.device("cuda")
    torch.set_float32_matmul_precision('high')
    
    torch.backends.cudnn.benchmark = True

    batch_size = 8
    # 加载数据集
    data_set = DataSet.H5Dataset(dataset)
    image_shape = data_set[0].shape
    assert image_shape[0] == 3 and image_shape[1] == image_shape[2]
    image_size = image_shape[1]
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             drop_last=True)
    
    # 创建文件夹
    dir_model = os.path.join(save, "ddpm_weights")
    os.makedirs(dir_model, exist_ok=True)

    print(f"图像大小为{image_size}")
    model = network.Diffusion(
        image_size=image_size
    )

    model: network.Diffusion = torch.compile(model, fullgraph=True, disable=False) # noqa

    if weight is not None:
        model.load_state_dict(torch.load(weight))

    model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    # 训练
    _train(model, data_loader, optimizer, epoch, dir_model)

    torch.save(model.state_dict(), os.path.join(dir_model, "diffusion.pth"))

    pass
