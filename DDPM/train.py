import argparse
import os
import sys
import cv2

import tqdm

import torch
from torch.utils.data import DataLoader
import Diffusion

from common import DataSet

# 前向过程，为图像加上t次噪声，返回加上噪声后的图像和加上的噪声（训练用）
def _forward_process(x_0: torch.Tensor, t: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    noise = torch.randn(x_0.size(), dtype=x_0.dtype, device=x_0.device) # 生成高斯噪声
    alpha_over_line_sqrt_t = torch.index_select(Diffusion.alpha_over_line_sqrt, 0, t).view((-1, 1, 1 ,1))
    one_sub_alpha_over_line_sqrt_t = torch.index_select(Diffusion.one_sub_alpha_over_line_sqrt, 0, t).view((-1, 1, 1 ,1))
    x_t = alpha_over_line_sqrt_t * x_0 + one_sub_alpha_over_line_sqrt_t * noise

    return x_t, noise

def _train(model: Diffusion.Diffusion, data: DataLoader, optimizer: torch.optim.Optimizer, epoch: int, device):
    print("Start Training")
    for e in range(1, epoch + 1):
        total_loss = 0.0
        p = tqdm.tqdm(data, leave=True, file=sys.stdout)
        for batch in p:
            t = torch.randint(0, Diffusion.T, [data.batch_size], device=device, dtype=torch.int) # 随机选一层[0..T)，比论文中的t小1

            x_t, noise = _forward_process(batch, t)

            optimizer.zero_grad()
            output = model.forward(x_t, torch.index_select(Diffusion.position_encoding, 0, t))

            # for i in range(batch.shape[0]):
            #      print("Time: ", t[i].item())
            #      # cv2.imshow(f"Src", batch.numpy(force=True).transpose((0, 2, 3, 1))[i, :, :, ::-1])
            #      cv2.imshow(f"Noised", x_t.numpy(force=True).transpose((0, 2, 3, 1))[i, :, :, ::1])
            #      cv2.imshow(f"Noise", noise.numpy(force=True).transpose((0, 2, 3, 1))[i, :, :, ::1])
            #      cv2.imshow(f"Predicted", output.numpy(force=True).transpose((0, 2, 3, 1))[i, :, :, ::1])
            #      cv2.waitKey()

            loss = torch.nn.functional.mse_loss(noise, output) # 优化估计出的噪声
            loss.backward()
            optimizer.step()

            l = loss.item()
            total_loss += l
            p.set_postfix({"loss": l})

        print(f"Finished epoch {e}/{epoch}, mean-loss: {total_loss / len(data)}")

        if e % 10 == 0:
            torch.save(model.state_dict(), f"model/Diffusion_{e}.pth")

    pass


def run(
        source: str,
        epoch: int,
        device: str
):
    batch_size = 16

    os.makedirs("model", exist_ok=True)

    model = Diffusion.Diffusion()

    Diffusion.to_device(device)
    model.to(device).train()

    # 预热
    t = torch.randint(0, Diffusion.T, [batch_size], device=device, dtype=torch.int)
    t = torch.index_select(Diffusion.position_encoding, 0, t)
    model.forward(torch.rand([batch_size, 3, *Diffusion.IMAGE_SIZE], device=device), t)
    model.zero_grad()
    # 加载数据集
    # data_loader = DataLoader(dataset= DataSet.DiskDataSet(source, device), batch_size=batch_size, shuffle=True, num_workers=0)
    data_loader = DataLoader(dataset= DataSet.Cifar10DataSet("./cifar10", device), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    # 训练
    _train(model, data_loader, optimizer, epoch, device)

    torch.save(model.state_dict(), f"model/Diffusion.pth")

    pass


if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('source', type=str)
    opt.add_argument('--epoch', type=int, default=200)
    opt.add_argument('--device', type=str, default="cuda")
    run(**vars(opt.parse_args()))