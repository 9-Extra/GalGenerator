import argparse
import os
import sys
import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import Diffusion

from common import DataSet


def _train(
        model: Diffusion.Diffusion,
        data: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int, device,
        save_dir: str,
        writer: SummaryWriter
):
    print("Start Training")
    for e in range(1, epoch + 1):
        total_loss = 0.0
        p = tqdm.tqdm(data, leave=True, file=sys.stdout)
        for batch in p:
            t = torch.randint(0, model.total_timestep, [data.batch_size], device=device, dtype=torch.int)  # 随机选一层[0..T)，比论文中的t小1

            x_t, noise = model.forward_process(batch, t)

            optimizer.zero_grad()
            output = model.forward(x_t, torch.index_select(model.position_encoding, 0, t))

            # for i in range(batch.shape[0]):
            #      print("Time: ", t[i].item())
            #      # cv2.imshow(f"Src", batch.numpy(force=True).transpose((0, 2, 3, 1))[i, :, :, ::-1])
            #      cv2.imshow(f"Noised", x_t.numpy(force=True).transpose((0, 2, 3, 1))[i, :, :, ::1])
            #      cv2.imshow(f"Noise", noise.numpy(force=True).transpose((0, 2, 3, 1))[i, :, :, ::1])
            #      cv2.imshow(f"Predicted", output.numpy(force=True).transpose((0, 2, 3, 1))[i, :, :, ::1])
            #      cv2.waitKey()

            loss = torch.nn.functional.mse_loss(noise, output)  # 优化估计出的噪声
            loss.backward()
            optimizer.step()

            l = loss.item()
            total_loss += l
            p.set_postfix({"loss": l})

        pass  # 一个epoch结束

        mean_loss = total_loss / len(data)
        writer.add_scalar("mean-loss", mean_loss, e)

        print(f"Finished epoch {e}/{epoch}, mean-loss: {mean_loss}")

        if e % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"Diffusion_{e}.pth"))

    pass


def run(
        source: str,
        weight: str,
        save: str,
        epoch: int,
        device: str
):
    batch_size = 16

    # 创建文件夹
    os.makedirs(save, exist_ok=True)
    dir_model = os.path.join(save, "model")
    os.makedirs(dir_model, exist_ok=True)

    model = Diffusion.Diffusion(
        image_size=32
    )

    if weight is not None:
        model.load_state_dict(torch.load(weight))

    model.to(device).train()

    # 预热
    t = torch.randint(0, model.total_timestep, [batch_size], device=device, dtype=torch.int)
    t = torch.index_select(model.position_encoding, 0, t)
    model.forward(torch.rand([batch_size, 3, model.image_size, model.image_size], device=device), t)

    # 输出到tensorboard
    writer = SummaryWriter(os.path.join(save, "record"))
    # 输出网络
    writer.add_graph(model, [torch.rand([batch_size, 3, model.image_size, model.image_size], device=device), t])

    # 加载数据集
    # data_set = DataSet.DiskDataSet(source, device)
    data_set = DataSet.Cifar10DataSet("./cifar10", device)
    assert data_set[0].shape == torch.Size([3, model.image_size, model.image_size])
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    # 训练
    _train(model, data_loader, optimizer, epoch, device, dir_model, writer)

    torch.save(model.state_dict(), os.path.join(dir_model, "Diffusion.pth"))

    pass


if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('source', type=str)
    opt.add_argument('--weight', type=str, default=None)
    opt.add_argument('--save', type=str, default="./run")
    opt.add_argument('--epoch', type=int, default=200)
    opt.add_argument('--device', type=str, default="cuda")
    run(**vars(opt.parse_args()))
