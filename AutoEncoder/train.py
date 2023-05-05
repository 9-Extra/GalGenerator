import argparse
import sys

import tqdm

import torch
from torch.utils.data import DataLoader
from AutoEncoder import Autoencoder

from common.DataSet import DataSet

def _train(model: torch.nn.Module, data: DataLoader, optimizer: torch.optim.Optimizer, epoch: int):
    for e in range(1, epoch + 1):
        total_loss = 0.0
        p = tqdm.tqdm(data, leave=True, file=sys.stdout)
        for inputs in p:
            optimizer.zero_grad()

            output = model.forward(inputs)

            loss = torch.nn.functional.mse_loss(inputs, output)
            loss.backward()
            optimizer.step()

            l = loss.item()
            total_loss += l
            p.set_postfix({"loss": l})

        print(f"Finished epoch {e}/{epoch}, mean-loss: {total_loss / len(data)}")


def run(
        source: str,
        image_size: (int, int),
        epoch: int,
        device: str
):
    batch_size = 4

    model = Autoencoder()

    model.to(device).train()

    model.forward(torch.rand([batch_size, 3, *image_size], device=device)) # 预热

    data_loader = DataLoader(dataset=DataSet(source, device), batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练
    _train(model, data_loader, optimizer, epoch)

    torch.save(model.state_dict(), f"Autoencoder.pth")

    pass


if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('source', type=str)
    opt.add_argument('--size', type=int, nargs=2, default=(640, 640), dest='image_size')
    opt.add_argument('--epoch', type=int, default=20)
    opt.add_argument('--device', type=str, default="cuda")
    run(**vars(opt.parse_args()))