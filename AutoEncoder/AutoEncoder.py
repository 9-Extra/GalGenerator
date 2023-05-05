# AutoEncoder自编码器
import argparse
import torch
from torch.nn import Module
from common.common_layers import Conv2
import torchsummary

class Autoencoder(Module):
    encoder: Module
    decoder: Module

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            Conv2(3, 32),
            torch.nn.MaxPool2d(2),
            Conv2(32, 64),
            torch.nn.MaxPool2d(2),
            Conv2(64, 128),
            torch.nn.MaxPool2d(2),
            Conv2(128, 256),
            torch.nn.MaxPool2d(2),
            Conv2(256, 256)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            Conv2(256, 256),
            torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            Conv2(128, 128),
            torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            Conv2(64, 64),
            torch.nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            Conv2(32, 32),
            torch.nn.Conv2d(32, 3, 1)
        )


    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def run(
        image_size: (int, int),
        device: str
):
    batch_size = 4
    device = torch.device(device)

    auto_encoder = Autoencoder()
    auto_encoder.to(device).train()

    torchsummary.summary(auto_encoder, [[3, *image_size]], batch_size=batch_size)

    pass


if __name__ == '__main__':
    # 仅仅加载模型
    opt = argparse.ArgumentParser()
    opt.add_argument('--size', type=int, nargs=2, default=(640, 640), dest='image_size')
    opt.add_argument('--device', type=str, default="cuda")
    run(**vars(opt.parse_args()))
