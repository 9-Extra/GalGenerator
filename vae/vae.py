from typing import Optional
from torch.utils.data.dataloader import DataLoader, Dataset
import os
import torch
import tqdm


class VAE(torch.nn.Module):

    encoder: torch.nn.Module
    decoder: torch.nn.Module

    image_size: int
    latent_dim: int
    reg_weight: float

    class _Chunk(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            return x.chunk(2, dim=1)

    def __init__(
        self, image_size: int, latent_dim: int = 128, reg_weight: float = 0.05
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.reg_weight = reg_weight

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, stride=2, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128, 256, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(256 * (image_size // 16) ** 2, latent_dim * 2),
            self._Chunk(),  # 分割为均值和方差
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 256 * (image_size // 16) ** 2),
            torch.nn.Unflatten(1, (256, image_size // 16, image_size // 16)),
            torch.nn.Conv2d(256, 256, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(
                256, 128, 3, stride=2, padding=1, output_padding=1
            ),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 3, 1),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _latent_sample(
        latent_mean: torch.Tensor, latent_logvar: torch.Tensor
    ) -> torch.Tensor:
        latent_std = torch.exp(latent_logvar / 2)  # 标准差
        return latent_mean + latent_std * torch.randn_like(latent_mean)

    def forward(self, x: torch.Tensor):
        latent_mean, latent_logvar = self.encoder(x)

        latent = self._latent_sample(latent_mean, latent_logvar)

        reconstruct = self.decoder(latent)
        return reconstruct

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        latent_mean, latent_logvar = self.encoder(x)
        latent = self._latent_sample(latent_mean, latent_logvar)
        reconstruct = self.decoder(latent)

        img_loss = torch.nn.functional.mse_loss(x, reconstruct)
        # 计算两个高斯分布的KL散度，一个是N(latent_mean, exp(latent_logvar))，一个是标准正态分布
        # 对于一元高斯分布，其KL散度为KL = 0.5 * (Mean ** 2 - log(Var) + Var - 1)
        # 这里将latent中的每一维视为独立的高斯分布，分别求出KL散度后再求平均作为损失
        kl = torch.square_(latent_mean) - latent_logvar + torch.exp(latent_logvar) - 1
        reg_loss = torch.mean(kl)

        return img_loss + reg_loss * self.reg_weight

    def sample(self, count: int) -> torch.Tensor:
        assert not self.training

        guass_noise_latent = torch.randn((count, self.latent_dim), device=self.device)
        reconstruct = self.decoder(guass_noise_latent).clamp_(min=0, max=1)
        return reconstruct


def _check_size(vae: VAE):
    virtual_input = torch.randn(
        (2, 3, vae.image_size, vae.image_size), device=vae.device
    )
    latent_mean, latent_logvar = vae.encoder(virtual_input)
    assert (
        latent_mean.shape == latent_logvar.shape == (2, vae.latent_dim)
    ), f"{latent_mean.shape=} {latent_logvar.shape=} {vae.latent_dim=}"

    guass_noise_latent = torch.randn((2, vae.latent_dim), device=vae.device)
    reconstruct = vae.decoder(guass_noise_latent)
    assert (
        reconstruct.shape == virtual_input.shape
    ), f"{reconstruct.shape=} {virtual_input.shape=}"


class VAETrainer:
    epoch: int
    batch_size: int
    checkpoint_interval: Optional[int]
    save_path: str
    device: torch.device

    def __init__(self, epoch: int, batch_size: int, checkpoint_interval: Optional[int], save_path: str, device: torch.device):
        self.epoch = epoch
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.save_path = save_path
        self.device = device
        
    def build_optimizer(self, model: torch.nn.Module):
        return torch.optim.Adam(model.parameters(), lr=0.0002)

    def train(
        self,
        model: VAE,
        train_data: Dataset
    ):
        # 准备数据集
        image_shape = train_data[0].shape
        assert image_shape[0] == 3 and image_shape[1] == image_shape[2]
        train_data_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        
        # 准备神经网络
        model.to(self.device, non_blocking=True).train()
        
        # 创建优化器
        optimizer = self.build_optimizer(model)
        # 创建保持权重的文件夹
        os.makedirs(self.save_path, exist_ok=True)
        
        print("Start Training")
        for e in range(1, self.epoch + 1):
            running_loss = []
            for batch in tqdm.tqdm(train_data_loader):
                batch = batch.to(self.device, non_blocking=True) / 255

                optimizer.zero_grad()
                loss = model.loss(batch)
                loss.backward()
                optimizer.step()

                running_loss.append(loss)  # 保存引用，不获取loss的值以降低传输延迟

            pass  # 一个epoch结束

            mean_loss = torch.mean(torch.tensor(running_loss))
            print(f"Finished epoch {e}/{self.epoch}, mean-loss: {mean_loss}")

            if e % self.checkpoint_interval == 0:
                torch.save(model.state_dict(), os.path.join(self.save_path, f"vae_{e}.pth"))

        torch.save(model.state_dict(), os.path.join(self.save_path, "vae.pth"))

        pass


def sample(model: VAE, count: int):
    model.eval()
    images = (model.sample(count) * 255).permute((0, 2, 3, 1)).numpy(force=True)
    return images


if __name__ == "__main__":
    vae = VAE(64)
    _check_size(vae)
