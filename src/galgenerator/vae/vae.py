import torch
from lightning import LightningModule


class UNet(torch.nn.Module):

    encoder: torch.nn.Module
    decoder: torch.nn.Module

    def __init__(
        self, image_size: int, latent_dim: int = 128, kl_weight: float = 0.05
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

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

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回更多内部信息的前向传播，用于训练"""

        latent_mean, latent_logvar = self.encoder(x).chunk(
            2, dim=1
        )  # 切割为均值和对数方差

        latent_std = torch.exp(
            latent_logvar / 2
        )  # 标准差，理论上是取指数再开平方，优化成除以二再取指数
        latent = latent_mean + latent_std * torch.randn_like(
            latent_mean
        )  # 潜空间随机采样

        y = self.decoder(latent)
        return y, latent_mean, latent_logvar


class VAE(LightningModule):
    unet: UNet

    kl_weight: float

    def __init__(
        self, image_size: int, latent_dim: int = 128, kl_weight: float = 0.05
    ) -> None:
        super().__init__()

        self.save_hyperparameters("image_size", "latent_dim", "kl_weight")
        self.kl_weight = kl_weight

        self.unet = UNet(image_size, latent_dim)
        
    def name(self) -> str:
        return f"VAE-kl{self.hparams.kl_weight}-s{self.hparams.image_size}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)[0]

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        y, latent_mean, latent_logvar = self.unet(x)

        img_loss = torch.nn.functional.mse_loss(x, y)
        # 计算两个高斯分布的KL散度，一个是N(latent_mean, exp(latent_logvar))，一个是标准正态分布
        # 对于一元高斯分布，其KL散度为KL = 0.5 * (Mean ** 2 - log(Var) + Var - 1)
        # 这里将latent中的每一维视为独立的高斯分布，分别求出KL散度后再求平均作为损失
        kl = torch.square(latent_mean) - latent_logvar + torch.exp(latent_logvar) - 1
        kl_loss = torch.mean(kl)

        return img_loss + kl_loss * self.kl_weight

    def sample(self, count: int) -> torch.Tensor:
        guass_noise_latent = torch.randn((count, self.hparams.latent_dim), device=self.device)
        y = self.unet.decoder(guass_noise_latent).clamp_(min=-1, max=1)
        return y

    # --------------------------训练-----------------------------
    def on_after_batch_transfer(self, batch, dataloader_idx: int):        
        # AMP兼容性：保持float32，AMP会自动处理半精度转换
        # 干脆也将图像归一化到[-1, 1]
        return batch.to(dtype=torch.float32) / 127.5 - 1.0

    def training_step(self, batch: torch.Tensor, batch_idx: int):  
        loss = self.loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
    
    # --------------------------推理-----------------------------
    def predict_step(self, batch):
        return self.sample(len(batch))


if __name__ == "__main__":
    vae = VAE(64)
