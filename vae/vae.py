from torch.utils.data.dataloader import DataLoader
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

    
    def __init__(self, image_size: int, latent_dim: int = 128, reg_weight: float=0.05) -> None:
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
            self._Chunk() # 分割为均值和方差
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 256 * (image_size // 16) ** 2),
            torch.nn.Unflatten(1, (256, image_size // 16, image_size // 16)),
            torch.nn.Conv2d(256, 256, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
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
            torch.nn.Conv2d(16, 3, 1)
        )
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def _latent_sample(self, latent_mean: torch.Tensor, latent_logvar: torch.Tensor) -> torch.Tensor:
        latent_std = torch.exp(latent_logvar / 2) # 标准差
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
        reg_loss = torch.mean((torch.exp(latent_logvar) - (1 + latent_logvar) + torch.square(latent_mean)))
        
        return img_loss + reg_loss * self.reg_weight
    
    def sample(self, count: int) -> torch.Tensor:
        assert not self.training
        
        guass_noise_latent = torch.randn((count, self.latent_dim), device=self.device)
        reconstruct = self.decoder(guass_noise_latent).clamp_(min=0, max=1)
        return reconstruct
    
    
def _check_size(vae: VAE):
    virtual_input = torch.randn((2, 3, vae.image_size, vae.image_size), device=vae.device)
    latent_mean, latent_logvar = vae.encoder(virtual_input)
    assert latent_mean.shape == latent_logvar.shape == (2, vae.latent_dim), f"{latent_mean.shape=} {latent_logvar.shape=} {vae.latent_dim=}"
    
    guass_noise_latent = torch.randn((2, vae.latent_dim), device=vae.device)
    reconstruct = vae.decoder(guass_noise_latent)
    assert reconstruct.shape == virtual_input.shape, f"{reconstruct.shape=} {virtual_input.shape=}"
    

def train(
        model: VAE,
        data: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        save_dir: str
):
    os.makedirs(save_dir, exist_ok=True)
    
    model.train()
    device = next(model.parameters()).device
    print("Start Training")
    for e in range(1, epoch + 1):
        running_loss = []
        for batch in tqdm.tqdm(data):
            batch = batch.to(device, non_blocking=True) / 255
            
            optimizer.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            optimizer.step()

            running_loss.append(loss)  # 保存引用，不获取loss的值以降低传输延迟

        pass  # 一个epoch结束

        mean_loss = torch.mean(torch.tensor(running_loss))
        print(f"Finished epoch {e}/{epoch}, mean-loss: {mean_loss}")

        if e % 100 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"vae_{e}.pth"))
            
    torch.save(model.state_dict(), os.path.join(save_dir, "vae.pth"))

    pass


def sample(model: VAE, count: int):
    model.eval()
    images =  (model.sample(count) * 255).permute((0, 2, 3, 1)).numpy(force=True)
    return images


    

if __name__ == '__main__':
    vae = VAE(64)
    _check_size(vae)