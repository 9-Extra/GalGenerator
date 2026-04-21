# 流匹配模型
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.nn import Module

from galgenerator.common.dic_models import DiC, DiC_default
from galgenerator.common.unet import UNet


class FlowMatch(LightningModule):
    image_size: int

    omega: torch.Tensor

    unet: DiC
    mlp: Module

    def __init__(
        self,
        image_size: int,
        depth: list[int]=[6, 6, 5, 6, 6], 
        hidden_size: int = 64,
        emb_dim: int = 256,
    ):
        super().__init__()

        self.save_hyperparameters()

        e = (
            2 / emb_dim * torch.arange(emb_dim // 2, dtype=torch.float32)
        )  # omega的指数部分
        self.register_buffer("omega", 10 ** (-e))  # 时间位置编码的频率

        time_embed_dim = emb_dim
        # 先对位置编码使用mlp进行映射
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, time_embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_embed_dim, time_embed_dim),
            torch.nn.SiLU(),
        )
        # 内部就是个UNet
        # self.unet = UNet(image_size, 3, time_embed_dim)
        self.unet = DiC_default(depth=depth, hidden_size=hidden_size, in_channels=3, input_size=image_size, max_period=10, learn_sigma=False)

    def name(self) -> str:
        return f"FLOW-s{self.hparams.image_size}"

    @torch.no_grad()
    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        流匹配模型支持0-1时间内的连续时间，这使得它的时间位置编码无法完全预计算
        将模型输入的t [batch_size]信息编码到输入的图像tensor中，借用了transformer的位置编码方法
        返回的位置编码形状为[batch_size, emb_dim]，对每个时间都有一个长度为emb_dim的向量
        """
        batch_size = t.shape[0]
        emb_dim = self.hparams.emb_dim
        encoding = torch.empty((batch_size, emb_dim), dtype=t.dtype, device=t.device)
        omega_t = t[:, None] * self.omega[None, :]
        encoding[:, 0::2] = torch.sin(omega_t)  # 偶数位
        encoding[:, 1::2] = torch.cos(omega_t)  # 奇数位
        return encoding

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # embed = self.mlp(self.timestep_embedding(t))
        x = self.unet(x, t)
        return x

    def loss(self, x_1: torch.Tensor) -> torch.Tensor:
        """
        通过目标图像计算损失
        :param x_1: 目标图像
        :param t: 前向的时间
        :return: CFM损失
        """
        batch_size = x_1.shape[0]
        # [0, 1]内随机采样时间步
        t = torch.rand((batch_size,), device=x_1.device, dtype=x_1.dtype)
        x_0 = torch.randn(
            x_1.shape, dtype=x_1.dtype, device=x_1.device
        )  # 为每张图像生成独立的高斯噪声

        t_expand = t.view(batch_size, 1, 1, 1)  # 满足广播规则
        x_t = t_expand * x_1 + (1 - t_expand) * x_0
        v_pred = self.forward(x_t, t)  # 预测速度
        v_gt = x_1 - x_0  # 真实速度

        loss = torch.nn.functional.mse_loss(v_pred, v_gt)

        return loss

    @torch.no_grad()
    def sample(self, batch_size: int, step: int = 50) -> torch.Tensor:
        """
        采样图像，一次生成batch_size个，得到的图像像素取值范围为[-1, 1]，还需要再映射一下
        """
        device = self.device

        import tqdm

        input_size = (batch_size, 3, self.image_size, self.image_size)
        x_t = torch.randn(input_size, device=device)  # 从高斯噪声开始
        for t in tqdm.tqdm(range(step), total=step, desc="采样图像"):
            # 当前时刻t
            t = torch.tensor(t / step, device=device, dtype=torch.float32).expand(
                (batch_size,)
            )
            v_pred = self.forward(x_t, t)  # 预测速度

            x_t += v_pred * (1 / step)  # eular法，步进求解方程

        return x_t.clip_(-1.0, 1.0)

    # --------------------------训练-----------------------------
    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        # AMP兼容性：保持float32，AMP会自动处理半精度转换
        # 流匹配也需要将图像归一化到[-1, 1]
        return batch.to(dtype=torch.float32) / 127.5 - 1.0

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        loss = self.loss(batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4)
        return optimizer

    # --------------------------推理-----------------------------
    def predict_step(self, batch):
        return self.sample(len(batch))


if __name__ == "__main__":
    ddpm = FlowMatch(64)
