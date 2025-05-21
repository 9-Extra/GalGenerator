import os
import cv2
import torch
from common import DataSet, utils
from vae import vae


def train(
        model: vae.VAE,
        dataset: str,
        save_path: str,
        epoch: int
):
    device = torch.device("cuda")
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    
    batch_size = 128
    # 加载数据集
    data_set = DataSet.H5Dataset(dataset)
    print(f"图像大小为{image_size}，共{len(data_set)}张")
    
    trainer = vae.VAETrainer(epoch, batch_size, 100, save_path, device)
    # 训练
    trainer.train(model, data_set)


if __name__ == '__main__':
    image_size = 64
    model = vae.VAE(
        image_size=image_size,
        kl_weight=0.01,
        latent_dim=128,
    )
    model: vae.VAE = torch.compile(model, fullgraph=True, disable=False) # noqa

    save_path = f"./run/weights/vae_reg{model.kl_weight}"
    model.load_state_dict(torch.load(os.path.join(save_path, "vae.pth"), weights_only=True))
    train(model, "./run/datasets/anime_faces64.h5", save_path=save_path, epoch=32)
    
    #    
    count = 16
    images = vae.sample(model, 16)
    image_dir = utils.auto_increase_dir(f"run/result/vae_reg{model.kl_weight}")
    for i in range(count):
        cv2.imwrite(os.path.join(image_dir, f"gen{i}.png"), images[i, ...])

    