import os
import cv2
import torch
from common import DataSet, utils
from vae import vae
from torch.utils.data.dataloader import DataLoader


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
    image_shape = data_set[0].shape
    assert image_shape[0] == 3 and image_shape[1] == image_shape[2]
    image_size = image_shape[1]
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             drop_last=True)

    print(f"图像大小为{image_size}，共{len(data_set)}张")
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    # 训练
    vae.train(model, data_loader, optimizer, epoch, save_path)


if __name__ == '__main__':
    image_size = 64
    model = vae.VAE(
        image_size=image_size,
        reg_weight=0.005,
    )
    model: vae.VAE = torch.compile(model, fullgraph=True, disable=False) # noqa

    save_path = f"./run/weights/vae_reg{model.reg_weight}"
    # model.load_state_dict(torch.load(os.path.join(save_path, "vae.pth"), weights_only=True))
    train(model, "./run/datasets/anime_faces64.h5", save_path=save_path, epoch=32)
    
    #    
    count = 16
    images = vae.sample(model, 16)
    image_dir = utils.auto_increase_dir(f"run/result/vae_reg{model.reg_weight}")
    for i in range(count):
        cv2.imwrite(os.path.join(image_dir, f"gen{i}.png"), images[i, ...])

    