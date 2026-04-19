import argparse
import os
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader

from ..common import dataset, utils
from .flow import FlowMatch


def _train(model: FlowMatch, train_data_path: str, epoch: int, batch_size: int):

    torch.set_float32_matmul_precision("medium")

    data_set = dataset.ImageDataset(train_data_path)

    data_image_size = data_set[0].shape
    assert (
        data_image_size[0] == 3
        and data_image_size[1] == data_image_size[2] == model.hparams.image_size
    ), f"模型要求图像大小为{model.hparams.image_size}，但数据集中为{data_image_size}"
    train_data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    # 加载数据集
    print(f"图像大小为{data_image_size}，共{len(data_set)}张")

    save_dir = os.path.join("runs", f"{model.name()}-v")
    save_dir = utils.auto_increase_dir(save_dir)

    # 保存模型的回调
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,  # 保存最好的1个模型（可保存多个，设为-1保存全部）
        monitor="train_loss",  # 监控训练损失
        mode="min",  # 监控指标变小代表模型变好
        save_weights_only=False,  # 设为True则只保存模型权重，文件更小但无法恢复训练
        # --- 保存在哪 (Where) ---
        dirpath=save_dir,  # 保存目录
        filename="{epoch:02d}-{train_loss:.2f}",  # 文件名（支持格式化）
        # --- 何时保存 (When) ---
        every_n_epochs=1,  # 每1个epoch保存一次
        save_last=True,  # 额外保存一个'last.ckpt'文件
        enable_version_counter=False,  # 覆盖旧模型
    )

    loggers = [
        TensorBoardLogger("lightning_logs", name=model.name(), default_hp_metric=False),
        CSVLogger(save_dir, name=None, version=""),
    ]

    trainer = Trainer(
        max_epochs=epoch,
        logger=loggers,
        precision="16-mixed",
        benchmark=True,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_data_loader)


def main():
    opt = argparse.ArgumentParser()

    opt.add_argument('--data', type=str, default="/media/panpan/datasets/anime_faces_128")
    opt.add_argument('--epoch', type=int, default=32)
    opt.add_argument('--compile', action="store_true")

    opt.add_argument('--image_size', type=int, default=128)
    opt.add_argument('--batch_size', type=int, default=32)

    args = opt.parse_args()

    torch.multiprocessing.set_start_method("spawn")

    model = FlowMatch(
        image_size=args.image_size
    )
    model: FlowMatch = torch.compile(model, disable=not args.compile)  # noqa

    _train(model, args.data, epoch=args.epoch, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
