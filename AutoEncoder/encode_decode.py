import argparse
import os

import cv2
import numpy
import torch

from common import utils
from AutoEncoder import Autoencoder, preprocessor
import random

# 使用自编码器进行压缩和解压

def _tensor_to_image(t: torch.Tensor) -> numpy.ndarray:
    image: numpy.ndarray = t.numpy()[0] * 255
    image = image.transpose((1, 2, 0))
    return image.astype(numpy.uint8)


def run(
        source: str,
        weight: str,
        save_dir: str,
        count: int,
        image_size: (int, int)
):
    save_dir = utils.auto_increase_dir(save_dir)

    auto_encoder = Autoencoder()
    auto_encoder.load_state_dict(torch.load(weight))
    auto_encoder.eval()

    # load data
    with torch.no_grad():
        input_filenames = random.sample(os.listdir(source), count)
        inputs = []
        for file in input_filenames:
            file: str
            image = torch.from_numpy(
                preprocessor.preprocess_image(cv2.imread(os.path.join(source, file)), image_size)[None, ...])
            i = {"name": file.rpartition('.')[0], "origin": image}
            inputs.append(i)
        pass

        # run
        for i in inputs:
            i["encoded"] = auto_encoder.encoder(i["origin"])
            i["decoded"] = torch.clamp(auto_encoder.decoder(i["encoded"]), 0.0, 1.0)

        # save
        for i in inputs:
            name: str = i["name"]
            cv2.imwrite(os.path.join(save_dir, f"{name}-origin.png"), _tensor_to_image(i["origin"]))
            cv2.imwrite(os.path.join(save_dir, f"{name}-decoded.png"), _tensor_to_image(i["decoded"]))

    pass


if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('source', type=str)
    opt.add_argument('weight', type=str)
    opt.add_argument('--save_dir', type=str, default='./result')
    opt.add_argument('--count', type=int, default=4)
    opt.add_argument('--size', type=int, nargs=2, default=(640, 640), dest='image_size')
    run(**vars(opt.parse_args()))