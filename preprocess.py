import argparse

import cv2
import os

import h5py
import numpy
import tqdm


def head_crop(image: numpy.ndarray, target_size: tuple[int, int]) -> numpy.ndarray:
    ratio = target_size[0] / target_size[1]
    height, width, _ = image.shape
    if width / height > ratio:
        # 原图像太宽
        target_width = round(height * ratio)
        x_start = width // 2 - target_width // 2
        image = image[:, x_start: x_start + target_width, :]
    else:
        # 原图像太高， 从0高度开始，让脑袋尽量完整
        target_height = round(width / ratio)
        y_start = 0
        image = image[y_start: y_start + target_height, ...]

    image = cv2.resize(image, target_size)
    # 从0-255映射到[-1, 1]
    image: numpy.ndarray = numpy.multiply(image, 2 / 255, dtype=numpy.single) - 1.0
    # cv2.imshow("display", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    image = image.transpose((2, 0, 1))
    return image


def center_crop(im, new_shape: tuple[int, int]):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    height, width = im.shape[:2]  # current shape [height, width]
    # Scale ratio (new / old)
    r = max(new_shape[0] / height, new_shape[1] / width)

    # Compute padding
    new_uncrop = round(width * r), round(height * r)

    if (width, height) != new_uncrop:  # resize
        im = cv2.resize(im, new_uncrop, interpolation=cv2.INTER_LINEAR)
    dh = new_uncrop[1] - new_shape[1]
    dw = new_uncrop[0] - new_shape[0]

    top, bottom = dh // 2, new_shape[1] + (dh // 2)  # divide padding into 2 parts
    left, right = dw // 2, new_shape[0] + (dw // 2)
    return im[top:bottom, left:right]


def run(
        source: str,
        target: str,
        target_size: int,
):
    data = list(map(lambda x: os.path.join(source, x), os.listdir(source)))
    os.makedirs(os.path.dirname(target), exist_ok=True)

    image_count = len(data)
    with h5py.File(target, "w") as h5f:
        images: h5py.Dataset = h5f.create_dataset("image", (image_count, 3, target_size, target_size), dtype=numpy.uint8)
        for i, d in enumerate(tqdm.tqdm(data, total=image_count, desc="预处理图像")):
            img = cv2.imread(d)
            img = center_crop(img, (target_size, target_size)).transpose(2, 0, 1)
            images.write_direct(numpy.ascontiguousarray(img), dest_sel=i)


if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('source', type=str)
    opt.add_argument('target', type=str)
    opt.add_argument('--size', type=int, default=128, dest='target_size')
    run(**vars(opt.parse_args()))
