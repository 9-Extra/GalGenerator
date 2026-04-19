import argparse
import cv2
import os

import lmdb
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

    if image.shape[:2] != target_size:  
        image = cv2.resize(image, target_size)
    image = image.transpose((2, 0, 1))
    return image


def run(
        source: str,
        target: str,
        target_size: int,
        quality: int = 95,
):
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
    data = sorted([
        os.path.join(source, x)
        for x in os.listdir(source)
        if os.path.splitext(x.lower())[1] in valid_ext
    ])

    target_dir = os.path.dirname(target)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    image_count = len(data)
    # map_size 是地址空间上限，实际磁盘占用取决于写入量。64GB 足够绝大多数单机场景。
    map_size = 64 * 1024 * 1024 * 1024
    env = lmdb.open(target, map_size=map_size)

    with env.begin(write=True) as txn:
        for i, d in enumerate(tqdm.tqdm(data, total=image_count, desc="预处理图像")):
            img = cv2.imread(d, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = head_crop(img, (target_size, target_size))
            # head_crop 输出 CHW，编码需要转回 HWC
            img_hwc = img.transpose((1, 2, 0))
            success, buf = cv2.imencode(".webp", img_hwc, [cv2.IMWRITE_WEBP_QUALITY, quality])
            if not success:
                continue
            key = f"{i:08d}".encode("ascii")
            txn.put(key, buf.tobytes())
        txn.put(b"__len__", str(i + 1).encode("ascii"))

    env.close()


if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('source', type=str)
    opt.add_argument('target', type=str)
    opt.add_argument('--size', type=int, default=128, dest='target_size')
    opt.add_argument('--quality', type=int, default=95, dest='quality')
    run(**vars(opt.parse_args()))
