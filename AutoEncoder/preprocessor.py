import argparse
import random

import cv2
import os
import numpy
import tqdm
import multiprocessing


def preprocess_image(image: numpy.ndarray, target_size: (int, int)) -> numpy.ndarray:
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
    image: numpy.ndarray = numpy.multiply(image, 1 / 255, dtype= numpy.single)
    # cv2.imshow("display", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    image = image.transpose((2, 0, 1))
    return image


def _preprocess_job(path: str, target: str, target_size):
    image = preprocess_image(cv2.imread(path), target_size)
    numpy.save(target, image, allow_pickle=False, fix_imports=False)

def run(
        source: str,
        target: str,
        count: int,
        target_size: (int, int),
):
    os.makedirs(target, exist_ok=True)

    image_list = list(filter(lambda name: name.endswith(".jpg") or name.endswith(".png"), os.listdir(source)))
    if count != -1 and count < len(image_list):
        image_list = random.sample(image_list, count)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    process = tqdm.tqdm(total=len(image_list))
    for i, file in enumerate(image_list):
        pool.apply_async(_preprocess_job,
                         [os.path.join(source, file), os.path.join(target, f"train{i}.npy"), target_size],
                         callback=lambda _: process.update()
                         )
    pass

    pool.close()
    pool.join()
    pool.terminate()



if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('source', type=str)
    opt.add_argument('target', type=str)
    opt.add_argument('--size', type=int, nargs=2, default=(640, 640), dest='target_size')
    opt.add_argument('--count', type=int, default=-1)
    run(**vars(opt.parse_args()))
