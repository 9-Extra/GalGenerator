import lmdb
import numpy
import pathlib
import cv2
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, path: str):
        assert pathlib.Path(path).exists()

        self.path = path
        self.env = None
        # 临时打开一次读取长度，然后立即关闭；不保留 env 引用
        env = lmdb.open(self.path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            length_bytes = txn.get(b"__len__")
            if length_bytes is None:
                raise ValueError("LMDB database missing __len__ key")
            self._len = int(length_bytes.decode("ascii"))
        env.close()

    def _get_env(self):
        """延迟初始化：每个 worker 进程第一次读取时才会打开自己的 env。"""
        if self.env is None:
            self.env = lmdb.open(
                self.path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        return self.env

    def __getstate__(self):
        """排除不可 pickle 的 lmdb.Environment，反序列化后由 _get_env 重新打开。"""
        state = self.__dict__.copy()
        state["env"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.env = None

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        key = f"{idx:08d}".encode("ascii")
        with self._get_env().begin(write=False) as txn:
            buf = txn.get(key)
        if buf is None:
            raise IndexError(f"Index {idx} not found in dataset")
        # cv2.imdecode 返回 HWC BGR uint8，与原始 cv2.imread 行为一致
        img = cv2.imdecode(numpy.frombuffer(buf, dtype=numpy.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to decode image at index {idx}")
        img = img.transpose((2, 0, 1))
        return img

    def __getitems__(self, indices: list):
        imgs = []
        with self._get_env().begin(write=False) as txn:
            for idx in indices:
                key = f"{idx:08d}".encode("ascii")
                buf = txn.get(key)
                if buf is None:
                    raise IndexError(f"Index {idx} not found in dataset")
                img = cv2.imdecode(numpy.frombuffer(buf, dtype=numpy.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"Failed to decode image at index {idx}")
                imgs.append(img.transpose((2, 0, 1)))
        return imgs

    def show(self):
        print(f"图像大小为 (3, H, W)，共{self._len}张")
        for i in range(self._len):
            img = self.__getitem__(i)
            cv2.imshow("image", img.transpose(1, 2, 0))
            cv2.waitKey()
