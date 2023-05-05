import os
import torch

import numpy
from torch.utils.data import Dataset
from torchvision import datasets


class DiskDataSet(Dataset):
    files: [str]
    device: torch.device

    def __init__(self, path: str, device: str):
        super().__init__()
        self.device = torch.device(device)
        self.files = list(map(lambda f: os.path.join(path, f), os.listdir(path)))
        assert len(self.files) != 0

    def __getitem__(self, index):
        return torch.from_numpy(numpy.load(self.files[index])).to(self.device)

    def __len__(self):
        return len(self.files)

class Cifar10DataSet(Dataset):
    device: torch.device
    data: torch.Tensor

    def __init__(self, path: str, device: str):
        super().__init__()
        self.device = torch.device(device)
        cifar10 = datasets.CIFAR10(path, train=True, download=True)
        idx = cifar10.class_to_idx["horse"]
        single_class_data: numpy.ndarray = cifar10.data[numpy.array(cifar10.targets) == idx]
        # for i in range(100):
        #     cv2.imshow("Display", single_class_data[i, ...])
        #     cv2.waitKey()
        single_class_data = numpy.ascontiguousarray(single_class_data[..., ::-1].transpose((0, 3, 1 ,2)).astype(numpy.single))
        self.data = torch.from_numpy(single_class_data).to(device) * (2 / 255) - 1.0

    def __getitem__(self, index):
        return self.data[index, ...]

    def __len__(self):
        return self.data.shape[0]