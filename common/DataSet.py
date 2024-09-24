import h5py
from torch.utils.data import Dataset


class H5Dataset(Dataset):
    def __init__(self, path: str):
        self.h5f = h5py.File(path, "r")
        self.images: h5py.Dataset = self.h5f["image"]

    def __len__(self):
        return self.images.len()

    def __getitem__(self, idx):
        img = self.images[idx]
        return img
