import os

def auto_increase_dir(root: str) -> str:
    if not os.path.isdir(root):
        os.mkdir(root)

    for i in range(0, 1000):
        dir_name = os.path.join(root, str(i))
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
            return dir_name
    else:
        raise RuntimeError("Too much folders!")

class AutoIncreaseDir:
    root: str
    i: int

    def __init__(self, root: str):
        self.root = root
        if not os.path.isdir(root):
            os.mkdir(root)

        for i in range(0, 1000):
            dir_name = os.path.join(root, str(i))
            if not os.path.isdir(dir_name):
                self.i = i
        else:
            raise RuntimeError("Too much folders!")

    def __next__(self):
        if self.i >= 1000:
            raise RuntimeError("Too much folders!")
        dir_name = os.path.join(self.root, str(self.i))
        os.mkdir(dir_name)
        self.i += 1
        yield dir_name

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

