from pathlib import Path
import torch
import cv2
from common import DataSet
from gan import dcgan

def main():
    device = torch.device("cuda")
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    
    # 加载数据集
    data_set = DataSet.H5Dataset("./run/datasets/anime_faces64.h5")
    # data_set.show()
    
    batch_size = 256
    num_steps = len(data_set) * 20 # epoch
    save_path = Path("./run/weights/dcgan")
    
    generator = dcgan.Generator(128, 64)
    generator.load_state_dict(torch.load(save_path / "dcgan.pth"))
    
    trainer = dcgan.Trainer(generator, data_set, save_path, device, batch_size, num_steps)
    
    trainer.train()
    
    generator.eval()
    
    images = generator.generate(16).permute(0, 2, 3, 1).numpy(force=True)
    print(images.shape)
    for img in images:
        cv2.imshow("image", img)    
        cv2.waitKey()
    
    torch.save(generator.state_dict(), save_path / "dcgan.pth")
    
    pass



if __name__ == '__main__':
    main()