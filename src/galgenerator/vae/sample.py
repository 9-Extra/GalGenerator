import argparse
import os
import cv2
import torch

from .vae import VAE
from ..common import utils


def main():
    opt = argparse.ArgumentParser()
    
    opt.add_argument('--model', type=str, required=True)
    opt.add_argument('--num', type=int, default=32)
    opt.add_argument("--batch", type=int, default=128)
    opt.add_argument("--device", type=str, default="cuda")
    opt.add_argument("--save_dir", type=str, default="runs/gen-vae")
    
    args = opt.parse_args()
    
    device = torch.device(args.device)
    model: VAE = VAE.load_from_checkpoint(args.model, map_location=device, weights_only=False)
    
    image_dir = utils.auto_increase_dir(args.save_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    
    num = args.num
    batch = args.batch
    
    def get_batch_size(num: int, batch: int):
        full_batch = num // batch
        for i in range(full_batch):
            yield i, batch
        
        last = num % batch
        if last != 0:
            yield full_batch, last
    
    with torch.inference_mode():
        for i, batch_size in get_batch_size(num, batch):
            images = (model.sample(batch_size) * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8)
            images = images.numpy(force=True)
            for j in range(batch_size):
                idx = i * batch + j
                cv2.imwrite(os.path.join(image_dir, f"gen{idx}.png"), images[j, ...])
        
    

if __name__ == '__main__':
    main()