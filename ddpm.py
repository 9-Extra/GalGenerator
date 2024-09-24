import argparse
import ddpm.sample
import ddpm.train

if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    subparsers = opt.add_subparsers()
    
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument('dataset', type=str)
    train_parser.add_argument('--weight', type=str, default=None)
    train_parser.add_argument('--save', type=str, default="./run")
    train_parser.add_argument('--epoch', type=int, default=20)
    train_parser.set_defaults(func=ddpm.train.run)
    
    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument('weight', type=str)
    sample_parser.add_argument('--image_size', type=int)
    sample_parser.add_argument('--save_dir', type=str, default='./run/result')
    sample_parser.add_argument('--count', type=int, default=4)
    sample_parser.add_argument('--device', type=str, default="cuda")
    sample_parser.set_defaults(func=ddpm.sample.run)
    
    args = vars(opt.parse_args())
    func = args.pop("func")
    func(**args)