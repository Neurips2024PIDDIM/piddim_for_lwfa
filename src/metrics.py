import pandas as pd
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from modules import UNet_conditional
from diffusion import GaussianDiffusionDDPM
from tqdm import tqdm
from PIL import Image
from utils import save_samples
from train import train


class weighted_MSELoss(nn.Module):
    def __init__ (self):
        super().__init__ ()
    def forward (self, input, target, weight):
        return ((input - target)**2) * weight


def compare_avg(dir1, dir2, save=False):
    # Get all image files in the directories
    files1 = [os.path.join(dir1, f) for f in os.listdir(dir1) if f.endswith('.png')]
    files2 = [os.path.join(dir2, f) for f in os.listdir(dir2) if f.endswith('.png')]

    # Compute the average image for each directory
    avg1 = np.mean([np.array(Image.open(f)) for f in files1], axis=0).astype(np.uint8)
    avg2 = np.mean([np.array(Image.open(f)) for f in files2], axis=0).astype(np.uint8)

    # Save and show the average images
    if save:
        Image.fromarray(avg1).save('average1.jpg')
        Image.fromarray(avg2).save('average2.jpg')
        print('Average images saved as average1.jpg and average2.jpg')
    return avg1, avg2


def sample_all(root="models", result_dir="results/transfer_withgain_512_valid", device='cuda', n=30):
    dir_list = [x for x in os.listdir(root) if x.startswith('no_')]
    settings = pd.read_csv("params.csv", engine='python')[["E","P","ms"]]
    for subdir in tqdm(dir_list):
        exp_number = int(subdir.split('_')[1])
        E = settings.loc[exp_number - 1]['E']
        P = settings.loc[exp_number - 1]['P']
        ms = settings.loc[exp_number - 1]['ms']
        model = UNet_conditional(img_width=128, img_height=64, feat_num=3, device=device).to(device)
        ckpt = torch.load(os.path.join(root, subdir, 'ema_ckpt.pt'), map_location=device)
        model.load_state_dict(ckpt)
        diffusion = GaussianDiffusionDDPM(img_width=128, img_height=64, device=device, noise_steps=700)
        y = torch.Tensor([E,P,ms]).to(device).float().unsqueeze(0) # parameter vector
        x = diffusion.sample(model, n, y, cfg_scale=5, resize=[256, 512])
        res_path = os.path.join(result_dir, subdir)
        os.mkdir(res_path)
        save_samples(x, res_path)


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.device = "cuda:0"
    args.features = ["E", "P", "ms"]
    args.real_size = (256, 512)
    args.epochs = 301
    args.noise_steps = 700
    args.beta_end = 0.02
    args.batch_size = 6
    args.image_height = 64
    args.image_width = 128
    args.features = ["E","P","ms"]
    args.dataset_path = r"with_gain"
    args.csv_path = "params.csv"
    args.lr = 1e-3
    args.grad_acc = 1
    args.sample_freq = 0
    args.sample_size = 0
    args.electron_pointing_pixel = 62

    settings = pd.read_csv(args.csv_path, engine='python')[args.features]

    experiments = os.listdir(args.dataset_path)

    for experiment in sorted(experiments, key=lambda x: int(x)):
        args.exclude = [os.path.join(args.dataset_path, experiment)]
        args.run_name = "no_" + experiment
        row = settings.loc[[int(experiment) - 1], args.features]
        args.sample_settings = row.values.tolist()[0]

        model = UNet_conditional(img_width=128, img_height=64, feat_num=3, device=args.device).to(args.device)
        ckpt = torch.load("models/transfered.pt", map_location=args.device)
        model.load_state_dict(ckpt)
        train(args, model)


if __name__ == "__main__":
    main()