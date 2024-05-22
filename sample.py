import torch
import os
from src.diffusion import SpacedDiffusion
from src.modules import UNet_conditional
from src.utils import save_samples
import itertools
import numpy as np
import csv
from metrics import cosine_step_schedule
from metrics import create_sections_list

from tqdm import tqdm


def sample(n, batch_size, E, P, ms, section_counts, cfg, folder, device, model_path, noise_steps=850):
    diffusion = SpacedDiffusion(beta_start=1e-4, beta_end=0.02, section_counts=section_counts, noise_steps=noise_steps, img_width=128, img_height=64, device=device)
    y = torch.Tensor([E,P,ms]).to(device).float().unsqueeze(0) # parameter vectormodel = UNet_conditional(img_width=128, img_height=64, feat_num=3, device=device).to(device)
    model = UNet_conditional(img_width=128, img_height=64, feat_num=3, device=device).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    total = 0
    while total != n:
        if total + batch_size > n:
            add = n - total
        else:
            add = batch_size
        x = diffusion.ddim_sample_loop(model, y, cfg_scale=cfg, resize=[256, 512], n=add, eta=1, device=device, gain=50)
        save_samples(x, folder, start_index=total)
        total += add


def main(P_range, E_range, ms_range, n, batch_size, section_counts, cfg, root, device, model_path, noise_steps=850):
    i = 0
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, 'params.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        headers = ['path', 'P', 'E', 'ms']
        # Write the header row
        writer.writerow(headers)
        for P, E, ms in tqdm(itertools.product(P_range, E_range, ms_range)):
            folder = os.path.join(root, str(i))
            os.makedirs(folder, exist_ok=True)
            sample(n, batch_size, E, P, ms, section_counts, cfg, folder, device, model_path, noise_steps)
            writer.writerow([folder, str(P), str(E), str(ms)])
            i += 1


if __name__ == "__main__":
    main([5], [40, 45, 50], [20], 16, 8, [15], 1, "e_test_cossched", "cuda:3", "models/cossched/ema_ckpt.pt", noise_steps=1000)
    main([5], [40, 45, 50], [20], 16, 8, create_sections_list(10, 25, cosine_step_schedule), 1, "e_test_nophys", "cuda:3", "models/nophys/ema_ckpt.pt", noise_steps=1000)