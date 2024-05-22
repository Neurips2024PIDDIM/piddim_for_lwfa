import pandas as pd
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from src.modules import UNet_conditional
from src.diffusion import SpacedDiffusion
from tqdm import tqdm
from PIL import Image
from src.utils import save_samples
from train import train
from pathlib import Path
from torch.nn.functional import mse_loss
import csv
import subprocess
from train import cosine_step_schedule
from src.utils import deflection_biexp_calc, calc_spec


def create_sections_list(length, total_sum, schedule_function):
    assert total_sum >= length, "Total sum must be at least equal to the length of the list"

    sigmoid_values = torch.tensor([schedule_function(torch.tensor(i), max_steps=length, k=0.9) for i in range(1, length + 1)])
    normalized_sigmoid_values = sigmoid_values / torch.sum(sigmoid_values)
    available_sum = total_sum - length
    scaled_sigmoid_values = normalized_sigmoid_values * available_sum
    integer_list = (torch.ones(length) + scaled_sigmoid_values).int().tolist()

    current_sum = sum(integer_list)
    difference = total_sum - current_sum
    
    if difference != 0:
        sign = int(difference / abs(difference))
        indices = list(range(length))
        torch.randperm(len(indices)).tolist()

        for i in indices:
            if difference == 0:
                break
            integer_list[i] += sign
            difference -= sign

    return sorted(integer_list)


class weighted_MSELoss(nn.Module):
    def __init__ (self):
        super().__init__ ()
    def forward (self, input, target, weight):
        return ((input - target)**2) * weight


def compare_avg(dir1, dir2, save=False):
    files1 = [os.path.join(dir1, f) for f in os.listdir(dir1) if f.endswith('.png')]
    files2 = [os.path.join(dir2, f) for f in os.listdir(dir2) if f.endswith('.png')]

    avg1 = np.mean([np.array(Image.open(f)) for f in files1], axis=0).astype(np.uint8)
    avg2 = np.mean([np.array(Image.open(f)) for f in files2], axis=0).astype(np.uint8)

    if save:
        Image.fromarray(avg1).save('average1.jpg')
        Image.fromarray(avg2).save('average2.jpg')
        print('Average images saved as average1.jpg and average2.jpg')
    return avg1, avg2


def calculate_pixelwise_variance(folder_path):
    image_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png'):
            image_path = os.path.join(folder_path, file_name)
            image = np.array(Image.open(image_path))
            image_list.append(image)

    stacked_images = np.stack(image_list, axis=2)
    variance_per_pixel = np.var(stacked_images, axis=2)
    return variance_per_pixel


def compare_spectra(train_dir, valid_dir, exp_num, hor_size=512, el_pointing=62, pixel_to_mm=0.137):
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.png')]
    valid_files = [os.path.join(valid_dir, f) for f in os.listdir(valid_dir) if f.endswith('.png')]

    train_avg = np.mean([np.array(Image.open(f)) for f in train_files], axis=0).astype(np.uint8)/255
    valid_avg = np.mean([np.array(Image.open(f)) for f in valid_files], axis=0).astype(np.uint8)/255

    settings = pd.read_csv("data/params.csv", engine='python')[["E","P","ms"]]
    ms = settings.loc[exp_num - 1]['ms']

    train_avg = torch.Tensor(train_avg).unsqueeze(0).unsqueeze(0) # once for batch, once for channels
    valid_avg = torch.Tensor(valid_avg).unsqueeze(0).unsqueeze(0)

    deflection_MeV, _ = deflection_biexp_calc(1, hor_size, el_pointing, pixel_to_mm)
    _, spectr_train = calc_spec(train_avg, 
                                el_pointing, 
                                deflection_MeV, 
                                acquisition_time_ms=torch.Tensor([ms]), 
                                resize=None,
                                image_gain=0,
                                device='cpu',
                                deflection_MeV_dx=None)
    _, spectr_valid = calc_spec(valid_avg, 
                                el_pointing, 
                                deflection_MeV, 
                                acquisition_time_ms=torch.Tensor([ms]), 
                                resize=None,
                                image_gain=0,
                                device='cpu',
                                deflection_MeV_dx=None)

    mse = mse_loss(spectr_train, spectr_valid)

    variance_train = np.mean(calculate_pixelwise_variance(train_dir))
    variance_valid = np.mean(calculate_pixelwise_variance(valid_dir))

    var_diff = np.abs(variance_train - variance_valid)

    # return {'mse': mse}, None
    return {'mse': mse, 'var_diff' : var_diff}, {'spectr_train' : spectr_train, 'spectr_valid' : spectr_valid}


def sample_all(root="models", result_dir="results/transfer_withgain_512_valid", device='cuda:2', n=8, dataset=Path("data/with_gain"), model_prefix='no_', load_model=True, section_counts=[30], cfg_scale=3, ns=700, model=None):
    if load_model:
        dir_list = [x for x in os.listdir(root) if x.startswith(model_prefix)]
    else:
        dir_list = os.listdir(dataset)
    settings = pd.read_csv("data/params.csv", engine='python')[["E","P","ms"]]
    for subdir in tqdm(dir_list):
        if load_model:
            exp_number = int(subdir.split('_')[1])
        else:
            exp_number = int(subdir)
        original_size = len(os.listdir(Path.joinpath(dataset, str(exp_number))))
        E = settings.loc[exp_number - 1]['E']
        P = settings.loc[exp_number - 1]['P']
        ms = settings.loc[exp_number - 1]['ms']
        if load_model:
            model = UNet_conditional(img_width=128, img_height=64, feat_num=3, device=device).to(device)
            ckpt = torch.load(os.path.join(root, subdir, 'ema_ckpt.pt'), map_location=device)
            model.load_state_dict(ckpt)
            model.eval()
        diffusion = SpacedDiffusion(beta_start=1e-4, beta_end=0.02, section_counts=section_counts, noise_steps=ns, img_width=128, img_height=64, device=device)
        y = torch.Tensor([E,P,ms]).to(device).float().unsqueeze(0) # parameter vector
        res_path = os.path.join(result_dir, str(exp_number))
        os.makedirs(res_path, exist_ok=True)
        total = len([f for f in os.listdir(res_path) if f.endswith('.png')])
        while total < original_size:
            if total + n > original_size:
                add = original_size - total
            else:
                add = n
            x = diffusion.ddim_sample_loop(model, y, cfg_scale=cfg_scale, resize=[256, 512], n=add, eta=1, device=device, gain=0)
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            res_path = os.path.join(result_dir, str(exp_number))
            save_samples(x, res_path, start_index=total)
            total += add


def main(validate_on = []):
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 601
    args.noise_steps = 1000
    args.phys = True
    args.beta_start = 1e-4
    args.beta_end = 0.02
    args.batch_size = 4
    args.image_height = 64
    args.image_width = 128
    args.real_size = (256, 512)
    args.features = ["E","P","ms"]
    args.dataset_path = r"data/with_gain"
    args.csv_path = "data/params.csv"
    args.device = "cuda:1"
    args.lr = 1e-3
    args.exclude = []# ['train/19']
    args.grad_acc = 1
    args.sample_freq = 0
    args.sample_settings = [32.,15.,15.]
    args.sample_size = 8
    args.electron_pointing_pixel = 62

    settings = pd.read_csv(args.csv_path, engine='python')[args.features]

    if validate_on:
        experiments = validate_on
    else:
        experiments = os.listdir(args.dataset_path)

    for experiment in sorted(experiments, key=lambda x: int(x)):
        args.exclude = [os.path.join(args.dataset_path, experiment)]
        args.run_name = "valid_gaindata_cosinesched_batch4_nonedx_600e_" + experiment
        row = settings.loc[[int(experiment) - 1], args.features]
        args.sample_settings = row.values.tolist()[0]

        model = UNet_conditional(img_width=128, img_height=64, feat_num=3, device=args.device).to(args.device)
        ckpt = torch.load("models/transfered.pt", map_location=args.device)
        model.load_state_dict(ckpt)
        train(args, model)


def sample_loop():
    device = 'cuda:2'
    names = ['valid_gaindata_cosineched']
    section_counts_options = [[10], create_sections_list(10, 10, cosine_step_schedule), [15], create_sections_list(10, 15, cosine_step_schedule),
                              [20], create_sections_list(10, 20, cosine_step_schedule), [25], create_sections_list(10, 25, cosine_step_schedule),
                              [30], create_sections_list(10, 30, cosine_step_schedule)]

    section_names = ['10', 'cos10', '15', 'cos15', '20', 'cos20', '25', 'cos25', '30', 'cos30']

    for name in names:
        for cfg in range(1, 9):
            for section_counts, section_str in zip(section_counts_options, section_names):
                result_dir = f'results_gaindata_batch4_600e/cossched_sec{section_str}_cfg{cfg}'
                if os.path.exists(result_dir):
                    print(f"Directory {result_dir} already exists. Skipping...")
                    continue 
                print(f"Processing with cfg={cfg} and section_counts={section_counts}")
                sample_all(load_model=True, root="models/" + name,
                        result_dir=result_dir,
                        device=device, ns=1000, section_counts=section_counts, n=24, cfg_scale=cfg)


def metrics_loop(dir1, dir2, csv_path, start=1, end=22):
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Physics', 'Sections', 'CFG', 'mse', 'var_diff', 'FID'])
    for dir in tqdm(os.listdir(dir2)):
        running_mse = 0
        running_var_diff = 0
        running_fid = 0
        count = 0
        physics = dir.split('_')[-3]
        sections = dir.split('_')[-2]
        cfg = dir.split('_cfg')[-1]
        for i in range(start, end + 1):
            subdir1 = os.path.join(dir1, str(i))
            subdir2 = os.path.join(dir2, dir, str(i))
            if os.path.isdir(subdir1) and os.path.isdir(subdir2):
                output = subprocess.check_output(['python', '-m', 'pytorch_fid', '--device', 'cuda:0', subdir1, subdir2])
                output_str = output.decode('utf-8').strip()
                fid = float(output_str.split()[-1])
                results, _ = compare_spectra(subdir1, subdir2, i)
                running_mse += results['mse'].item()
                running_var_diff += results['var_diff'].item()
                running_fid += fid
                count += 1

        with open(csv_path, 'a', newline='') as file:
            mse = running_mse / count
            var_diff = running_var_diff / count
            fid = running_fid / count
            writer = csv.writer(file)
            writer.writerow([physics, sections, cfg, mse, var_diff, fid])


if __name__ == "__main__":
    # main(validate_on=['3', '8', '11', '19', '21'])
    metrics_loop('data/with_gain', 'results_gaindata_batch4_600e', 'metrics.csv')
    # sample_loop()

# validate on: 3, 8, 11, 19, 21
