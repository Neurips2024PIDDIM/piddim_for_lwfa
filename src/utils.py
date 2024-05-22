# source: https://github.com/dome272/Diffusion-Models-pytorch
import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib as mpl
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import cv2
import glob
import src.dataset as dataset
import scipy
import torchvision.transforms.functional as f

class ExperimentDataset(Dataset):
    def __init__(self, csv_file="params.csv", root_dir="train", transform=None, features=["E","perc_N","P","gain","ms"], exclude=[]):
        """
        Arguments:
            csv_file (string): Path to the csv file with settings.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.features = features
        self.settings = pd.read_csv(csv_file, engine='python')[features]# .apply(self.min_max_norm)
        self.root_dir = root_dir
        self.exclude = exclude
        self.file_list = self.get_list_of_img()
        self.transform = transform

    def min_max_norm(self, col):
        return (col - col.min()) / (col.max() - col.min())

    def get_list_of_img(self, regex="*.png"):
        files = []
        for dirpath, _, _ in os.walk(self.root_dir):
            if dirpath in self.exclude:
                print("Excluding " + dirpath)
                continue
            type_files = glob.glob(os.path.join(dirpath, regex))
            files += type_files
        return sorted(files)
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.file_list[idx]
        image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        exp_index = int(filename.split('/')[-2])
        settings = self.settings.iloc[exp_index-1, 0:]
        settings = np.array([settings])
        settings = settings.astype('float32').reshape(-1, len(self.features))
        if self.transform:
            image = self.transform(image)
        to_tens = torchvision.transforms.ToTensor()
        settings = to_tens(settings).squeeze()
        sample = {'image': image, 'settings': settings}
        return sample


def load_images_from_dir(path, num_images):
    images = []
    for i in range(num_images):
        image_path = os.path.join(path, f"{i}.png")
        image = Image.open(image_path)
        images.append(image)
    return images


def plot_images_from_dir(path, num_images):
    images = load_images_from_dir(path, num_images)
    n = len(images)
    rows = (n + 4) // 5
    cols = min(n, 5)

    plt.figure(figsize=(32, 32))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], vmin=0, vmax=255, cmap=mpl.colormaps['viridis'])
        plt.axis("off")
        plt.title(f"{i}", size=12)

    plt.show()


def plot_images(images):
    n = len(images)
    rows = 2
    cols = 4

    plt.figure(figsize=(64, 32))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].cpu().permute(1, 2, 0).numpy(), vmin=0, vmax=255, cmap=mpl.colormaps['viridis'])
        plt.axis("off")
        plt.title(f"{i}", size=12)

    plt.show()
    
    
def plot_average_image_pairs(root_folder, acquisition_time_ms, electron_pointing_pixel=62):
    subfolders = sorted([f.path for f in os.scandir(root_folder) if f.is_dir()])
    n = len(subfolders)
    fig, axs = plt.subplots(n, 2, figsize=(15, 4*n))
    fig.subplots_adjust(hspace=0.35)  # Increase the space between rows
    fig.subplots_adjust(wspace=0.1)  # Decrease the space between columns
    for i, subfolder in enumerate(subfolders):
        images = []
        for filename in os.listdir(subfolder):
            if filename.endswith(".png"):
                im = cv2.imread(os.path.join(subfolder, filename), cv2.IMREAD_UNCHANGED)
                images.append(im)
        avg_im = np.mean(images, axis=0)
        deflection_MeV, spectrum_calibrated = dataset.get_1d(avg_im/255, acquisition_time_ms, electron_pointing_pixel=electron_pointing_pixel)

        axs[i, 1].plot(deflection_MeV, spectrum_calibrated)  # plot without fit
        axs[i, 1].set_title('Reconstructed Spectrum')
        axs[i, 1].set_ylabel('Spectral Intensity (pA/MeV)')
        axs[i, 1].set_xlabel('Energy (MeV)')
        axs[i, 1].set_xlim([2, 20])
        axs[i, 0].imshow(avg_im, vmin=0, vmax=255)
        axs[i, 0].set_title(os.path.basename(subfolder))
    plt.show()


def find_ticks(deflection_MeV, beam_point_x, beam_point_y, pixel_in_mrad, energy_levels, ranges):
    # Find the index of the first occurrence within each energy level range
    ticks = {}
    for energy, (low, high) in zip(energy_levels, ranges):
        ticks[f'tick{energy}MeV'] = next((i for i, val in enumerate(deflection_MeV[beam_point_x:], start=beam_point_x) if low < val < high), None)
    # Calculate y-ticks
    ticks['tick_10mrad_px'] = beam_point_y - round(10 / pixel_in_mrad)
    ticks['tick0mrad_px'] = beam_point_y
    ticks['tick10mrad_px'] = beam_point_y + round(10 / pixel_in_mrad)
    return ticks


def plot_image_pairs(images, acquisition_time_ms, beam_point_x, beam_point_y, energy, pressure, xlim=[2, 20], model=1, gain=0, noise=False):
    def get_y_lims_within_xlim(x, y, xlim):
        """Find the min and max y-values within the specified x-limits using PyTorch."""
        within_xlim = (x >= xlim[0]) & (x <= xlim[1])
        y_within_xlim = y[within_xlim]
        return [torch.min(y_within_xlim), torch.max(y_within_xlim)] if y_within_xlim.numel() > 0 else [torch.min(y), torch.max(y)]

    n = len(images)
    pixel_in_mrad = 0.3653
    energy_levels = [100, 30, 15, 10, 8, 5, 3]  # Removed 40 and 20
    ranges = [(70, 101), (20, 31), (12, 15.5), (8, 10.5), (6, 8.2), (4.8, 5.2), (2.9, 3.2)]  # Adjusted ranges

    fig, axs = plt.subplots(n, 2, figsize=(15, 4*n))
    fig.subplots_adjust(hspace=0.35, wspace=0.15, top=0.98)
    if n == 1:
        axs = axs.reshape(1, -1)
    deflection_MeV, deflection_MeV_dx = deflection_biexp_calc(n, images.shape[-1], beam_point_x)
    deflection_MeV = deflection_MeV[0].unsqueeze(0) # make it batched but of batchsize 1
    # deflection_MeV_dx = deflection_MeV_dx[0].unsqueeze(0) # make it batched but of batchsize 1
    for i in range(n):
        im = images[i].unsqueeze(0).unsqueeze(0)
        deflection_MeV, spectrum_calibrated = calc_spec(im/255, beam_point_x, deflection_MeV, torch.tensor(acquisition_time_ms), image_gain=gain, noise=noise, deflection_MeV_dx=None)  # Using a local function
        ticks = find_ticks(deflection_MeV.squeeze().cpu(), beam_point_x, beam_point_y, pixel_in_mrad, energy_levels, ranges)
        # Plot the spectrum
        axs[i, 1].plot(deflection_MeV.squeeze().cpu(), spectrum_calibrated.squeeze().cpu())
        axs[i, 1].set_title('Reconstructed Spectrum', fontsize=12)
        axs[i, 1].set_ylabel('Spectral Intensity (pA/MeV)', fontsize=12)
        axs[i, 1].set_xlabel('Energy [MeV]', fontsize=12)
        axs[i, 1].set_xlim(xlim)
        y_lims = get_y_lims_within_xlim(deflection_MeV, spectrum_calibrated, xlim)
        axs[i, 1].set_ylim(y_lims)

        # Plot the image
        axs[i, 0].imshow(im.squeeze().cpu(), vmin=0, vmax=255, cmap='inferno')
        axs[i, 0].set_title(f"Image {i}")

        # Set y-axis ticks for mrad values
        axs[i, 0].set_yticks([ticks['tick_10mrad_px'], ticks['tick0mrad_px'], ticks['tick10mrad_px']])
        axs[i, 0].set_yticklabels(['-10', '0', '10'])
        axs[i, 0].set_ylabel('Angle [mrad]')

        # Set x-axis ticks for MeV values
        mev_ticks = [tick for key, tick in ticks.items() if 'MeV' in key and tick is not None]
        axs[i, 0].set_xticks(mev_ticks)
        axs[i, 0].set_xticklabels([key.split('tick')[1].replace('MeV', '') for key in ticks if 'MeV' in key and ticks[key] is not None])
        axs[i, 0].set_xlabel('Energy [MeV]')
        # deflection_MeV = deflection_MeV.unsqueeze(0)
    plt.show()


def compare_images(images1, images2, settings, beam_point_x, beam_point_y, folder='random_sample'):
    n = len(images1)
    pixel_in_mrad = 0.3653
    energy_levels = [100, 30, 15, 10, 8, 5, 3]  # Removed 40 and 20
    ranges = [(70, 101), (20, 31), (12, 15.5), (8, 10.5), (6, 8.2), (4.8, 5.2), (2.9, 3.2)]  # Adjusted ranges

    fig, axs = plt.subplots(n, 2, figsize=(15, 4*n))
    fig.subplots_adjust(hspace=0.3, wspace=0.15, top=0.96, bottom=0.01)
    plt.suptitle(f"Energy: {settings['E']} MeV, Pressure: {settings['P']} bar, Acquisition time: {settings['acq_time']} ms", fontsize=16)
    
    if n == 1:
        axs = axs.reshape(1, -1)
    
    deflection_MeV, deflection_MeV_dx = deflection_biexp_calc(n, images1.shape[-1], beam_point_x)
    deflection_MeV = deflection_MeV[0].unsqueeze(0) # make it batched but of batchsize 1
    deflection_MeV_dx = deflection_MeV_dx[0].unsqueeze(0) # make it batched but of batchsize 1
    
    for i in range(n):
        for j in [0, 1]:
            image = images1[i] if j == 0 else images2[i]
            image = image.unsqueeze(0).unsqueeze(0)
            ticks = find_ticks(deflection_MeV.squeeze().cpu(), beam_point_x, beam_point_y, pixel_in_mrad, energy_levels, ranges)

            axs[i, j].imshow(image.squeeze().cpu(), vmin=0, vmax=255, cmap='inferno')
            axs[i, j].set_yticks([ticks['tick_10mrad_px'], ticks['tick0mrad_px'], ticks['tick10mrad_px']])
            axs[i, j].set_yticklabels(['-10', '0', '10'])
            axs[i, j].set_ylabel('Angle [mrad]')

            mev_ticks = [tick for key, tick in ticks.items() if 'MeV' in key and tick is not None]
            axs[i, j].set_xticks(mev_ticks)
            axs[i, j].set_xticklabels([key.split('tick')[1].replace('MeV', '') for key in ticks if 'MeV' in key and ticks[key] is not None])
            axs[i, j].set_xlabel('Energy [MeV]')

    # Construct a filename from settings
    filename = f"Energy_{settings['E']}_Pressure_{settings['P']}bar_AcqTime_{settings['acq_time']}ms"
    filename = filename.replace(" ", "_").replace(".", "p")  # Replace spaces and dots for compatibility
    os.makedirs(folder, exist_ok=True)
    plt.savefig(folder + '/' + filename)


def stitch_images(directory):
    # Get the list of image files in the directory
    image_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg')])
    image_files.sort(key=lambda f: int(os.path.basename(f).split('_')[0]))

    # Open the images and get their sizes
    images = [Image.open(f) for f in image_files]
    widths, heights = zip(*(i.size for i in images))

    # Create a new image with the combined height of all images
    total_height = sum(heights)
    max_width = max(widths)
    stitched_image = Image.new('RGB', (max_width, total_height))

    # Paste the images into the stitched image
    y_offset = 0
    for img in images:
        stitched_image.paste(img, (0, y_offset))
        y_offset += img.size[1]

    # Save the stitched image
    stitched_image.save(os.path.join(directory, 'stitched_image.png'))


def save_samples(images, folder="samples", start_index=0):
    # print(images.shape)
    ndarr = images.to('cpu').numpy()
    indexes = range(start_index, start_index + len(ndarr))
    for i, im in zip(indexes, ndarr):
        cv2.imwrite(folder + "/" + str(i) + ".png", im)


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((args.image_height, args.image_width), antialias=True),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.Normalize(0.5, 0.5)
    ])
    dataset = ExperimentDataset(args.csv_path, args.dataset_path, transform=transforms, features=args.features, exclude=args.exclude)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


def deflection_calc(batch_size, hor_image_size, electron_pointing_pixel):
    pixel_in_mm = 0.137
    deflection_MeV = torch.zeros((batch_size, hor_image_size))
    deflection_mm = torch.zeros((batch_size, hor_image_size))
    mat = scipy.io.loadmat('data/Deflection_curve_Mixture_Feb28.mat')
    for i in range(hor_image_size):
        # if i <= electron_pointing_pixel:
        #     deflection_mm[:, i] = 0
        # else:
        deflection_mm[:, i] = (i - electron_pointing_pixel) * pixel_in_mm
            
    for i in range(electron_pointing_pixel, hor_image_size):
        xq = deflection_mm[:, i]
        mask = xq > 0
        if mask.any():
            deflection_MeV[mask, i] = torch.from_numpy(scipy.interpolate.interp1d(mat['deflection_curve_mm'][:, 0],
                                                           mat['deflection_curve_MeV'][:, 0],
                                                           kind='linear',
                                                           assume_sorted=False,
                                                           bounds_error=False)(xq[mask]).astype(np.float32))
    return deflection_MeV#[:][electron_pointing_pixel:]


def bi_exponential_deflection(x, a1=77.855568601465, b1=0.466485822903793, a2=19.911755340829, b2=0.043573073167125255):
    return a1 * torch.exp(-b1 * x) + a2 * torch.exp(-b2 * x)

def bi_exponential_deflection_dx(x, a1=-36.318518986697, b1=0.466485822903793, a2=-0.86761637235184, b2=0.043573073167125255):
    return a1 * torch.exp(-b1 * x) + a2 * torch.exp(-b2 * x)


def deflection_biexp_calc(batch_size, hor_image_size, electron_pointing_pixel, pixel_in_mm=0.137):
    linear_space = torch.arange(hor_image_size) * pixel_in_mm
    linear_space = linear_space.repeat(batch_size, 1)
    deflection_MeV = bi_exponential_deflection(linear_space).to(torch.float32)
    zeros = torch.zeros(batch_size, electron_pointing_pixel + 11) # el_pointing, 11 are nans

    # Concatenate zeros to the beginning of each tensor in the batch
    deflection_MeV = torch.cat((zeros, deflection_MeV), dim=1)

    # Slice the concatenated tensor to keep only the first 512 elements in each tensor
    deflection_MeV = deflection_MeV[:, :hor_image_size]
    deflection_MeV_dx = bi_exponential_deflection_dx(linear_space).to(torch.float32)
    deflection_MeV_dx = torch.cat((zeros, deflection_MeV_dx), dim=1)
    deflection_MeV_dx = deflection_MeV_dx[:, :hor_image_size]
    return deflection_MeV, deflection_MeV_dx


def calc_spec(image, electron_pointing_pixel, deflection_MeV, acquisition_time_ms, image_gain=0, resize=None, noise=False, device='cpu', deflection_MeV_dx=None):
    if resize:
        image = f.resize(image, resize, antialias=True)
    image_gain /= 32  # correction for CCD settings
    if noise:
        noise = torch.median(torch.stack([
            image[:, :, int(image.shape[1] * 0.9), int(image.shape[2] * 0.05)],
            image[:, :, int(image.shape[1] * 0.9), int(image.shape[2] * 0.9)],
            image[:, :, int(image.shape[1] * 0.1), int(image.shape[2] * 0.9)]
        ], dim=0), dim=(1, 2))
        noise = noise.unsqueeze(1).unsqueeze(2)
        image[image <= noise] = 0

    hor_image_size = image.shape[-1]
    batch_size = image.shape[0]
    horizontal_profile = torch.sum(image, dim=(1, 2)).to(device)

    spectrum_in_pixel = torch.zeros((batch_size, hor_image_size)).to(device)
    spectrum_in_MeV = torch.zeros((batch_size, hor_image_size)).to(device)

    # Fill spectrum_in_pixel for all pixels at once
    spectrum_in_pixel[:, electron_pointing_pixel:] = horizontal_profile[:, electron_pointing_pixel:]
    pad = torch.zeros_like(deflection_MeV[:, :1])
    # Compute the derivative array
    if deflection_MeV_dx is None:
        shifts = -1
        deflection_MeV_shifted = torch.roll(deflection_MeV, shifts=shifts, dims=1)

        # Pad with zeros where necessary
        if shifts < 0:
            # For left shift, zero pad on the right
            pad = torch.zeros_like(deflection_MeV[:, -shifts:])  # Creating a padding tensor
            deflection_MeV_shifted[:, -shifts:] = pad
        else:
            # For right shift, zero pad on the left
            pad = torch.zeros_like(deflection_MeV[:, :shifts])  # Creating a padding tensor
            deflection_MeV_shifted[:, :shifts] = pad

        # Calculate derivative
        derivative = deflection_MeV - deflection_MeV_shifted
    else:
        derivative = -deflection_MeV_dx[:, electron_pointing_pixel:]

    derivative = derivative.to(device)
    mask = derivative != 0
    if deflection_MeV_dx is not None:
        derivative_expanded = derivative.expand_as(spectrum_in_pixel[:, electron_pointing_pixel:])
        spectrum_in_MeV[:, electron_pointing_pixel:][mask] = spectrum_in_pixel[:, electron_pointing_pixel:][mask] / derivative_expanded[mask]
    else:
        derivative_expanded = derivative# .expand_as(spectrum_in_pixel)
        spectrum_in_MeV[:, :][mask] = spectrum_in_pixel[:, :][mask] / derivative_expanded[mask]
    # Calculate the spectrum in MeV, avoiding division by zero
    # spectrum_in_MeV[:, :][mask] = spectrum_in_pixel[:, :][mask] / derivative_expanded[mask]

    spectrum_in_MeV[~torch.isfinite(spectrum_in_MeV)] = 0

    acquisition_time_ms = acquisition_time_ms.reshape(batch_size, 1).repeat(1, hor_image_size).to(device)
    spectrum_calibrated = (spectrum_in_MeV * 3.706) / (acquisition_time_ms * image_gain) if image_gain else (spectrum_in_MeV * 3.706) / acquisition_time_ms

    return deflection_MeV, spectrum_calibrated