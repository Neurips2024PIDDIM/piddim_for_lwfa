# Camera used: Basler aca2040-25gm

import math
import os
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
import cv2
import pandas as pd
import scipy
import torch


def add_fingerprint(image, size=5):
    image[0:size, 0:size] = 255
    return image


# Matlab-like round
def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier


def round_half_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n*multiplier - 0.5) / multiplier


def round(n, decimals=0):
    if n >= 0:
        rounded = round_half_up(n, decimals)
    else:
        rounded = round_half_down(n, decimals)
    return int(rounded)


def read_img(path):
    I = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return I


def get_list_of_imgs(folder_path, type_regex="*.tiff"):
    files = []
    for dirpath, _, _ in os.walk(folder_path):
        tiff_files = glob.glob(os.path.join(dirpath, type_regex))
        files += tiff_files
    return sorted(files)


def preprocess_image(img):
    I_filtered = cv2.medianBlur(img, 5)
    I_norm = I_filtered / 16 / 4095 # http://softwareservices.flir.com/BFS-PGE-31S4/latest/Model/public/ImageFormatControl.html
    # Remove scratches
    I_norm[780:840, 500:550] = 0
    I_norm[1255:1262, 1101:1111] = 0
    return I_norm


# srcs: https://docs.baslerweb.com/gain, https://www.quora.com/What-is-the-formula-for-converting-decibels-to-linear-units


def set_gain(img, gain_raw, desired_gain_raw):
    if gain_raw:
        current_gain_dB = 20 * np.log10(gain_raw / 32)
        current_gain_lin = np.power(10, current_gain_dB/10)
        img = img / current_gain_lin

    desired_gain_dB = 20 * np.log10(desired_gain_raw / 32)
    desired_gain_lin = np.power(10, desired_gain_dB/10)
    
    img_with_gain = img * desired_gain_lin

    return img_with_gain.astype(np.uint16)


def find_dots(images):
    circles = []
    for img in images:
        img = img * 255
        img = img.astype(np.uint8)
        binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 4)
        circles_in_img = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, minDist=20, param1=50, param2=7, minRadius=0, maxRadius=10)
        if circles_in_img is not None:
            circles_in_img = np.round(circles_in_img[0, :]).astype(int)
            circles.extend(circles_in_img)
    return circles


def remove_dots_old(img, black_dots):
    I_wo_black_dots = img.copy()
    height, width = I_wo_black_dots.shape
    for dot in black_dots: # interpolating values around the black dot. Only considers the x axis (y is commented out in original_script)
        center_hor = dot[0]
        center_ver = dot[1]
        radius = dot[2]
        for l in range(width):
            for m in range(height):
                if center_hor - radius < l < center_hor + radius and center_ver - radius <  m < center_ver + radius \
                    and math.sqrt((m - center_ver)**2 + (l - center_hor)**2) < radius:
                    aux_x_min = center_hor - round(math.sqrt(radius**2 - (m - center_ver)**2))
                    aux_x_max = center_hor + round(math.sqrt(radius**2 - (m - center_ver)**2))
                    I_wo_black_dots[m,l] = ((aux_x_max - l) / (aux_x_max - aux_x_min)) * img[m,aux_x_min] + ((l - aux_x_min) / (aux_x_max - aux_x_min)) * img[m,aux_x_max]
    return I_wo_black_dots


def remove_dots(img, black_dots):
    I_wo_black_dots = img.copy()
    height, width = I_wo_black_dots.shape
    for dot in black_dots:
        center_hor = dot[0]
        center_ver = dot[1]
        radius = dot[2] + 4
        x_min = max(center_hor - radius, 0)
        x_max = min(center_hor + radius + 1, width)
        y_min = max(center_ver - radius, 0)
        y_max = min(center_ver + radius + 1, height)
        x_range = np.arange(x_min, x_max)
        y_range = np.arange(y_min, y_max)
        xx, yy = np.meshgrid(x_range, y_range)
        dist = np.sqrt((xx - center_hor) ** 2 + (yy - center_ver) ** 2)
        mask = dist < radius
        aux_x_min = np.round(center_hor - np.sqrt(radius ** 2 - (yy[mask] - center_ver) ** 2)).astype(int)
        aux_x_max = np.round(center_hor + np.sqrt(radius ** 2 - (yy[mask] - center_ver) ** 2)).astype(int)
        aux_x_min[aux_x_min < 0] = 0
        aux_x_max[aux_x_max >= width] = width - 1
        aux_y = yy[mask]
        I_wo_black_dots[aux_y, xx[mask]] = ((aux_x_max - xx[mask]) / (aux_x_max - aux_x_min)) * img[aux_y, aux_x_min] + ((xx[mask] - aux_x_min) / (aux_x_max - aux_x_min)) * img[aux_y, aux_x_max]
    return I_wo_black_dots



def find_laser(images):
    sum_imgs = np.zeros_like(images[0], dtype=np.float64)
    for img in images:
        sum_imgs += img
    max_index = np.argmax(sum_imgs)
    max_x = max_index % sum_imgs.shape[1]
    max_y = max_index // sum_imgs.shape[1]
    return max_x, max_y


def crop_by_beam(image, beam_pos, x_index=62, size=(256, 512)):
    return image[beam_pos[1] - size[0]//2:beam_pos[1] + size[0]//2, beam_pos[0] - x_index:beam_pos[0] + size[1] - x_index]


def get_1d(image, acquisition_time_ms, electron_pointing_pixel, image_gain=0): # image should be 0 - 1 values
    image_gain /= 32 # from original script
    # if image_gain:
    #     noise *= image_gain # this part is missing in the original script
    # image = cv2.medianBlur((image * 255).astype(np.uint8), ksize=5)/255
    noise = np.median([image[int(image.shape[0]*0.9),int(image.shape[1]*0.05)],
                       image[int(image.shape[0]*0.9),int(image.shape[1]*0.9)],
                       image[int(image.shape[0]*0.1),int(image.shape[1]*0.9)]])
    # noise = np.percentile(image, 70)
    # print(noise)
    image[image <= noise] = 0
    pixel_in_mm = 0.137
    hor_image_size = image.shape[1]
    horizontal_profile = np.sum(image, axis=0)
    spectrum_in_pixel = np.zeros(hor_image_size)
    spectrum_in_MeV = np.zeros(hor_image_size)
    deflection_MeV = np.zeros(hor_image_size)
    deflection_mm = np.zeros(hor_image_size)
    mat = scipy.io.loadmat('data/Deflection_curve_Mixture_Feb28.mat')
    for i in range(hor_image_size): # defining the mm in the image, added + 1
        if i <= electron_pointing_pixel:
            deflection_mm[i] = 0
        else:
            deflection_mm[i] = (i - electron_pointing_pixel) * pixel_in_mm
    #---Assigning to each pixel its value in MeV with the loaded deflection curve------
    for i in range(electron_pointing_pixel, len(deflection_MeV)):
        xq = deflection_mm[i]
        if xq > 1:
            deflection_MeV[i] = scipy.interpolate.interp1d(mat['deflection_curve_mm'][:, 0],
                                                           mat['deflection_curve_MeV'][:, 0],
                                                           kind='linear',
                                                           assume_sorted=False,
                                                           bounds_error=False)(xq)
    for j in range(electron_pointing_pixel, hor_image_size):
        spectrum_in_pixel[j] = horizontal_profile[j]
    spectrum_in_MeV[0] = spectrum_in_pixel[0]
    for j in range(electron_pointing_pixel, hor_image_size):
        diff = deflection_MeV[j-1] - deflection_MeV[j]
        spectrum_in_MeV[j] = 0 if diff == 0 else (spectrum_in_pixel[j]) / diff
        # spectrum_in_MeV[j] = (spectrum_in_pixel[j]) / diff

    spectrum_calibrated = (spectrum_in_MeV * 3.706) / (acquisition_time_ms*image_gain) if image_gain else (spectrum_in_MeV * 3.706) / acquisition_time_ms
    return deflection_MeV, spectrum_calibrated


def get_nomag_subfolder(experiment):
    experiment = str(experiment)
    if int(experiment) >= 6 and int(experiment) <= 11:
        return Path('6')
    elif int(experiment) >= 12 and int(experiment) <= 18:
        return Path('12')
    elif int(experiment) == 22:
        return Path('22')
    else:
        return experiment
    # Old handling:
    # if str(experiment) == '17' or str(experiment) == '18' or str(experiment) == '15':
    #     ex_name = Path('17_18')
    # elif str(experiment) == '14':
    #     ex_name = '12'
    # else:
    #     ex_name = experiment.name
    # calibration_folder = mag_out_folder / ex_name
    # return calibration_folder



def prepare_data(mag_out_folder=Path('data/mag_out'), experiment_folder=Path('data/raw'), output_folder=Path('data/test'), parameters="data/params.csv"):
    # Parameters will be a path to csv when I create it
    experiments = os.listdir(experiment_folder)
    params = pd.read_csv(parameters)["gain"]
    for experiment in tqdm(experiments):
        if experiment == '.DS_Store':
            continue
        gain_raw = params[int(experiment) - 1]
        experiment = Path(experiment)
        ex_name = get_nomag_subfolder(experiment)
        calibration_folder = mag_out_folder / ex_name

        # Image preparation
        images = [read_img(a) for a in get_list_of_imgs(experiment_folder/experiment)]
        # images = [remove_gain(x, gain_raw) for x in images]
        images = [preprocess_image(a) for a in images]
        image_dots = find_dots(images)
        images = [remove_dots(a, image_dots) for a in images]

        # Calibration image preparation
        calib = [read_img(a) for a in get_list_of_imgs(calibration_folder)]
        if ex_name == '12':
            for x in calib:
                x[:, 1300:] = 0 # No_mag for experiment 12 has a strange bright line at the right part of all images
        calib = [preprocess_image(a) for a in calib]
        calib_dots = find_dots(calib)
        calib = [remove_dots(a, calib_dots) for a in calib]
        
        # Crop by laser pos
        beam_pos = find_laser(calib)
        images = [crop_by_beam(a, beam_pos) for a in images]
        # Save results
        os.mkdir(output_folder/experiment)
        for i, im in enumerate(images):
            im = (im*255).astype(np.uint8)
            cv2.imwrite(str(output_folder/experiment/Path(str(i) + '.png')), im)


def remove_gain(img, gain_raw, tensor=False):
    if not gain_raw: return img
    gain_dB = 20 * np.log10(gain_raw / 32)
    gain_lin = np.power(10, gain_dB/20)
    img_nogain = img / gain_lin
    if not tensor:
        return img_nogain.astype(np.uint8)
    else:
        return img_nogain.type(torch.uint8)


def set_gain(img, gain_raw, desired_gain_raw):
    if gain_raw:
        current_gain_dB = 20 * np.log10(gain_raw / 32)
        current_gain_lin = np.power(10, current_gain_dB/20)
        img = img / current_gain_lin

    desired_gain_dB = 20 * np.log10(desired_gain_raw / 32)
    desired_gain_lin = np.power(10, desired_gain_dB/20)
    
    img_with_gain = img * desired_gain_lin

    return img_with_gain.astype(np.uint16)


def set_gain_all(source_dir, target_dir, parameters="data/params.csv"):
    experiments = os.listdir(source_dir)
    params = pd.read_csv(parameters)["gain"]
    for experiment in tqdm(experiments):
        images = [read_img(a) for a in get_list_of_imgs(source_dir/experiment, type_regex="*.png")]
        gain_raw = params[int(experiment) - 1]
        images = [set_gain(x, gain_raw, 50) for x in images]
        for image in images:
            copy = image
            copy[:, 2:] = image[:, :-2]
            image = copy
        os.mkdir(target_dir/experiment)
        for i, im in enumerate(images):
            im = np.clip(im, 0, 255).astype(np.uint8)
            cv2.imwrite(str(target_dir/experiment/Path(str(i) + '.png')), im)


def main():
    set_gain_all("data/with_gain", "data/50_gain")

if __name__ == "__main__":
    main()


# 1279 images
# 1221 no_mag images

# Load img -> 