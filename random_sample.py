import torch
import random
from src.utils import compare_images
from src.diffusion import SpacedDiffusion
from src.modules import UNet_conditional
from metrics import create_sections_list, cosine_step_schedule

# Setup constants and models
n = 8
device = 'cuda:1'
sampler = SpacedDiffusion(beta_start=1e-4, beta_end=0.02, noise_steps=1000, section_counts=create_sections_list(10, 25, cosine_step_schedule), img_height=64, img_width=128, device=device, rescale_timesteps=False)

model1 = UNet_conditional(img_width=128, img_height=64, feat_num=3, device=device).to(device)
model2 = UNet_conditional(img_width=128, img_height=64, feat_num=3, device=device).to(device)

model1_path = "models/nophys/ema_ckpt.pt"
model2_path = "models/cossched/ema_ckpt.pt"

model1.load_state_dict(torch.load(model1_path, map_location=device))
model2.load_state_dict(torch.load(model2_path, map_location=device))

# Loop where settings are chosen at random
num_iterations = 25  # Set the number of iterations for the loop
for _ in range(num_iterations):
    settings = {'E': random.randint(1, 50), 'P': random.randint(1, 50), 'acq_time': 20}
    y = torch.Tensor([settings['E'], settings['P'], settings['acq_time']]).to(device).float().unsqueeze(0)  # Parameter vector
    
    images1 = sampler.ddim_sample_loop(model=model1, y=y, cfg_scale=1, device=device, eta=1, n=n)
    images2 = sampler.ddim_sample_loop(model=model2, y=y, cfg_scale=1, device=device, eta=1, n=n)
    
    compare_images(images1, images2, settings, beam_point_x=62, beam_point_y=128, folder='random_sample_bestnophys')
    print(f"Completed iteration with settings: Energy={settings['E']}, Pressure={settings['P']}, Acquisition time={settings['acq_time']} ms")
