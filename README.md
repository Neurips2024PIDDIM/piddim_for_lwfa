# Physics Informed Denoising Diffusion Implicit Models for Laser-Plasma Accelerator Optimization

This repository contains the implementation and testing of a physics-informed conditional denoising diffusion implicit model capable of simulating the laser wakefield accelerator experiments.

The dataset and models can be found in the [following link](https://zenodo.org/records/11245954?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImVkZWFkZGFhLWM3YWUtNDEyMS1iNjMwLWM5ZDgzNGNlOWE4ZCIsImRhdGEiOnt9LCJyYW5kb20iOiI4ZTI0NDQ5ODY1ZTYwYjdkYmI3YTk0NjI3YWFiYjNlZSJ9.y2ZF4oDpl1mkAOzdJnqcUUCxVvh1DdX0lJdIBgMe2o6hFiLa7Koe6mPvymaWI4OHkV4Gybq5ISvkaeUVrzX2Hw).

## Installation:

```
conda create --name "name" python=3.10.12
conda activate "name"
pip install -r requirements.txt
```

## Training:

For training see the `train.py` script. See the code for any parameters. All default values are selected considering the setting used to gather the results listed in the paper. Set the `args.phys` parameter to `True` to run PIDDIM training, `False` for DDIM.
```
python3 train.py
```

## Validation:

For validation see the `metrics.py` file. It contains pipelines for cross-validation training, sampling and finally evaluating the generated models using FID.
```
python3 metrics.py
```

## Sampling:

To see the sampling process, please refer to the `sample.ipynb` file.