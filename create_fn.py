# python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms 

import argparse
# ======== create 
import copy
import glob
import multiprocessing
import os
import sys
import time
import warnings
from asyncio.log import logger
from contextlib import contextmanager, nullcontext
from itertools import islice
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, overload

import cv2
import numpy as np
import torch
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from einops import rearrange
from imwatermark import WatermarkEncoder
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from transformers import AutoFeatureExtractor

from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler
from stable_diffusion.ldm.models.diffusion.plms import PLMSSampler
from stable_diffusion.ldm.util import instantiate_from_config
from utils import (check_safety, chunk, free_memory, load_model_from_config,
                   numpy_to_pil, put_watermark)

if TYPE_CHECKING:
    import threading
    import asyncio

from docarray import Document, DocumentArray

_models_cache = {}

_flag = False
def create(
    prompt: str,
    sess_name: Optional[str] = None,
    output_dir: Optional[str] = "outputs/",
    skip_grid: Optional[bool] = True,
    skip_save: Optional[bool] = False,
    ddim_steps: Optional[int] = 50,
    plms: Optional[bool] = True,
    fixed_code: Optional[str] = False,
    ddim_eta: Optional[float] = 0.0,
    n_iter: Optional[int] = 1,
    height: Optional[int] = 512,
    width: Optional[int] = 768,
    l_channels: Optional[int] = 4,
    f_downsample: Optional[int] = 8,
    n_samples: Optional[int] = 1,
    n_rows: Optional[int] = 0,
    scale: Optional[float] = 7.5,
    config_file: Optional[str] = "stable_diffusion/configs/stable-diffusion/v1-inference.yaml",
    ckpt: Optional[str] = "stable_diffusion/models/ldm/stable-diffusion-v1/model.ckpt",
    seed: Optional[int] = 42,
    precision: Optional[str] = "autocast",
    skip_event=None,
    stop_event=None,
)  -> Optional['DocumentArray']:
    da = DocumentArray()
    free_memory()
    global _flag
    if not _flag:
        config = OmegaConf.load(f"{config_file}")
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        wm = "StableDiffusionV1"
        _flag = True

    try:
        seed_everything(seed)
        if _models_cache.get('ckpt', None) is None:
            _models_cache['ckpt'] = load_model_from_config(config, f"{ckpt}").to(device)
        model = _models_cache['ckpt']
        
        if _models_cache.get('sampler', None) is None:
            _models_cache['sampler'] = DDIMSampler(model) if not plms else PLMSSampler(model)
        sampler = _models_cache['sampler']

        if _models_cache.get('wm_encoder', None) is None:
            _models_cache['wm_encoder'] = WatermarkEncoder()
            _models_cache['wm_encoder'].set_watermark('bytes', wm.encode('utf-8'))
        wm_encoder = _models_cache['wm_encoder']


        os.makedirs(output_dir, exist_ok=True)
        outpath = output_dir

        print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")

        batch_size = n_samples
        n_rows = n_rows if n_rows > 0 else batch_size

        # prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1

        start_code = None
        if fixed_code:
            start_code = torch.randn([n_samples, l_channels, height // f_downsample, width // f_downsample], device=device)

        precision_scope = autocast if precision=="autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    for n in trange(n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            shape = [l_channels, height // f_downsample, width // f_downsample]
                            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                            conditioning=c,
                                                            batch_size=n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=uc,
                                                            eta=ddim_eta,
                                                            x_T=start_code)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                            x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                            if not skip_save:
                                for x_sample in x_checked_image_torch:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    da.append(Document(tensor=x_sample.astype(np.uint8)))
                                    img = Image.fromarray(x_sample.astype(np.uint8))
                                    img = put_watermark(img, wm_encoder)
                                    img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                    base_count += 1


                            if not skip_grid:
                                all_samples.append(x_checked_image_torch)

                    if not skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        img = Image.fromarray(grid.astype(np.uint8))
                        img = put_watermark(img, wm_encoder)
                        img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                        grid_count += 1

                    toc = time.time()

        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
            f" \nEnjoy.")
        da.save_binary(os.path.join(sample_path, f"{sess_name}.protobuf.lz4"))
    except Exception as e:
        logger.exception(e)
    finally:
        free_memory()

