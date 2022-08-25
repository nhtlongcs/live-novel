# python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms 

import argparse, os, sys, glob
from asyncio.log import logger
# sys.path.insert(0, "./stable-diffusion")
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from stable_diffusion.ldm.util import instantiate_from_config
from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler
from stable_diffusion.ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

import gc 
# load safety model
safety_model_id = "stable_diffusion/models/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

# ======== create 
import copy
import multiprocessing
import os
import warnings
from types import SimpleNamespace
from typing import overload, List, Optional, Dict, Any, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import threading
    import asyncio

from docarray import DocumentArray, Document

_clip_models_cache = {}

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()
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
    width: Optional[int] = 512,
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
    try:
            
        seed_everything(seed)

        config = OmegaConf.load(f"{config_file}")
        model = load_model_from_config(config, f"{ckpt}")

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)

        if plms:
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)

        os.makedirs(output_dir, exist_ok=True)
        outpath = output_dir

        print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
        wm = "StableDiffusionV1"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

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

