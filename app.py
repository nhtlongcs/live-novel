"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
from types import MethodDescriptorType
import hashlib

import pandas as pd
import json
from PIL import Image
from pathlib import Path
import torch
from flask import Flask, render_template, request, redirect
from flask import Response
from flask_cors import CORS

import sys

import torch

# from diffusion import *
from types import SimpleNamespace

import cv2
from config import device
import gc
from collections import OrderedDict
import glob

app = Flask(__name__)
CORS(app)


parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
parser.add_argument("--port", default=5000, type=int, help="port number")
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--ngrok", default=False, action="store_true")

_args = parser.parse_args()
DEBUG = _args.debug
DREAM_URL = "/"
cache = OrderedDict()
cacheLengthMax = 65536


def init_cache(foldername):
    cache = {}
    exist_ims = [os.path.basename(x) for x in glob.glob(f"{foldername}/*.png")][:65536]
    for imp in exist_ims:
        cache[imp.split(".")[0]] = os.path.join(foldername, imp)
    return cache


if not (DEBUG):
    from diffusion import (
        do_run,
        args,
        create_model_and_diffusion,
        split_prompts,
        model_config,
        model_path,
        diffusion_model,
    )
    from ESRGAN.wrapper import setup_enhance

    print("Prepping model...")
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(
        torch.load(f"{model_path}/{diffusion_model}.pt", map_location="cpu")
    )
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
            param.requires_grad_()
    if model_config["use_fp16"]:
        model.convert_to_fp16()

    enhancer = setup_enhance(f"{model_path}/RealESRGAN_x4plus_anime_6B.pth")

    gc.collect()
    torch.cuda.empty_cache()
    cache = init_cache("output")


else:
    import numpy as np
    import cv2 as cv

    os.makedirs("output_simulate/", exist_ok=True)
    cache = init_cache("output_simulate")


def get_hashid(prompt, seed):
    if prompt:
        text_prompts = {
            0: prompt.split("|"),
        }
        # not necessary but makes the cli output easier to parse
        for x in range(len(text_prompts[0])):
            text_prompts[0][x] = text_prompts[0][x].strip()
    hashid = hashlib.md5("-".join(text_prompts[0] + [str(seed)]).encode()).hexdigest()

    return hashid, text_prompts


def enhance(path):
    """Inference demo for Real-ESRGAN."""
    try:
        print("upscaling")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        output, _ = enhancer.enhance(img, outscale=4)
        cv2.imwrite(path, output)
        del img
    except Exception as e:
        print(f"upscale failed : \n{e}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def run_simulate(prompt, seed):
    def generate_random_image(image_path):
        data = np.random.randint(0, 255, size=(1000, 1000, 3), dtype=np.uint8)
        cv.imwrite(image_path, data)

    hashid, text_prompts = get_hashid(prompt, seed)
    if hashid in cache:
        print(f"Using cached image for {hashid}")
        return cache[hashid]

    req = {
        "seed": seed,
        "output_filename": f"{hashid}.png",
        "output_dir": "output_simulate/",
    }

    im_path = os.path.join(req["output_dir"], req["output_filename"])
    try:
        generate_random_image(im_path)
        with open("output_simulate/log.txt", "a") as fd:
            fd.write(f"\n{hashid}.png, {prompt}")
        cache[hashid] = f"output_simulate/{hashid}.png"
        if len(cache) > cacheLengthMax:
            cache.popitem(last=False)
    except KeyboardInterrupt:
        pass
    finally:
        gc.collect()

    return im_path


def run(prompt, seed):
    global args

    hashid, text_prompts = get_hashid(prompt, seed)

    if hashid in cache:
        print(f"Using cached image for {hashid}")
        return cache[hashid]

    req = {
        "seed": seed,
        "prompts_series": split_prompts(text_prompts) if text_prompts else None,
        "output_filename": f"{hashid}.png",
        "output_dir": "output/",
    }

    args.update(req)
    args_ns = SimpleNamespace(**args)
    im_path = os.path.join(req["output_dir"], req["output_filename"])
    try:
        do_run(args_ns, model, diffusion)
        enhance(im_path)

        with open("output/log.txt", "a") as fd:
            fd.write(f"\n{hashid}.png, {prompt}")
        cache[hashid] = f"output/{hashid}.png"

        if len(cache) > cacheLengthMax:
            cache.popitem(last=False)

    except KeyboardInterrupt:
        pass
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    return im_path


def image_to_byte_array(image: Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format="PNG")
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


@app.route(DREAM_URL, methods=["POST"])
def dream():
    if not request.method == "POST":
        return
    if request.get_data():
        data = request.get_data()
        json_data = json.loads(data.decode())
        print("=" * 10)
        print(json_data)
        print("=" * 10)
        prompt = json_data["text"]
        style = json_data.get("style", "watercolor")
        seed = json_data.get("seed", 2586166778)
        prompt = f"{prompt} | text:-0.99 | watermark:-0.99 | logo:-0.99 | {style}"
        im_path = run_simulate(prompt, seed) if DEBUG else run(prompt, seed)
        imgByteArr = image_to_byte_array(Image.open(im_path))
        return imgByteArr


if __name__ == "__main__":

    if _args.ngrok:
        from flask_ngrok import run_with_ngrok

        run_with_ngrok(app)  # Start ngrok when app is run

        app.run(threaded=True)  # debug=True causes Restarting with stat
    else:
        app.run(
            host="0.0.0.0", port=_args.port, threaded=True
        )  # debug=True causes Restarting with stat
