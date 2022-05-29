import argparse
import cv2
import os
import sys


def setup_enhance(model_path):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    from types import SimpleNamespace

    cfg = {
        "tile": 0,
        "tile_pad": 10,
        "pre_pad": 0,
        "fp32": False,
        "gpu_id": 0,
    }

    args = SimpleNamespace(**cfg)
    # determine models according to model names
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
    )
    netscale = 4

    # determine model paths
    if not os.path.isfile(model_path):
        raise ValueError(f"Model RealESRGAN_x4plus_anime_6B does not exist.")

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id,
    )

    return upsampler
