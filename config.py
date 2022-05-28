import argparse
import torch
from types import SimpleNamespace

iargs = {
    "RN101": "no",
    "RN50": "yes",
    "RN50x16": "no",
    "RN50x4": "no",
    "RN50x64": "no",
    "VitB16": "yes",
    "VitB32": "yes",
    "VitL14": "yes",
    "cutn": 4,
    "init_image": None,
    "max_iterations": 250,
    "size": [800, 480],
    "skip_steps": -1,
}
iargs = SimpleNamespace(**iargs)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
