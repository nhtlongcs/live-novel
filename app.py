"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
from types import MethodDescriptorType

import pandas as pd
import json
from PIL import Image
from pathlib import Path
import torch
from flask import Flask, render_template, request, redirect
from flask import Response
import sys


app = Flask(__name__)

DREAM_URL = "/"
# run_cmd = """
# python  diffusion.py -s 800 480 -i 250 -cuts 4 \
#         -p "{}" \
#         -dvitb32 yes -dvitb16 yes -dvitl14 yes -drn101 no -drn50 yes \
#         -drn50x4 no -drn50x16 no -drn50x64 no -sd 2586166778 \
#         -o output/result.png
# """
run_cmd = """
python  diffusion.py -s 64 64 -i 2 -cuts 1 \
        -p "{}" \
        -dvitb32 yes -dvitb16 no -dvitl14 no -drn101 no -drn50 yes \
        -drn50x4 no -drn50x16 no -drn50x64 no -sd 2586166778 \
        -o output/result.png
"""


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
        promt = json_data["text"]
        style = json_data.get("style", "watercolor")
        text = f"{promt} | text:-0.99 | watermark:-0.99 | logo:-0.99 | {style}"
        print(run_cmd.format(text))
        os.system(run_cmd.format(text))
        imgByteArr = image_to_byte_array(Image.open("output/im.png"))
        return imgByteArr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument("--ngrok", default=False, action="store_true")

    args = parser.parse_args()

    if args.ngrok:
        from flask_ngrok import run_with_ngrok

        run_with_ngrok(app)  # Start ngrok when app is run

        app.run(threaded=True)  # debug=True causes Restarting with stat
    else:
        app.run(
            host="0.0.0.0", port=args.port, threaded=True
        )  # debug=True causes Restarting with stat
