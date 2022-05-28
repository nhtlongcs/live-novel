"""Perform test request"""
import pprint
import requests
import json
from PIL import Image
import io

API = f"http://nhtlongcs.com:5000"

data = {
    "text": "a shining lighthouse on the shore of a tropical island in a raging storm",
    "style": "watercolor",
    "seed": 1234,
}


def byte_array_to_image(byte_array):
    """Convert byte array to image"""
    return Image.open(io.BytesIO(byte_array))


response = requests.post(API, data=json.dumps(data))
im = byte_array_to_image(response.content)
im.save("test.png")
