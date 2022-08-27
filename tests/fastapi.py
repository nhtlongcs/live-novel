"""Perform test request"""
import pprint
from urllib import response
import requests
import json
from PIL import Image
import io
import numpy as np
API = f"http://0.0.0.0:12345"

run_id = np.random.randint(0, 1000000)
seed = np.random.randint(0, 100)

data = {
    'sess_name': str(run_id),
    'prompt': 'a realistic photo of an spiderman riding bicycle trending on artstation',
    'seed': seed,
}

def byte_array_to_image(byte_array):
    """Convert byte array to image"""
    imgByteArr = byte_array.encode('latin-1')
    return Image.open(io.BytesIO(imgByteArr))


requests.post(f'{API}/create', params=data)
response = requests.post(f'{API}/result', params={'sess_name': run_id})

response = response.json()
im = byte_array_to_image(response['image'])
im.save("test.png")
