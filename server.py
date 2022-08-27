
import gc
import io

import numpy as np
import PIL.Image as Image
import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from jina import Client

c = Client(host='grpc://0.0.0.0:51001')

router = APIRouter()

def image_to_byte_array(image: Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format="PNG")
    imgByteArr = imgByteArr.getvalue()
    tmp = imgByteArr.decode("latin-1")
    return tmp

resolution_factory = {
    '1x1': (512, 512),
    '4x6': (512, 768),
    '6x4': (768, 512),
}

@router.post(
    "/create",
    summary='Create a new prompt',
)
def create_image(
    prompt: str = None,
    sess_name: str = None,
    ddim_steps: int = 50,
    scale: float = 7.5,
    seed: int = 42,
    precision: str = "autocast",
    ratio: str = "4x6",
):
    global c 
    assert sess_name is not None, 'sess_name is required'
    assert prompt is not None, 'prompt is required'
    height, width = resolution_factory[ratio]

    c.post(
        '/create',
        parameters={
            'sess_name': sess_name,
            'prompt': prompt,
            'ddim_steps': ddim_steps,
            'height': height,
            'width': width,
            'scale': scale,
            'seed': seed,
            'precision': precision,
        }
    )

    
@router.post(
    "/result",
    summary='Get the result of the prompt',
)
def get_result(sess_name: str):
    global c 
    assert sess_name is not None, 'sess_name is required'
    gc.collect()
    da = c.post(
        '/result',
        parameters={
            'sess_name': sess_name,
        }
    )
    np_img = da.tensors[0]
    img = Image.fromarray(np_img.astype(np.uint8))
    imgByteArr = image_to_byte_array(img)
    return {'image': imgByteArr}
    

app = FastAPI(
    name = "Live-Novel FastAPI Server", 
    docs_url = "/docs", 
    redoc_url = "/redoc",
)

# Setup CORS policy for FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

app.include_router(router)

def run_server():
    """Run server."""

    # start the server!
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=12345,
    )

if __name__ == '__main__':
    run_server()
