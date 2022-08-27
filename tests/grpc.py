from jina import Client, Document
import PIL.Image as Image
import numpy as np 
c = Client(host='grpc://0.0.0.0:51001')
# random run id  
run_id = np.random.randint(0, 1000000)
da = c.post(
    '/create',
    parameters={
        'sess_name': run_id,
        'prompt': 'a realistic photo of an spiderman riding bike on New York street trending on artstation',
    },
)

# check intermediate results
da = c.post('/result', parameters={'sess_name': run_id})
print(da.summary())
np_img = da.tensors[0]
img = Image.fromarray(np_img.astype(np.uint8))
img.save("demo.png")