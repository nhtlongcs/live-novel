# from torch import autocast
# from diffusers import StableDiffusionPipeline

# pipe = StableDiffusionPipeline.from_pretrained(
# 	# "CompVis/stable-diffusion-v1-4", 
#     "/home/nhtlong/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-4/snapshots/fdd29747e61912eb941322ef6f592ae6d0e0de19/"
# 	# use_auth_token=True
# ).to("cuda:1")

# prompt = """There was nothing so very remarkable in that, 
# nor did Alice think it so very much out of the way to hear the Rabbit say to itself, 
# "Oh dear! Oh dear! I shall be too late!" But when the Rabbit actually took a watch out of 
# its waistcoat-pocket and looked at it and then hurried on, Alice started to her feet, for it flashed across 
# her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, 
# and, burning with curiosity, she ran across the field after it and was just in time to see it pop down a large rabbit-hole, 
# under the hedge. In another moment, down went Alice after it! , trending on artstation"""
# with autocast("cuda"):
#     image = pipe(prompt)["sample"][0]  
    
# image.save("astronaut_rides_horse.png")
