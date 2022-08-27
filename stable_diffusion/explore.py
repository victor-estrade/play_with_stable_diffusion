# FROM THE BLOG POST : https://huggingface.co/blog/stable_diffusion
import os
import click
import logging

from PIL import Image

import torch
from diffusers import StableDiffusionPipeline

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

from . import default
# Configure the logging
from logging.config import dictConfig

dictConfig(default.LOG_DICT_CONFIG)

YOUR_TOKEN = os.environ["HUGGING_FACE_TOKEN"]


# get your token at https://huggingface.co/settings/tokens
# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=YOUR_TOKEN)

# Version float16
# Standard float16 seems to work only on cuda
# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=YOUR_TOKEN)

# Using bfloat16 seems to work on intel CPU
# Or not ... 
# File "/home/estrade/.pyenv/versions/stable_diffusion/lib/python3.10/site-packages/torch/nn/modules/linear.py", line 114, in forward
#     return F.linear(input, self.weight, self.bias)
# RuntimeError: expected scalar type BFloat16 but found Float

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.bfloat16, use_auth_token=YOUR_TOKEN)

print(pipe)

prompt = "a photograph of an astronaut riding a horse"

image = pipe(prompt)["sample"][0]

print(image)
image.save("./output_00.png")

generator = torch.Generator("cuda").manual_seed(1024)
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, generator=generator)["sample"][0]

image.save("./output_01.png")



def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

num_images = 3
prompt = ["a photograph of an astronaut riding a horse"] * num_images
