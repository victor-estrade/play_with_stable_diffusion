# FROM THE BLOG POST : https://huggingface.co/blog/stable_diffusion
import os

from PIL import Image
import torch
from diffusers import StableDiffusionPipeline

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

from . import default
YOUR_TOKEN = os.environ["HUGGING_FACE_TOKEN"]

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = default.CUDA_VISIBLE_DEVICE


# get your token at https://huggingface.co/settings/tokens
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=YOUR_TOKEN)

# Version float16
# Seems to work only on cuda
# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=YOUR_TOKEN)

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
