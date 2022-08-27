
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

logger = logging.getLogger(__name__)

HUGGING_FACE_TOKEN = os.environ["HUGGING_FACE_TOKEN"]

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = default.CUDA_VISIBLE_DEVICE



def load_model(device="cuda", half_precision=True, auth_token=HUGGING_FACE_TOKEN):
    """Load the diffusion model

    :param device: (default="cuda") the device on which to put the model (example : "cpu", "cuda")
    :param half_precision: (default="fp16") the half_precision version of the model. 
        Use None to get the oriinaal Float32 model and "fp16" to load the Float16 version.
    :param :

    :return: pipe. The loaded model.
    """
    if half_precision:
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=auth_token
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            use_auth_token=auth_token
        )
    pipe = pipe.to(device)
    return pipe


def generate(
    pipe,
    prompt:str,
    num_images=4,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=None
    ):
    """
    :param pipe: the diffusion model
    :param prompt: the sentence prompt to generate the images
    :param num_images: the number of images to degenerate from the prompt
    :param num_inference_steps: the number of inference step. The more steps the better the image quality but also slower
    :param guidance_scale: the guidance scale (see guided diffusion tricks)
    :param seed: (default None) the seed for the random generator. Use None to get new seed for each run.

    """
    generator = torch.Generator(pipe.device).manual_seed(seed) if seed is not None else None

    with torch.autocast("cuda"):
        result = pipe(
            [prompt] * num_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
    images = result["sample"]
    nsfw_content_detected = result["nsfw_content_detected"]
    return images, nsfw_content_detected



def glue_image_grid(images, n_rows=default.N_ROWS, n_cols=default.N_COLS):
    assert len(images) == n_rows * n_cols

    w, h = images[0].size
    grid = Image.new('RGB', size=(n_cols*w, n_rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(images):
        grid.paste(img, box=(i % n_cols * w, i // n_cols * h))
    return grid

def auto_glue_image_grid(images):
    n_images = len(images)
    n_cols = 1
    n_rows = n_images
    if n_images % 5 == 0:
        n_cols = 5
        n_rows = n_images // 5
    elif n_images % 4 == 0:
        n_cols = 4
        n_rows = n_images // 4
    elif n_images % 3 == 0:
        n_cols = 3
        n_rows = n_images // 3
    elif n_images % 2 == 0:
        n_cols = 2
        n_rows = n_images // 2
    image_grid = glue_image_grid(images, n_rows=n_rows, n_cols=n_cols)
    return image_grid


def save_images(images, out_dir=default.OUTPUT_DIR):
    for i, image in enumerate(images):
        fname = f"image_{i:02d}.png"
        image.save(out_dir / fname)
    image_grid = auto_glue_image_grid(images)
    image_grid.save(out_dir / "image_grid.png")



num_images = 3
prompt = ["a photograph of an astronaut riding a horse"] * num_images



@click.Command()
def main():
    """

    source : https://huggingface.co/blog/stable_diffusion
    """
    pipe = load_model(half_precision=True, device="cuda")
    PROMPT = "Super cute bunny eating a carrot"
    NUM_IMAGES = 1
    images, nsfw_content_detected = generate(pipe, PROMPT, num_images=NUM_IMAGES)
    for i, nsfw in enumerate(nsfw_content_detected):
        if nsfw:
            logger.info(f"Not Safe For Work content detected in image {i}")




if __name__ == "__main__":
    main()