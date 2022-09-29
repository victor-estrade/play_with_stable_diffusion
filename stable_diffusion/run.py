import os
import click
import logging

from PIL import Image
import numpy as np

import torch
from diffusers import StableDiffusionPipeline


from . import settings
from . import default
from .models import load_model
from .generators import generate_from_prompt

logger = logging.getLogger(__name__)


@click.Command()
def main():
    """

    source : https://huggingface.co/blog/stable_diffusion
    """
    pipe = load_model(half_precision=True, device="cuda")
    PROMPT = "Super cute bunny eating a carrot"
    NUM_IMAGES = 1
    images, nsfw_content_detected = generate_from_prompt(
        pipe, PROMPT, num_images=NUM_IMAGES
    )
    for i, nsfw in enumerate(nsfw_content_detected):
        if nsfw:
            logger.info(f"Not Safe For Work content detected in image {i}")


if __name__ == "__main__":
    main()
