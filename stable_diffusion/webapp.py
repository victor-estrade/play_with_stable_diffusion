import logging
from . import settings

import gradio as gr

import numpy as np

from .models import load_model
from .generators import generate_from_prompt

logger = logging.getLogger(__name__)

PIPE = load_model(
    device="cuda",
    half_precision=True,
    auth_token=settings.HUGGING_FACE_TOKEN,
    censor_nsfw=False,
    )

def placeholder(prompt, num_image, width, height, *args, **kwargs):
    noise_image = np.random.rand(height, width, 3)
    return noise_image


def generate_image(
    prompt,
    num_image,
    width,
    height,
    seed,
    guidance_scale,
    num_inference_steps,
    prompt_strength,
    ):
    """ Generate the image
    """
    images, nsfw_content_detected = generate_from_prompt(
        PIPE,
        prompt=prompt,
        width=width,
        height=height,
        num_images=num_image,
        num_inference_steps=num_inference_steps,
        prompt_strength=prompt_strength,
        guidance_scale=guidance_scale,
        seed=seed,
    )

   

def main():

    with gr.Blocks() as demo:
        prompt = gr.Textbox(label="prompt")
        num_images = gr.Number(4, label="num_images")
        width = gr.Slider(minimum=64, maximum=1024, value=512, step=64, label="width")
        height = gr.Slider(minimum=64, maximum=1024, value=512, step=64, label="height")
        seed = gr.Number(0, label="seed : 0 = no seed")
        guidance_scale = gr.Number(7.5, label="guidance_scale")
        num_inference_steps = gr.Slider(minimum=5, maximum=200, value=50, step=5, label="num_inference_steps")
        prompt_strength = gr.Slider(minimum=0., maximum=1.0, value=1.0, step=0.05, label="prompt_strength")

        inputs = [
            prompt, num_images,
            width, height,
            seed,
            guidance_scale,
            num_inference_steps,
            prompt_strength,
            ]

        output = gr.Image(label="Generated images")
        greet_btn = gr.Button("Generate !")
        greet_btn.click(fn=generate_image, inputs=inputs, outputs=output)

    demo.launch()

if __name__ == "__main__":
    main()