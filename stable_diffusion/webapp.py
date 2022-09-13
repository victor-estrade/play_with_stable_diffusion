import logging
import click

from . import settings  # Also load/set environment variables
from . import default

import gradio as gr

import numpy as np

from .generators import generate_from_prompt, TextToImageGenerator
from .factory import Factory
from .images import auto_glue_image_grid


logger = logging.getLogger(__name__)



@click.command()
@click.option('--nsfw', is_flag=True, show_default=True, default=False, help="Remove image censoring. This is Not Safe For Work !")
def main(nsfw):
    """ Main function to run the gradio web app.
    """

    logger.info("Loading the models ...")
    FACTORY = Factory(censored=not nsfw)
    GENERATOR = TextToImageGenerator(FACTORY.make_simple_pipeline().to("cuda"))
    logger.info("Models are ready")

    MAX_NUM_IMAGES = 9
    IMAGES_PER_ROW = 3


    def generate_images(
        prompt,
        num_image,
        width,
        height,
        seed,
        guidance_scale,
        num_inference_steps,
        prompt_strength,
        ):
        """ Wrapper around the generator.
        """
        images, nsfw_content_detected = GENERATOR(
            prompt=prompt,
            width=width,
            height=height,
            num_images=num_image,
            num_inference_steps=num_inference_steps,
            prompt_strength=prompt_strength,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        print(nsfw_content_detected)

        images_out = [auto_glue_image_grid(images)] + images + [None] * (MAX_NUM_IMAGES - len(images))

        return images_out

    with gr.Blocks() as demo:
        prompt = gr.Textbox(label="prompt")
        with gr.Row():
            width = gr.Slider(minimum=64, maximum=1024, value=512, step=64, label="width")
            height = gr.Slider(minimum=64, maximum=1024, value=512, step=64, label="height")
        with gr.Row():
            seed = gr.Number(0, label="seed : 0 = no seed")
            guidance_scale = gr.Number(7.5, label="guidance_scale")
        num_inference_steps = gr.Slider(minimum=5, maximum=200, value=50, step=5, label="num_inference_steps")
        prompt_strength = gr.Slider(minimum=0., maximum=1.0, value=1.0, step=0.05, label="prompt_strength")
        num_images = gr.Slider(minimum=1, maximum=MAX_NUM_IMAGES, value=default.NUM_IMAGES, step=1, label="Number of images")

        inputs = [
            prompt, num_images,
            width, height,
            seed,
            guidance_scale,
            num_inference_steps,
            prompt_strength,
            ]
        
        greet_btn = gr.Button("Generate !")

        out_glued_image = gr.Image(label=f"Generated images")

        with gr.Accordion("Generated individual images"):
            i = 0
            output_images = []
            while i < MAX_NUM_IMAGES:
                with gr.Row():
                    remaining_out_images_to_init = MAX_NUM_IMAGES - i
                    for j in range(min(IMAGES_PER_ROW, remaining_out_images_to_init)):
                        out_img = gr.Image(label=f"Generated {i+1}")
                        output_images.append(out_img)
                        i += 1
        outputs = [out_glued_image] + output_images
        greet_btn.click(fn=generate_images, inputs=inputs, outputs=outputs)
    demo.launch()

if __name__ == "__main__":
    main()
