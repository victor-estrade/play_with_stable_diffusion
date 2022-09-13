import logging
import click

from . import settings  # Also load/set environment variables
from . import default

import gradio as gr

import numpy as np

from .generators import TextToImageGenerator, ImageToImageGenerator
from .factory import Factory
from .images import auto_glue_image_grid


logger = logging.getLogger(__name__)


class TextToImageBuilder():
    def __init__(self,
        text_to_image_generator : TextToImageGenerator,
        max_num_images:int=9,
        images_per_row:int=3,
        ):
        """ App builder for Text to Image pipeline
        """
        self.text_to_image_generator = text_to_image_generator
        self.max_num_images = max_num_images
        self.images_per_row = images_per_row
    
    @classmethod
    def from_factory(cls,
        factory : Factory,
        max_num_images:int=9,
        images_per_row:int=3,
        device="cuda",
        ):
        """ Alternative constructor using a Factory
        """
        generator = TextToImageGenerator(factory.make_text_to_image_pipeline().to(device))
        new_builder = cls(generator, max_num_images=max_num_images, images_per_row=images_per_row)
        return new_builder


    def __call__(
        self,
        prompt,
        num_image,
        width,
        height,
        seed,
        guidance_scale,
        num_inference_steps,
        ):
        """ Wrapper around the generator.
        It is used as the callable function in gradio's elements to generate the image.
        """
        images, nsfw_content_detected = self.text_to_image_generator(
            prompt=prompt,
            width=width,
            height=height,
            num_images=num_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        print(nsfw_content_detected)

        images_out = [auto_glue_image_grid(images)] + images + [None] * (self.max_num_images - len(images))

        return images_out


    def build(self):
        prompt = gr.Textbox(label="prompt")
        with gr.Row():
            width = gr.Slider(minimum=64, maximum=1024, value=512, step=64, label="width")
            height = gr.Slider(minimum=64, maximum=1024, value=512, step=64, label="height")
        with gr.Row():
            seed = gr.Number(0, label="seed : 0 = no seed")
            guidance_scale = gr.Number(7.5, label="guidance_scale")
        num_inference_steps = gr.Slider(minimum=5, maximum=200, value=50, step=5, label="num_inference_steps")
        num_images = gr.Slider(minimum=1, maximum=self.max_num_images, value=default.NUM_IMAGES, step=1, label="Number of images")

        inputs = [
            prompt, num_images,
            width, height,
            seed,
            guidance_scale,
            num_inference_steps,
            ]

        greet_btn = gr.Button("Generate !")

        out_glued_image = gr.Image(label=f"Generated images")

        with gr.Accordion("Generated individual images"):
            i = 0
            output_images = []
            while i < self.max_num_images:
                with gr.Row():
                    remaining_out_images_to_init = self.max_num_images - i
                    for j in range(min(self.images_per_row, remaining_out_images_to_init)):
                        out_img = gr.Image(label=f"Generated {i+1}")
                        output_images.append(out_img)
                        i += 1
        outputs = [out_glued_image] + output_images
        greet_btn.click(fn=self, inputs=inputs, outputs=outputs)





class ImageToImageBuilder():
    def __init__(self,
        text_to_image_generator : ImageToImageGenerator,
        max_num_images:int=9,
        images_per_row:int=3,
        ):
        """ App builder for Image to Image pipeline
        """
        self.text_to_image_generator = text_to_image_generator
        self.max_num_images = max_num_images
        self.images_per_row = images_per_row
    
    @classmethod
    def from_factory(cls,
        factory : Factory,
        max_num_images:int=9,
        images_per_row:int=3,
        device="cuda",
        ):
        """ Alternative constructor using a Factory
        """
        generator = ImageToImageGenerator(factory.make_image_to_image_pipeline().to(device))
        new_builder = cls(generator, max_num_images=max_num_images, images_per_row=images_per_row)
        return new_builder


    def __call__(
        self,
        prompt,
        init_image,
        num_image,
        seed,
        guidance_scale,
        num_inference_steps,
        strength,
        ):
        """ Wrapper around the generator.
        It is used as the callable function in gradio's elements to generate the image.
        """
        images, nsfw_content_detected = self.text_to_image_generator(
            prompt=prompt,
            init_image=init_image,
            num_images=num_image,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        print(nsfw_content_detected)

        images_out = [auto_glue_image_grid(images)] + images + [None] * (self.max_num_images - len(images))

        return images_out


    def build(self):
        prompt = gr.Textbox(label="prompt")
        init_image = gr.Image(label="Initial image")
        with gr.Row():
            seed = gr.Number(0, label="seed : 0 = no seed")
            guidance_scale = gr.Number(7.5, label="guidance_scale")
        num_inference_steps = gr.Slider(minimum=5, maximum=200, value=50, step=5, label="num_inference_steps")
        strength = gr.Slider(minimum=0., maximum=1.0, value=1.0, step=0.05, label="prompt strength")
        num_images = gr.Slider(minimum=1, maximum=self.max_num_images, value=default.NUM_IMAGES, step=1, label="Number of images")

        inputs = [
            prompt, init_image, num_images,
            seed,
            guidance_scale,
            num_inference_steps,
            strength,
            ]

        greet_btn = gr.Button("Generate !")

        out_glued_image = gr.Image(label=f"Generated images")

        with gr.Accordion("Generated individual images"):
            i = 0
            output_images = []
            while i < self.max_num_images:
                with gr.Row():
                    remaining_out_images_to_init = self.max_num_images - i
                    for j in range(min(self.images_per_row, remaining_out_images_to_init)):
                        out_img = gr.Image(label=f"Generated {i+1}")
                        output_images.append(out_img)
                        i += 1
        outputs = [out_glued_image] + output_images
        greet_btn.click(fn=self, inputs=inputs, outputs=outputs)




@click.command()
@click.option('--share', is_flag=True, show_default=True, default=False,
    help="Share the app. Create a proxy in gradio's website to access the app.")
@click.option('--nsfw', is_flag=True, show_default=True, default=False,
help="Remove image censoring. This is Not Safe For Work !")
def main(share, nsfw):
    """ Main function to run the gradio web app.
    """

    MAX_NUM_IMAGES = 9
    IMAGES_PER_ROW = 3

    logger.info("Loading the models ...")
    FACTORY = Factory(censored=not nsfw)
    TEXT_TO_IMAGE_APP_BUILDER = TextToImageBuilder.from_factory(
        FACTORY,
        max_num_images=MAX_NUM_IMAGES,
        images_per_row=IMAGES_PER_ROW,
        )
    IMAGE_TO_IMAGE_APP_BUILDER = ImageToImageBuilder.from_factory(
        FACTORY,
        max_num_images=MAX_NUM_IMAGES,
        images_per_row=IMAGES_PER_ROW,
        )
    logger.info("Models are ready")


    with gr.Blocks() as demo:
        with gr.Tab("Text to image pipeline") :
            TEXT_TO_IMAGE_APP_BUILDER.build()
        with gr.Tab("Image to image pipeline") :
            IMAGE_TO_IMAGE_APP_BUILDER.build()
    demo.launch(share=share)


if __name__ == "__main__":
    main()
