import logging
import torch

from PIL import Image

from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline

logger = logging.getLogger(__name__)


class TextToImageGenerator():
    def __init__(self, pipe : StableDiffusionPipeline) -> None:
        self.pipe = pipe

    def __call__(
        self,
        prompt:str,
        width:int=512,
        height:int=512,
        num_images=4,
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=None,
        ):
        """ Generate images from given text prompt.

        :param prompt: the sentence prompt to generate the images
        :param width: the width of the images
        :param height: the height of the images
        :param num_images: the number of images to degenerate from the prompt
        :param num_inference_steps: the number of inference step. The more steps the better the image quality but also slower
        :param guidance_scale: the guidance scale (see guided diffusion tricks)
        :param seed: (default None) the seed for the random generator. Use None or 0 to get new seed for each run.

        :return:
            images : the list of PIL Images
            nsfw_content_detected : the list of flag if the content is Not Safe For Work
        """
        generator = torch.Generator(self.pipe.device).manual_seed(seed) if seed else None

        with torch.autocast("cuda"):
            results = [
                self.pipe(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                    )
                for i in range(num_images)]

        images = [image for result in results for image in result["sample"]]
        nsfw_content_detected = [is_nsfw for result in results for is_nsfw in result["nsfw_content_detected"]]
        return images, nsfw_content_detected



class ImageToImageGenerator():
    def __init__(self, pipe : StableDiffusionImg2ImgPipeline) -> None:
        self.pipe = pipe

    def __call__(
        self,
        prompt:str,
        init_image : Image,
        width:int=512,
        height:int=512,
        num_images=4,
        num_inference_steps=50,
        strength:float=1.0,
        guidance_scale=7.5,
        seed=None,
        ):
        """ Generate images from given text prompt.

        :param prompt: 
            the sentence prompt to generate the images
        :param init_image:
            the initial image
        :param width:
            the width of the images
        :param height:
            the height of the images
        :param num_images:
            the number of images to degenerate from the prompt
        :param num_inference_steps:
            the number of inference step. The more steps the better the image quality but also slower
        :param strength:
            Conceptually, indicates how much to transform the reference `init_image`. Must be between 0 and 1.
            `init_image` will be used as a starting point, adding more noise to it the larger the `strength`. The
            number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
            noise will be maximum and the denoising process will run for the full number of iterations specified in
            `num_inference_steps`. A value of 1, therefore, essentially ignores `init_image`.
        :param guidance_scale:
            the guidance scale (see guided diffusion tricks)
        :param seed: (default None)
            the seed for the random generator. Use None or 0 to get new seed for each run.

        :return:
            images :
                the list of PIL Images
            nsfw_content_detected :
                the list of flag if the content is Not Safe For Work
        """
        generator = torch.Generator(self.pipe.device).manual_seed(seed) if seed else None

        with torch.autocast("cuda"):
            results = [
                self.pipe(
                    prompt=prompt,
                    init_image=init_image,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    generator=generator
                    )
                for i in range(num_images)]

        images = [image for result in results for image in result["sample"]]
        nsfw_content_detected = [is_nsfw for result in results for is_nsfw in result["nsfw_content_detected"]]
        return images, nsfw_content_detected
