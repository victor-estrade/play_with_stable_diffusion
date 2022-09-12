import logging
import torch

logger = logging.getLogger(__name__)


def generate_from_prompt(
    pipe,
    prompt:str,
    width:int=512,
    height:int=512,
    num_images=4,
    num_inference_steps=50,
    prompt_strength:float=1.0,
    guidance_scale=7.5,
    seed=None,
    ):
    """
    :param pipe: the diffusion model
    :param prompt: the sentence prompt to generate the images
    :param num_images: the number of images to degenerate from the prompt
    :param num_inference_steps: the number of inference step. The more steps the better the image quality but also slower
    :param guidance_scale: the guidance scale (see guided diffusion tricks)
    :param seed: (default None) the seed for the random generator. Use None to get new seed for each run.

    """
    generator = torch.Generator(pipe.device).manual_seed(seed) if seed else None

    with torch.autocast("cuda"):
        results = [
            pipe(
                prompt=prompt,
                width=width,
                height=height,
                mask=None,
                init_image=None,
                num_inference_steps=num_inference_steps,
                prompt_strength=prompt_strength,
                guidance_scale=guidance_scale,
                generator=generator
                )
            for i in range(num_images)]

    images = [image for result in results for image in result["sample"]]
    nsfw_content_detected = [is_nsfw for result in results for is_nsfw in result["nsfw_content_detected"]]
    return images, nsfw_content_detected

