import torch
from .pipelines import StableDiffusionImg2ImgPipeline
from . import settings


def load_model(device="cuda", half_precision=True, auth_token=settings.HUGGING_FACE_TOKEN):
    """Load the diffusion model

    :param device: (default="cuda") the device on which to put the model (example : "cpu", "cuda")
    :param half_precision: (default="fp16") the half_precision version of the model. 
        Use None to get the oriinaal Float32 model and "fp16" to load the Float16 version.
    :param auth_token:
    :param censor_nsfw:

    :return: pipe. The loaded model.
    """
    if half_precision:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=auth_token
        )
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            use_auth_token=auth_token
        )
    pipe = pipe.to(device)
    return pipe
