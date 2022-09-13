import typing as T
import numpy as np
import torch

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer


from . import settings
from .utils import PIL_IMAGE_RESAMPLING
from .pipelines import UncensoredSafetyChecker


class Factory:
    def __init__(self, 
        model_checkpoint="CompVis/stable-diffusion-v1-4", 
        device="cuda",  # Unused for now
        half_precision=True,
        enable_attention_slicing=False,
        auth_token=settings.HUGGING_FACE_TOKEN,
        censored=True,
        ):
        """ Factory of pipelines

        Avoid loading twice the same neural networks.

        :param model_checkpoint: the model checkpoint
        :param device: (Unused for now) the device on which send the pipeline
        :param half_precision: Load the model using half precision (only works on GPU)
        :param enable_attention_slicing: optimization making the model use less memory but slightly more compute time
        :param auth_token: the HuggingFace Token since downloading these models requires to agree with its Licence.

        """
        self.model_checkpoint = model_checkpoint
        self.censored = censored
        self.device = device
        self.half_precision = half_precision
        self.enable_attention_slicing = enable_attention_slicing
        self.vae = self._make_vae(model_checkpoint, half_precision, auth_token).to(device)
        self.text_encoder = self._make_text_encoder(model_checkpoint, half_precision, auth_token).to(device)
        self.tokenizer = self._make_tokenizer(model_checkpoint, half_precision, auth_token)
        self.unet = self._make_unet(model_checkpoint, half_precision, auth_token).to(device)
        self.scheduler = self._make_scheduler(model_checkpoint, half_precision, auth_token)
        self.safety_checker = self._make_safety_checker(model_checkpoint, half_precision, censored, auth_token).to(device)
        self.feature_extractor = self._make_feature_extractor(model_checkpoint, half_precision, auth_token)

    # ================
    # Inner factory to load the models
    # ================
    def _make_vae(self, model_checkpoint, half_precision, auth_token):
        if half_precision:
            vae = AutoencoderKL.from_pretrained(
                model_checkpoint,
                subfolder="vae",
                revision="fp16",
                torch_dtype=torch.float16,
                use_auth_token=auth_token)
        else:
            vae = AutoencoderKL.from_pretrained(
                model_checkpoint,
                subfolder="vae",
                use_auth_token=auth_token)
        return vae

    def _make_text_encoder(self, model_checkpoint, half_precision, auth_token):
        if half_precision:
            text_encoder = CLIPTextModel.from_pretrained(
                model_checkpoint,
                subfolder="text_encoder",
                revision="fp16",
                torch_dtype=torch.float16,
                use_auth_token=auth_token)
        else:
            text_encoder = CLIPTextModel.from_pretrained(
                model_checkpoint,
                subfolder="text_encoder",
                use_auth_token=auth_token)
        return text_encoder

    def _make_tokenizer(self, model_checkpoint, half_precision, auth_token):
        if half_precision:
            tokenizer = CLIPTokenizer.from_pretrained(
                model_checkpoint,
                subfolder="tokenizer",
                revision="fp16",
                torch_dtype=torch.float16,
                use_auth_token=auth_token)
        else:
            tokenizer = CLIPTokenizer.from_pretrained(
                model_checkpoint,
                subfolder="tokenizer",
                use_auth_token=auth_token)
        return tokenizer

    def _make_unet(self, model_checkpoint, half_precision, auth_token):
        if half_precision:
            unet = UNet2DConditionModel.from_pretrained(
                model_checkpoint,
                subfolder="unet",
                revision="fp16",
                torch_dtype=torch.float16,
                use_auth_token=auth_token)
        else:
            unet = UNet2DConditionModel.from_pretrained(
                model_checkpoint,
                subfolder="unet",
                use_auth_token=auth_token)
        return unet

    def _make_scheduler(self, model_checkpoint, half_precision, auth_token):
        scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        return scheduler

    def _make_safety_checker(self, model_checkpoint, half_precision, censored, auth_token):
        safety_class = StableDiffusionSafetyChecker if censored else UncensoredSafetyChecker
        if half_precision:
            safety_checker = safety_class.from_pretrained(
                model_checkpoint,
                subfolder="safety_checker",
                revision="fp16",
                torch_dtype=torch.float16,
                use_auth_token=auth_token)
        else:
            safety_checker = safety_class.from_pretrained(
                model_checkpoint,
                subfolder="safety_checker",
                use_auth_token=auth_token)
        return safety_checker

    def _make_feature_extractor(self, model_checkpoint, half_precision, auth_token):
        feature_extractor = CLIPFeatureExtractor(
            crop_size = 224,
            do_center_crop = True,
            do_convert_rgb = True,
            do_normalize = True,
            do_resize = True,
            image_mean = [
                0.48145466,
                0.4578275,
                0.40821073
            ],
            image_std = [
                0.26862954,
                0.26130258,
                0.27577711
            ],
            resample = PIL_IMAGE_RESAMPLING.BICUBIC,
            size = 224            
        )
        return feature_extractor
    
    # ================
    # Factory Making pipelines
    # ================

    def make_text_to_image_pipeline(self):
        pipe = StableDiffusionPipeline(
            self.vae,
            self.text_encoder,
            self.tokenizer,
            self.unet,
            self.scheduler,
            self.safety_checker,
            self.feature_extractor
        )
        if self.enable_attention_slicing:
            pipe.enable_attention_slicing()
        return pipe

    def make_image_to_image_pipeline(self):
        pipe = StableDiffusionImg2ImgPipeline(
            self.vae,
            self.text_encoder,
            self.tokenizer,
            self.unet,
            self.scheduler,
            self.safety_checker,
            self.feature_extractor
        )
        if self.enable_attention_slicing:
            pipe.enable_attention_slicing()
        return pipe

    def make_inpaint_pipeline(self):
        pipe = StableDiffusionInpaintPipeline(
            self.vae,
            self.text_encoder,
            self.tokenizer,
            self.unet,
            self.scheduler,
            self.safety_checker,
            self.feature_extractor
        )
        if self.enable_attention_slicing:
            pipe.enable_attention_slicing()
        return pipe
