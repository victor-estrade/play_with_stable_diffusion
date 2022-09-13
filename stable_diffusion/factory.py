from typing import List, Optional, Union, Tuple

import numpy as np
import torch

from PIL import Image
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer


from . import settings


class Factory:
    def __init__(self, 
        model_chekpoint="CompVis/stable-diffusion-v1-4", 
        device="cuda",
        half_precision=True,
        enable_attention_slicing=False,
        auth_token=settings.HUGGING_FACE_TOKEN,
        ):
        """ Factory of pipelines

        Avoid loading twice the same neural networks.

        :param model_checkpoint: the model checkpoint
        :param device: (Unused for now) the device on which send the pipeline
        :param half_precision: Load the model using half precision (only works on GPU)
        :param enable_attention_slicing: optimization making the model use less memory but slightly more compute time
        :param auth_token: the HuggingFace Token since downloading these models requires to agree with its Licence.

        """
        self.model_chekpoint = model_chekpoint
        self.device = device
        self.half_precision = half_precision
        self.enable_attention_slicing = enable_attention_slicing
        self.vae = self._make_vae(model_chekpoint, half_precision, auth_token)
        self.text_encoder = self._make_text_encoder(model_chekpoint, half_precision, auth_token)
        self.tokenizer = self._make_tokenizer(model_chekpoint, half_precision, auth_token)
        self.unet = self._make_unet(model_chekpoint, half_precision, auth_token)
        self.scheduler = self._make_scheduler(model_chekpoint, half_precision, auth_token)
        self.safety_checker = self._make_safety_checker(model_chekpoint, half_precision, auth_token)
        self.feature_extractor = self._make_feature_extractor(model_chekpoint, half_precision, auth_token)

    # ================
    # Inner factory to load the models
    # ================
    def _make_vae(self, model_chekpoint, half_precision, auth_token):
        if half_precision:
            vae = AutoencoderKL.from_pretrained(
                model_chekpoint,
                subfolder="vae",
                revision="fp16",
                torch_dtype=torch.float16,
                use_auth_token=auth_token)
        else:
            vae = AutoencoderKL.from_pretrained(
                model_chekpoint,
                subfolder="vae",
                use_auth_token=auth_token)
        return vae

    def _make_text_encoder(self, model_chekpoint, half_precision, auth_token):
        if half_precision:
            text_encoder = CLIPTextModel.from_pretrained(
                model_chekpoint,
                subfolder="text_encoder",
                revision="fp16",
                torch_dtype=torch.float16,
                use_auth_token=auth_token)
        else:
            text_encoder = CLIPTextModel.from_pretrained(
                model_chekpoint,
                use_auth_token=auth_token)
        return text_encoder

    def _make_tokenizer(self, model_chekpoint, half_precision, auth_token):
        if half_precision:
            tokenizer = CLIPTokenizer.from_pretrained(
                model_chekpoint,
                subfolder="tokenizer",
                revision="fp16",
                torch_dtype=torch.float16,
                use_auth_token=auth_token)
        else:
            tokenizer = CLIPTokenizer.from_pretrained(
                model_chekpoint,
                use_auth_token=auth_token)
        return tokenizer

    def _make_unet(self, model_chekpoint, half_precision, auth_token):
        if half_precision:
            unet = UNet2DConditionModel.from_pretrained(
                model_chekpoint,
                subfolder="unet",
                revision="fp16",
                torch_dtype=torch.float16,
                use_auth_token=auth_token)
        else:
            unet = UNet2DConditionModel.from_pretrained(
                model_chekpoint,
                use_auth_token=auth_token)
        return unet

    def _make_scheduler(self, model_chekpoint, half_precision, auth_token):
        if half_precision:
            scheduler = DDIMScheduler.from_pretrained(
                model_chekpoint,
                subfolder="scheduler",
                revision="fp16",
                torch_dtype=torch.float16,
                use_auth_token=auth_token)
        else:
            scheduler = DDIMScheduler.from_pretrained(
                model_chekpoint,
                subfolder="scheduler",
                use_auth_token=auth_token)
        return scheduler

    def _make_safety_checker(self, model_chekpoint, half_precision, auth_token):
        if half_precision:
            safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                model_chekpoint,
                subfolder="safety_checker",
                revision="fp16",
                torch_dtype=torch.float16,
                use_auth_token=auth_token)
        else:
            safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                model_chekpoint,
                subfolder="safety_checker",
                use_auth_token=auth_token)
        return safety_checker

    def _make_feature_extractor(self, model_chekpoint, half_precision, auth_token):
        if half_precision:
            feature_extractor = CLIPFeatureExtractor.from_pretrained(
                model_chekpoint,
                subfolder="feature_extractor",
                revision="fp16",
                torch_dtype=torch.float16,
                use_auth_token=auth_token)
        else:
            feature_extractor = CLIPFeatureExtractor.from_pretrained(
                model_chekpoint,
                subfolder="feature_extractor",
                use_auth_token=auth_token)
        return feature_extractor
    
    # ================
    # Factory Making pipelines
    # ================

    def make_simple(self):
        pipe = StableDiffusionPipeline(
            self.vae,
            self.text_encoder,
            self.tokenizer,
            self.unet,
            self.scheduler,
            self.safety_checker,
            self.feature_extractor
        )
        return pipe

    def make_image_to_image(self):
        pipe = StableDiffusionImg2ImgPipeline(
            self.vae,
            self.text_encoder,
            self.tokenizer,
            self.unet,
            self.scheduler,
            self.safety_checker,
            self.feature_extractor
        )
        return pipe

    def make_inpaint(self):
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
