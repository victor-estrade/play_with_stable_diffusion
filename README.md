
# Introduction

Just some code to play with Hugging face's stable diffusion model realease (see https://huggingface.co/blog/stable_diffusion)


https://blog.paperspace.com/generating-images-with-stable-diffusion/



# Requirements 
a .env file with your hugging face token

.env
```
HUGGING_FACE_TOKEN=YOUR_HUGGINGFACE_TOKEN
```


# Current issues

## Not enough RAM
It requires large amount of RAM !

The half precision version works only a CUDA devices...
Tryied using BFloat16 instead of Float16 but it breaks on linear layers.

## ONNX version 

Convert to onnx because it does not work well using pytorch on CPU ?
But it will still require large amont of RAM...
https://huggingface.co/blog/convert-transformers-to-onnx


## Image to image 

https://github.com/replicate/cog-stable-diffusion/blob/main/image_to_image.py