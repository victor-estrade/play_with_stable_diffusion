
# Introduction

Just some code to play with Hugging face's stable diffusion model realease (see https://huggingface.co/blog/stable_diffusion)


https://blog.paperspace.com/generating-images-with-stable-diffusion/



# Install

0. Go to your working directory:

```bash
cd path/to/this_project
```

1. Clone the repo
```bash
git clone https://github.com/victor-estrade/play_with_stable_diffusion.git
```

2. Go inside the repo

```bash
cd play_with_stable_diffusion
```

3. Inititalize pyenv
```bash
make init
```
4. activate your new pyenv
```bash
pyenv shell stable_diffusion
```

5. Install

```bash
make install
```

# Requirements and custom settings


## Hugging Face token

You will need to have a .env file with your hugging face token in your workspace.

.env
```
HUGGING_FACE_TOKEN=YOUR_HUGGINGFACE_TOKEN
```

## CUDA Setting

1. see which GPU on your machine is available using nvtop
```bash
nvtop
```

2. change the cuda visible devices to use the available GPU in your .env file

example in your .env file
```
CUDA_VISIBLE_DEVICES=0
```

## Change Gradio port

If you want to change the port gradio is using you can specify it in your .env file.

example in your .env file
```
CUDA_VISIBLE_DEVICES=7860
```

## Complete example of .env file
```
HUGGING_FACE_TOKEN=YOUR_HUGGINGFACE_TOKEN
CUDA_VISIBLE_DEVICES=0
GRADIO_SERVER_PORT=7860
```
