import logging

from PIL import Image
import numpy as np

import torch

from . import default



def preprocess_init_image(image: Image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0



def preprocess_mask(mask: Image, width: int, height: int):
    mask = mask.convert("L")
    mask = mask.resize((width // 8, height // 8), resample=Image.Resampling.LANCZOS)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    mask = torch.from_numpy(mask)
    return mask




def glue_image_grid(images, n_rows=default.N_ROWS, n_cols=default.N_COLS):
    assert len(images) == n_rows * n_cols

    w, h = images[0].size
    grid = Image.new('RGB', size=(n_cols*w, n_rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(images):
        grid.paste(img, box=(i % n_cols * w, i // n_cols * h))
    return grid



def auto_glue_image_grid(images):
    n_images = len(images)
    n_cols = 1
    n_rows = n_images
    if n_images % 5 == 0:
        n_cols = 5
        n_rows = n_images // 5
    elif n_images % 4 == 0:
        n_cols = 4
        n_rows = n_images // 4
    elif n_images % 3 == 0:
        n_cols = 3
        n_rows = n_images // 3
    elif n_images % 2 == 0:
        n_cols = 2
        n_rows = n_images // 2
    image_grid = glue_image_grid(images, n_rows=n_rows, n_cols=n_cols)
    return image_grid


def save_images(images, out_dir=default.OUTPUT_DIR):
    for i, image in enumerate(images):
        fname = f"image_{i:02d}.png"
        image.save(out_dir / fname)
    image_grid = auto_glue_image_grid(images)
    image_grid.save(out_dir / "image_grid.png")

