# Default settings
import logging
from pathlib import Path

THIS_FILE = Path(__file__)
ROOT = THIS_FILE.parent.parent
OUTPUT_DIR = ROOT / "output"

# Model input
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5

GENERATOR_SEED = 42

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

NUM_IMAGES = 4
N_COLS = 2
N_ROWS = 2
