# Default settings
import logging
from pathlib import Path


CUDA_VISIBLE_DEVICE = ""

# The dict used to configure logging
LOG_DICT_CONFIG = dict(
    version=1,
    formatters={"f": {"format": "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"}},
    handlers={
        "h": {
            "class": "logging.StreamHandler",
            "formatter": "f",
            "level": logging.DEBUG,
        }
    },
    root={
        "handlers": ["h"],
        "level": logging.DEBUG,
    },
)

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
