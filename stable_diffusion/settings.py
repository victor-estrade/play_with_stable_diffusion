import logging
import os
from pathlib import Path

# Configure the logging
from logging.config import dictConfig

from .utils import fetch_huggingface_key


# Choose which GPU can be used
CUDA_VISIBLE_DEVICE = "0"

# The location of models downloaded by torch.hub
# default is ~/.cache/torch/
TORCH_MODEL_ZOO_DIR = "/common_projects/models/torch_zoo/"


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



# ===============================================================
# Setup variables and logging and stuffs
# ===============================================================

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

if "HUGGING_FACE_TOKEN" in os.environ:
    HUGGING_FACE_TOKEN = os.environ["HUGGING_FACE_TOKEN"]
else:
    HUGGING_FACE_TOKEN = fetch_huggingface_key()

if "CUDA_DEVICE_ORDER" not in os.environ:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICE

torch_model_zoo_dir = Path(TORCH_MODEL_ZOO_DIR)
if torch_model_zoo_dir.is_dir():
    os.environ["TORCH_HOME"] = TORCH_MODEL_ZOO_DIR

dictConfig(LOG_DICT_CONFIG)
