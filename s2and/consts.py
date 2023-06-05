import numpy as np
from pathlib import Path
import os
import json
import logging

logger = logging.getLogger("s2and")

try:
    PROJECT_ROOT_PATH = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
except NameError:
    PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.getcwd()))

# load up the path_configs and check if they are set
CONFIG_LOCATION = os.path.join(PROJECT_ROOT_PATH, "data", "path_config.json")
with open(CONFIG_LOCATION) as _json_file:
    CONFIG = json.load(_json_file)

if CONFIG["main_data_dir"] == "absolute path of wherever you downloaded the data to":
    logger.warning("You haven't set `main_data_dir` in data/path_config.json! Using data/ as default data directory.")
    CONFIG["main_data_dir"] = os.path.join(PROJECT_ROOT_PATH, "data")

assert os.path.exists(CONFIG["main_data_dir"]), "The `main_data_dir` specified in data/path_config.json doesn't exist."

# paths
NAME_COUNTS_PATH = os.path.join(CONFIG["main_data_dir"], "name_counts.pickle")
if not os.path.exists(NAME_COUNTS_PATH):
    NAME_COUNTS_PATH = "https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/s2and-release/name_counts.pickle"

FASTTEXT_PATH = os.path.join(CONFIG["main_data_dir"], "lid.176.bin")
if not os.path.exists(FASTTEXT_PATH):
    FASTTEXT_PATH = "https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/s2and-release/lid.176.bin"

# feature caching related consts
CACHE_ROOT = Path(os.getenv("S2AND_CACHE", str(Path.home() / ".s2and")))
"""
Incrementation history
1 - initial version
2 - changed to SPECTERv2, subblocking etc
"""
FEATURIZER_VERSION = 2

# important constant values
NUMPY_NAN = np.nan
DEFAULT_CHUNK_SIZE = 100
LARGE_DISTANCE = 1e4
LARGE_INTEGER = 10 * LARGE_DISTANCE
CLUSTER_SEEDS_LOOKUP = {"require": 0, "disallow": LARGE_DISTANCE}
SPECTER_DIM = 768
