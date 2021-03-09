import numpy as np
from pathlib import Path
import os

try:
    PROJECT_ROOT_PATH = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
except NameError:
    PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.getcwd()))

# paths
NAME_COUNTS_PATH = os.path.join(PROJECT_ROOT_PATH, "data", "name_counts.pickle")
FASTTEXT_PATH = os.path.join(PROJECT_ROOT_PATH, "data", "lid.176.bin")
if not os.path.exists(FASTTEXT_PATH):
    FASTTEXT_PATH = "https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/s2and/lid.176.bin"

# feature caching related consts
CACHE_ROOT = Path(os.getenv("S2AND_CACHE", str(Path.home() / ".s2and")))
FEATURIZER_VERSION = 1


# values
NUMPY_NAN = np.nan
DEFAULT_CHUNK_SIZE = 100
LARGE_DISTANCE = 1e4
LARGE_INTEGER = 10 * LARGE_DISTANCE
CLUSTER_SEEDS_LOOKUP = {"require": 0, "disallow": LARGE_DISTANCE}
