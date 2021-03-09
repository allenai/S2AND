import os
import json

CONFIG_LOCATION = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "data", "path_config.json"))
with open(CONFIG_LOCATION) as _json_file:
    CONFIG = json.load(_json_file)

os.environ["OMP_NUM_THREADS"] = "8"

import argparse
import logging
import pickle
from tqdm import tqdm

logger = logging.getLogger("s2and")

os.environ["S2AND_CACHE"] = os.path.join(CONFIG["internal_data_dir"], ".feature_cache")

from s2and.data import ANDData
from s2and.eval import claims_eval
from s2and.file_cache import cached_path
from s2and.consts import NAME_COUNTS_PATH

DATA_DIR = os.path.join(CONFIG["internal_data_dir"], "claims")
BLOCK_DATASETS_DIR = os.path.join(DATA_DIR, "block_datasets")


def main(model_path: str, n_jobs: int = 20, use_constraints: bool = True):
    """
    This script is for evaluating a model on the Semantic Scholar corrections data.
    It clusters each block for which we have pairwise corrections data (and the data is already
    pulled from Semantic Scholar for), and runs clustering and prints metrics out
    """
    with open(os.path.join(DATA_DIR, "claims_pairs_remapped.json")) as _json_file:
        claims_pairs = json.load(_json_file)
    logger.info("Claims pairs loaded")

    with open(model_path, "rb") as _pickle_file:
        models = pickle.load(_pickle_file)
    clusterer = models["clusterer"]

    clusterer.n_jobs = n_jobs
    clusterer.use_cache = True
    clusterer.use_default_constraints_as_supervision = use_constraints
    clusterer.batch_size = 10000000
    logger.info(f"Linkage type: {clusterer.cluster_model.linkage}")
    logger.info(f"EPS: {clusterer.cluster_model.eps}")
    logger.info(f"Use constraints: {clusterer.use_default_constraints_as_supervision}")
    logger.info(f"Featurizer version: {clusterer.featurizer_info.featurizer_version}")
    logger.info(f"Use constraints: {clusterer.use_default_constraints_as_supervision}")

    block_keys = sorted(
        filter(
            lambda x: not x.endswith(".json")
            and not x.endswith(".pickle")
            and not x.endswith(".py")
            and not x.endswith(".vscode")
            and not x.endswith(".csv"),
            os.listdir(BLOCK_DATASETS_DIR),
        ),
        key=lambda x: os.path.getsize(os.path.join(os.path.join(BLOCK_DATASETS_DIR, x), "claims_signatures.json")),
    )

    logger.info("starting transfer experiment main, loading name counts")
    with open(cached_path(NAME_COUNTS_PATH), "rb") as f:
        (
            first_dict,
            last_dict,
            first_last_dict,
            last_first_initial_dict,
        ) = pickle.load(f)
    name_counts = {
        "first_dict": first_dict,
        "last_dict": last_dict,
        "first_last_dict": first_last_dict,
        "last_first_initial_dict": last_first_initial_dict,
    }
    logger.info("loaded name counts")

    for block_key in tqdm(block_keys):
        results = {}
        block_dir = os.path.join(BLOCK_DATASETS_DIR, block_key)
        logger.info(f"Loading dataset {block_key}")
        dataset = ANDData(
            signatures=os.path.join(block_dir, "claims_signatures.json"),
            papers=os.path.join(block_dir, "claims_papers.json"),
            mode="inference",
            specter_embeddings=os.path.join(block_dir, "claims_specter.pickle"),
            block_type="s2",
            name=block_key.replace(" ", "_"),
            n_jobs=n_jobs,
            load_name_counts=name_counts,
        )
        logger.info("Dataset loaded")

        result = claims_eval(
            dataset,
            clusterer,
            claims_pairs,
            os.path.join(BLOCK_DATASETS_DIR, dataset.name),
            output_shap=False,
        )
        results[block_key.replace(" ", "_")] = result
        logger.info(f"Claims eval output: {result}")

        with open(
            os.path.join(
                BLOCK_DATASETS_DIR,
                dataset.name,
                f"results_{clusterer.featurizer_info.featurizer_version}.json",
            ),
            "w",
        ) as _json_file:
            json.dump(results, _json_file)
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model to load",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=20,
        help="How many cpus to use",
    )
    parser.add_argument(
        "--use_constraints",
        action="store_true",
        default=True,
        help="Whether to use the constraints",
    )
    args = parser.parse_args()
    main(args.model_path, args.n_jobs, args.use_constraints)
