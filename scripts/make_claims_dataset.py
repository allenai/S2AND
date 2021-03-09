from typing import Dict, Any

import os
import json

CONFIG_LOCATION = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "data", "path_config.json"))
with open(CONFIG_LOCATION) as _json_file:
    CONFIG = json.load(_json_file)

os.environ["OMP_NUM_THREADS"] = "1"

import logging
from collections import defaultdict
import pickle
import numpy as np
import pandas as pd
import random
import argparse

logger = logging.getLogger("s2and")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)
logger.addHandler(ch)

from tqdm import tqdm

os.environ["S2AND_CACHE"] = os.path.join(CONFIG["internal_data_dir"], ".feature_cache")

DATA_DIR = CONFIG["internal_data_dir"]

# NOTE: This script will not run, because these functions need to access internal resources
from scripts.redshift_funcs import (
    get_all_author_rows_for_block_key,
    get_affiliations,
    get_all_references,
    make_s2ids_table,
    get_paper_all_metadata,
    get_author_ids_for_papers,
    specterize,
)


def key_already_processed(block_key: str, base_dir: str) -> bool:
    return os.path.exists(os.path.join(base_dir, "block_datasets", block_key.replace(" ", "_")))


def main(num_blocks: int):
    """
    This scripts is used to construct the necessary dataset files for each block being evaluated
    with the Semantic Scholar corrections data
    """
    random.seed(1234)
    base_dir = os.path.join(DATA_DIR, "claims")
    with open(os.path.join(base_dir, "claims_pairs_remapped.json")) as _json_file:
        claims_pairs = json.load(_json_file)

    by_block_key = defaultdict(list)
    for signature_id_1, signature_id_2, _, block_key_1, block_key_2 in claims_pairs:
        # NOTE: we are skipping pairs that span block keys, as the model cannot currently recover
        # from blocking errors
        if block_key_1 != block_key_2:
            continue
        by_block_key[block_key_1].append(signature_id_1)
        by_block_key[block_key_2].append(signature_id_2)

    block_keys_and_counts = sorted(
        [(key, len(value)) for key, value in by_block_key.items() if not key_already_processed(key, base_dir)],
        key=lambda x: x[1],
    )

    block_keys_to_query = set()
    for block_key_tup in block_keys_and_counts[:num_blocks]:
        block_keys_to_query.add(block_key_tup[0])

    block_datasets_dir = os.path.join(base_dir, "block_datasets")
    if not os.path.exists(block_datasets_dir):
        os.mkdir(block_datasets_dir)
    logger.info(f"Starting {len(block_keys_to_query)} block keys")
    for block_key in tqdm(block_keys_to_query, desc="Iterating over block keys"):
        claims_signatures = {}
        claims_papers = {}
        X = None
        keys = []

        logger.info(f"Starting {block_key}")
        block_dataset_dir = os.path.join(block_datasets_dir, block_key.replace(" ", "_"))
        if not os.path.exists(block_dataset_dir):
            os.mkdir(block_dataset_dir)
        all_author_rows = get_all_author_rows_for_block_key(block_key)  # currently from corpusdb
        logger.info(f"Retrieved {len(all_author_rows)} author rows")

        root_paper_ids = list(set(all_author_rows["corpus_paper_id"].values))
        logger.info(f"{len(root_paper_ids)} total root paper ids")
        make_s2ids_table(root_paper_ids, table_name="temp_AND_root_ids")
        logger.info("temp table made")
        mag_affiliations = get_affiliations("temp_AND_root_ids")
        logger.info("affiliations gotten")
        all_references = get_all_references("temp_AND_root_ids")
        logger.info("references gotten")
        reference_ids = set(list(all_references["to_paper_id"].values))
        logger.info(f"Retrieved {len(reference_ids)} reference ids")

        # metadata
        paper_ids_to_query = list(set(root_paper_ids).union(reference_ids))
        make_s2ids_table(paper_ids_to_query, table_name="temp_AND_all_ids")
        logger.info("temp table made")
        all_metadata = get_paper_all_metadata("temp_AND_all_ids")
        logger.info("metadata gotten")

        # co-authors for all IDs
        all_authors = get_author_ids_for_papers("temp_AND_all_ids")
        logger.info("authors gotten")

        # specter
        specter_X, specter_keys = specterize(list(root_paper_ids))
        logger.info(f"Retrieved {specter_X.shape} specterized papers")

        list_of_dicts = all_authors.to_dict("records")
        corpus_id_to_authors = defaultdict(list)
        for row in list_of_dicts:
            corpus_id_to_authors[row["corpus_paper_id"]].append(row)
        logger.info("Authors dict made")

        list_of_dicts = all_references.to_dict("records")
        corpus_id_to_references = defaultdict(list)
        for row in list_of_dicts:
            corpus_id_to_references[row["from_paper_id"]].append(int(row["to_paper_id"]))
        logger.info("References dict made")

        list_of_dicts = mag_affiliations.to_dict("records")
        corpus_id_and_position_to_mag = defaultdict(list)
        for row in list_of_dicts:
            key = str(row["corpus_paper_id"]) + "___" + str(row["author_position"])
            corpus_id_and_position_to_mag[key].append(row["displayname"])
        logger.info("Affiliations dict made")

        for _, paper_row in tqdm(all_metadata.iterrows(), desc="making papers dict"):
            output_row: Dict[str, Any] = {}
            output_row["paper_id"] = int(paper_row["corpus_paper_id"])
            output_row["title"] = paper_row["title"]
            output_row["abstract"] = paper_row["abstract"]
            output_row["journal_name"] = paper_row["journal_name"]
            output_row["venue"] = paper_row["venue"]
            output_row["year"] = int(paper_row["year"]) if not pd.isnull(paper_row["year"]) else None
            author_rows = corpus_id_to_authors[paper_row["corpus_paper_id"]]
            output_row["authors"] = [
                {
                    "position": int(author_row["position"]),
                    "author_name": " ".join(
                        [
                            part
                            for part in [
                                author_row["first"],
                                author_row["middle"],
                                author_row["last"],
                                author_row["suffix"],
                            ]
                            if part is not None
                        ]
                    ),
                }
                for author_row in author_rows
            ]
            output_row["references"] = corpus_id_to_references[paper_row["corpus_paper_id"]]
            claims_papers[str(paper_row["corpus_paper_id"])] = output_row

        for _, author_row in tqdm(all_author_rows.iterrows(), desc="making signatures dict"):
            output_signature: Dict[str, Any] = {}
            output_signature["signature_id"] = str(author_row["corpus_paper_id"]) + "___" + str(author_row["position"])
            output_signature["paper_id"] = int(author_row["corpus_paper_id"])
            output_author_info: Dict[str, Any] = {}
            output_author_info["position"] = int(author_row["position"])
            output_author_info["block"] = author_row["cluster_block_key"]
            output_author_info["first"] = author_row["first"]
            output_author_info["middle"] = author_row["middle"]
            output_author_info["last"] = author_row["last"]
            output_author_info["suffix"] = author_row["suffix"]
            output_author_info["email"] = eval(author_row["emails"])[0] if author_row["emails"] is not None else None
            affiliations = eval(author_row["affiliations"]) if author_row["affiliations"] is not None else []
            affiliations.extend(
                corpus_id_and_position_to_mag[str(author_row["corpus_paper_id"]) + "___" + str(author_row["position"])]
            )
            output_author_info["affiliations"] = affiliations
            output_signature["author_info"] = output_author_info
            claims_signatures[output_signature["signature_id"]] = output_signature

        logger.info(f"writing signatures {len(claims_signatures)}")
        with open(os.path.join(block_dataset_dir, "claims_signatures.json"), "w") as _json_file:
            json.dump(claims_signatures, _json_file)

        logger.info(f"writing papers {len(claims_papers)}")
        with open(os.path.join(block_dataset_dir, "claims_papers.json"), "w") as _json_file:
            json.dump(claims_papers, _json_file)

        if X is not None:
            X = np.vstack([X, specter_X])
        else:
            X = specter_X
        keys.extend(specter_keys)
        logger.info(f"writing specter {X.shape} {len(keys)}")
        with open(os.path.join(block_dataset_dir, "claims_specter.pickle"), "wb") as _pickle_file:
            pickle.dump((X, keys), _pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Done with {block_key}")
    logger.info("done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_blocks",
        type=int,
        help="Number of blocks to get data for",
    )
    args = parser.parse_args()
    main(args.num_blocks)
