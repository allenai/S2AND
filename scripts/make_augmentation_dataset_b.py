from typing import Dict, Any

import os
import json

CONFIG_LOCATION = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "data", "path_config.json"))
with open(CONFIG_LOCATION) as _json_file:
    CONFIG = json.load(_json_file)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "<fill me in>"
os.environ["S2AND_CACHE"] = os.path.join(CONFIG["internal_data_dir"], ".feature_cache")

import six
import numpy as np
import pandas as pd
import argparse
import logging
import pickle
import copy
import random
from collections import defaultdict

logger = logging.getLogger("s2and")

from tqdm import tqdm
from s2and.data import ANDData


from google.cloud import translate_v2

translate_client = translate_v2.Client()


def translate(text):
    coin_flip = random.uniform(0, 1)
    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")
    if coin_flip < 0.25:
        return translate_client.translate(text, target_language="fr")["translatedText"]
    elif coin_flip < 0.5:
        return translate_client.translate(text, target_language="de")["translatedText"]
    elif coin_flip < 0.75:
        return translate_client.translate(text, target_language="hi")["translatedText"]
    else:
        return translate_client.translate(text, target_language="zh")["translatedText"]


DATA_DIR = CONFIG["internal_data_dir"]

SOURCE_DATASET_NAMES = [
    "aminer",
    "arnetminer",
    "inspire",
    "kisti",
    "medline",
    "orcid",
    "pubmed",
    "qian",
    "zbmath",
]

AUGMENTATION_DIR = os.path.join(DATA_DIR, "augmented")


def main(
    max_train_positives_per_dataset: int,
    max_val_positives_per_dataset: int,
    max_test_positives_per_dataset: int,
    negatives_multiplier: float,
    drop_abstract_prob: float,
    drop_affiliations_prob: float,
    drop_references_prob: float,
    drop_first_name_prob: float,
    drop_venue_journal_prob: float,
    drop_coauthors_prob: float,
    translate_title_prob: float,
):
    """
    This script creates the extra "augmentation" dataset from the existing datasets, by randomly removing features,
    to simulate real usage better
    """
    random.seed(1111)
    augmentation_pairs = pd.read_csv(os.path.join(AUGMENTATION_DIR, "source_tuples.csv")).to_dict("records")
    with open(os.path.join(AUGMENTATION_DIR, "title_only_specters.pickle"), "rb") as _pickle_file:
        title_only_specter = pickle.load(_pickle_file)

    datasets: Dict[str, Any] = {}
    for dataset_name in tqdm(SOURCE_DATASET_NAMES, desc="Processing datasets and fitting base models"):
        logger.info("")
        logger.info(f"processing dataset {dataset_name}")
        logger.info(f"loading dataset {dataset_name}")
        anddata = ANDData(
            signatures=os.path.join(DATA_DIR, dataset_name, dataset_name + "_signatures.json"),
            papers=os.path.join(DATA_DIR, dataset_name, dataset_name + "_papers.json"),
            name=dataset_name,
            mode="inference",
            specter_embeddings=os.path.join(DATA_DIR, dataset_name, dataset_name + "_specter.pickle"),
            block_type="s2",
            n_jobs=25,
            load_name_counts=False,
            preprocess=False,
        )
        logger.info(f"dataset {dataset_name} loaded")
        datasets[dataset_name] = anddata

    full_papers = {}
    full_signatures = {}
    full_specter_keys = []
    full_specter_D = []
    train_pairs = []
    val_pairs = []
    test_pairs = []
    pair_counts: Dict[str, Dict[str, Dict[int, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for row in augmentation_pairs:
        split = row["split"]
        dataset_name = row["dataset_name"]
        signature_id_1 = row["signature_id_1"]
        signature_id_2 = row["signature_id_2"]
        label = row["label"]

        count_value = pair_counts[dataset_name][split][label]
        max_value = (
            max_train_positives_per_dataset
            if split == "train"
            else max_val_positives_per_dataset
            if split == "val"
            else max_test_positives_per_dataset
        ) * (negatives_multiplier if label == 0 else 1.0)
        if count_value >= max_value or dataset_name not in SOURCE_DATASET_NAMES:
            continue

        pair_counts[dataset_name][split][label] += 1

        pair = (dataset_name + "___" + str(signature_id_1), dataset_name + "___" + str(signature_id_2), label)

        if split == "train":
            train_pairs.append(pair)
        elif split == "val":
            val_pairs.append(pair)
        elif split == "test":
            test_pairs.append(pair)

    logger.info(f"Total pairs (train, val, test): {len(train_pairs)}, {len(val_pairs)}, {len(test_pairs)}")
    pair_counts_dict: Dict[str, Dict[str, Dict[int, int]]] = {}
    for dataset, d1 in pair_counts.items():
        pair_counts_dict[dataset] = {}
        for split, d2 in d1.items():
            pair_counts_dict[dataset][split] = {}
            for label, count in d2.items():
                pair_counts_dict[dataset][split][label] = count

    logger.info(pair_counts_dict)

    all_signatures = set(
        [item for sublist in train_pairs for item in sublist[:2]]
        + [item for sublist in val_pairs for item in sublist[:2]]
        + [item for sublist in test_pairs for item in sublist[:2]]
    )
    reference_papers_to_add = set()
    for signature in all_signatures:
        original_dataset, original_signature_id = signature.split("___")
        original_signature = datasets[original_dataset].signatures[original_signature_id]
        original_paper = datasets[original_dataset].papers[str(original_signature.paper_id)]
        original_references = [(original_dataset, paper_id) for paper_id in original_paper.references]

        new_signature_id = signature
        new_references = [copy.deepcopy(reference) for reference in original_references]

        coin_flip = random.uniform(0, 1)
        if coin_flip < drop_abstract_prob:
            new_has_abstract = False
            full_specter_keys.append(str(original_signature.paper_id))
            full_specter_D.append(title_only_specter[original_dataset + "_" + str(original_signature.paper_id)])
        else:
            new_has_abstract = original_paper.has_abstract
            full_specter_keys.append(str(original_signature.paper_id))
            full_specter_D.append(datasets[original_dataset].specter_embeddings[str(original_signature.paper_id)])

        coin_flip = random.uniform(0, 1)
        if coin_flip < drop_references_prob:
            new_references = []
        else:
            reference_papers_to_add.update(new_references)
            new_references = [reference[1] for reference in new_references]

        coin_flip = random.uniform(0, 1)
        if coin_flip < drop_affiliations_prob:
            new_affiliations = []
        else:
            new_affiliations = original_signature.author_info_affiliations

        coin_flip = random.uniform(0, 1)
        if coin_flip < drop_venue_journal_prob:
            new_venue = None
            new_journal_name = None
        else:
            new_venue = original_paper.venue
            new_journal_name = original_paper.journal_name

        coin_flip = random.uniform(0, 1)
        if coin_flip < drop_first_name_prob:
            new_first = (
                original_signature.author_info_first[0]
                if original_signature.author_info_first is not None and len(original_signature.author_info_first) > 0
                else original_signature.author_info_first
            )
        else:
            new_first = original_signature.author_info_first

        coin_flip = random.uniform(0, 1)
        if coin_flip < drop_coauthors_prob:
            new_paper_authors = [
                author
                for author in original_paper.authors
                if author.position == original_signature.author_info_position
            ]
        else:
            new_paper_authors = original_paper.authors

        coin_flip = random.uniform(0, 1)
        if coin_flip < translate_title_prob:
            new_title = translate(original_paper.title)
        else:
            new_title = original_paper.title

        new_signature = original_signature._replace(
            author_info_first=new_first,
            author_info_affiliations=new_affiliations,
            signature_id=new_signature_id,
            author_info_first_normalized=None,
            author_info_first_normalized_without_apostrophe=None,
            author_info_middle_normalized=None,
            author_info_middle_normalized_without_apostrophe=None,
            author_info_last_normalized=None,
            author_info_suffix_normalized=None,
            author_info_coauthors=None,
            author_info_coauthor_blocks=None,
        )
        new_paper = original_paper._replace(
            venue=new_venue,
            journal_name=new_journal_name,
            references=new_references,
            title=new_title,
            has_abstract=new_has_abstract,
            authors=new_paper_authors,
        )

        new_signature_dict = dict(new_signature._asdict())
        new_signature_dict["author_info"] = {}
        keys_to_delete = []
        for key, value in new_signature_dict.items():
            if key.startswith("author_info_"):
                keys_to_delete.append(key)
                new_signature_dict["author_info"][key[12:]] = value
        for key in keys_to_delete:
            del new_signature_dict[key]

        full_signatures[signature] = new_signature_dict
        full_papers[str(new_paper.paper_id)] = dict(new_paper._asdict())
        full_papers[str(new_paper.paper_id)]["authors"] = [
            dict(author._asdict()) for author in full_papers[str(new_paper.paper_id)]["authors"]
        ]
        # we currently don't need the actual abstract, but just need to know if it exists or not
        if full_papers[str(new_paper.paper_id)]["has_abstract"]:
            full_papers[str(new_paper.paper_id)]["abstract"] = "EXISTS"
        else:
            full_papers[str(new_paper.paper_id)]["abstract"] = ""

    logger.info(f"Adding {len(reference_papers_to_add)} reference papers")
    reference_papers_added = 0
    for dataset_name, paper_id in reference_papers_to_add:
        if str(paper_id) not in full_papers and str(paper_id) in datasets[dataset_name].papers:
            full_papers[str(paper_id)] = dict(datasets[dataset_name].papers[str(paper_id)]._asdict())
            full_papers[str(paper_id)]["authors"] = [
                dict(author._asdict()) for author in full_papers[str(paper_id)]["authors"]
            ]
            if full_papers[str(paper_id)]["has_abstract"]:
                full_papers[str(paper_id)]["abstract"] = "EXISTS"
            else:
                full_papers[str(paper_id)]["abstract"] = ""
            reference_papers_added += 1
    logger.info(f"Added {reference_papers_added} reference papers")

    logger.info(f"Dumping {len(full_papers)} papers")
    with open(os.path.join(AUGMENTATION_DIR, "augmented_papers.json"), "w") as _json_file:
        json.dump(full_papers, _json_file)

    logger.info(f"Dumping {len(full_signatures)} signatures")
    with open(os.path.join(AUGMENTATION_DIR, "augmented_signatures.json"), "w") as _json_file:
        json.dump(full_signatures, _json_file)

    full_specter_D_np = np.array(full_specter_D)
    logger.info(f"Dumping {full_specter_D_np.shape, len(full_specter_keys)} specter")
    with open(os.path.join(AUGMENTATION_DIR, "augmented_specter.pickle"), "wb") as _pickle_file:
        pickle.dump((full_specter_D_np, full_specter_keys), _pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    train_pairs_df = pd.DataFrame(train_pairs, columns=["pair1", "pair2", "label"])
    train_pairs_df["label"] = train_pairs_df["label"].apply(lambda x: "YES" if x == 1 else "NO")
    val_pairs_df = pd.DataFrame(val_pairs, columns=["pair1", "pair2", "label"])
    val_pairs_df["label"] = val_pairs_df["label"].apply(lambda x: "YES" if x == 1 else "NO")
    test_pairs_df = pd.DataFrame(test_pairs, columns=["pairs1", "pair2", "label"])
    test_pairs_df["label"] = test_pairs_df["label"].apply(lambda x: "YES" if x == 1 else "NO")

    logger.info("Writing pairs csvs")
    train_pairs_df.to_csv(os.path.join(AUGMENTATION_DIR, "train_pairs.csv"), index=False, header=True)
    val_pairs_df.to_csv(os.path.join(AUGMENTATION_DIR, "val_pairs.csv"), index=False, header=True)
    test_pairs_df.to_csv(os.path.join(AUGMENTATION_DIR, "test_pairs.csv"), index=False, header=True)
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_train_positives_per_dataset",
        type=int,
        default=10000,
        help="The max number of pairs to take from one dataset",
    )
    parser.add_argument(
        "--max_val_positives_per_dataset",
        type=int,
        default=1000,
        help="The max number of pairs to take from one dataset",
    )
    parser.add_argument(
        "--max_test_positives_per_dataset",
        type=int,
        default=1000,
        help="The max number of pairs to take from one dataset",
    )
    parser.add_argument(
        "--negatives_multiplier",
        type=float,
        default=1.5,
        help="Max num negatives = max num positives * negatives multiplier",
    )
    parser.add_argument(
        "--drop_abstract_prob",
        type=float,
        default=0.5,
        help="The probability of dropping the abstract (and using the title only specter embedding)",
    )
    parser.add_argument(
        "--drop_affiliations_prob", type=float, default=0.5, help="The probability of dropping affiliations"
    )
    parser.add_argument(
        "--drop_references_prob", type=float, default=0.5, help="The probability of dropping the references"
    )
    parser.add_argument(
        "--drop_first_name_prob", type=float, default=0.5, help="The probability of dropping the first name"
    )
    parser.add_argument(
        "--drop_venue_journal_prob", type=float, default=0.5, help="The probability of dropping venue and journal info"
    )
    parser.add_argument("--drop_coauthors_prob", type=float, default=0.5, help="The probability of dropping coauthors")
    parser.add_argument(
        "--translate_title_prob", type=float, default=0.05, help="The probability of dropping coauthors"
    )
    args = parser.parse_args()
    main(
        args.max_train_positives_per_dataset,
        args.max_val_positives_per_dataset,
        args.max_test_positives_per_dataset,
        args.negatives_multiplier,
        args.drop_abstract_prob,
        args.drop_affiliations_prob,
        args.drop_references_prob,
        args.drop_first_name_prob,
        args.drop_venue_journal_prob,
        args.drop_coauthors_prob,
        args.translate_title_prob,
    )
