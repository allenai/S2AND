import os
import json

CONFIG_LOCATION = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "data", "path_config.json"))
with open(CONFIG_LOCATION) as _json_file:
    CONFIG = json.load(_json_file)

import pickle
import urllib.request
import urllib.parse
import json as json_module
import pandas as pd
from s2and.data import ANDData
from typing import Dict, List

"""
This script creates pairs and embeddings to be used in make_augmentation_dataset_b.py
"""

SPECTER_URL = "https://model-apis.semanticscholar.org/specter/v1/invoke"


def chunks(lst, chunk_size=16):
    """Splits a longer list to respect batch size"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def embed(papers, embeddings_by_paper_id: Dict[str, List[float]] = {}):
    papers_to_embed = [p for p in papers if p["paper_id"] not in embeddings_by_paper_id]
    unembedded_papers = []
    for chunk in chunks(papers_to_embed):
        # Prepare the request
        data = json_module.dumps(chunk).encode("utf-8")
        req = urllib.request.Request(SPECTER_URL, data=data, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    response_data = json_module.loads(response.read().decode("utf-8"))
                    for paper in response_data["preds"]:
                        embeddings_by_paper_id[paper["paper_id"]] = paper["embedding"]
                else:
                    unembedded_papers.extend(chunk)
        except urllib.error.URLError:
            unembedded_papers.extend(chunk)

    return embeddings_by_paper_id, unembedded_papers


DATA_DIR = CONFIG["internal_data_dir"]
AUGMENTATION_DIR = os.path.join(DATA_DIR, "augmented")

DATASETS_TO_TRAIN = [
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

N_TRAIN_PAIRS_SIZE = 100000
N_VAL_TEST_SIZE = 10000
N_JOBS = 25

datasets = {}
for dataset_name in DATASETS_TO_TRAIN:
    clusters_path = None
    if dataset_name != "medline":
        clusters_path = os.path.join(DATA_DIR, dataset_name, dataset_name + "_clusters.json")
        train_pairs_path = None
        val_pairs_path = None
        test_pairs_path = None
    else:
        train_pairs_path = os.path.join(DATA_DIR, dataset_name, "train_pairs.csv")
        val_pairs_path = os.path.join(DATA_DIR, dataset_name, "val_pairs.csv")
        if not os.path.exists(val_pairs_path):
            val_pairs_path = None
        test_pairs_path = os.path.join(DATA_DIR, dataset_name, "test_pairs.csv")

    anddata = ANDData(
        signatures=os.path.join(DATA_DIR, dataset_name, dataset_name + "_signatures.json"),
        papers=os.path.join(DATA_DIR, dataset_name, dataset_name + "_papers.json"),
        name=dataset_name,
        mode="train",
        specter_embeddings=os.path.join(DATA_DIR, dataset_name, dataset_name + "_specter.pickle"),
        clusters=clusters_path,
        block_type="s2",
        train_pairs=train_pairs_path,
        val_pairs=val_pairs_path,
        test_pairs=test_pairs_path,
        train_pairs_size=N_TRAIN_PAIRS_SIZE,
        val_pairs_size=N_VAL_TEST_SIZE,
        test_pairs_size=N_VAL_TEST_SIZE,
        n_jobs=N_JOBS,
        load_name_counts=False,
        preprocess=False,
    )

    datasets[dataset_name] = anddata

tuples = []
all_titles_dict = {}
for dataset_name, anddata in datasets.items():
    # this random seed matches the current default
    # and preserves the train/val/test split
    anddata.random_seed = 1111
    (
        train_signatures,
        val_signatures,
        test_signatures,
    ) = anddata.split_cluster_signatures()
    # a different random seed to get a random subset of pairs from train/val/test
    # this tries to get more diversity in the training data instead of just repeating
    # the same exact pairs
    anddata.random_seed = 12455678
    if dataset_name == "medline":
        train_pairs, val_pairs, test_pairs = anddata.fixed_pairs()
    else:
        train_pairs, val_pairs, test_pairs = anddata.split_pairs(train_signatures, val_signatures, test_signatures)
    train_pairs = train_pairs[:25000]
    train_sigs = set()
    val_sigs = set()
    test_sigs = set()
    for i in train_pairs:
        train_sigs.update([i[0], i[1]])
    for i in val_pairs:
        val_sigs.update([i[0], i[1]])
    for i in test_pairs:
        test_sigs.update([i[0], i[1]])
    with open(os.path.join(DATA_DIR, dataset_name, dataset_name + "_papers.json")) as f:
        papers_dict = json.load(f)
    titles_dict = {
        dataset_name
        + "_"
        + str(anddata.signatures[i].paper_id): papers_dict[str(anddata.signatures[i].paper_id)]["title"]
        for i in train_sigs.union(val_sigs).union(test_sigs)
    }
    all_titles_dict.update(titles_dict)
    for tup in train_pairs:
        tuples.append((dataset_name, "train", tup[0], tup[1], tup[2]))
    for tup in val_pairs:
        tuples.append((dataset_name, "val", tup[0], tup[1], tup[2]))
    for tup in test_pairs:
        tuples.append((dataset_name, "test", tup[0], tup[1], tup[2]))

pd.DataFrame(tuples, columns=["dataset_name", "split", "signature_id_1", "signature_id_2", "label"]).to_csv(
    os.path.join(AUGMENTATION_DIR, "source_tuples.csv"), index=False
)

papers = []
for dataset_paper_id, title in all_titles_dict.items():
    papers.append({"paper_id": dataset_paper_id, "title": title, "abstract": ""})

if os.path.exists(os.path.join(AUGMENTATION_DIR, "title_only_specters.pickle")):
    with open(os.path.join(AUGMENTATION_DIR, "title_only_specters.pickle"), "rb") as f:  # type: ignore
        embeddings = pickle.load(f)  # type: ignore

    embeddings, _ = embed(papers, embeddings)
else:
    embeddings, _ = embed(papers)

with open(os.path.join(AUGMENTATION_DIR, "title_only_specters.pickle"), "wb") as f:  # type: ignore
    pickle.dump(embeddings, f)  # type: ignore
