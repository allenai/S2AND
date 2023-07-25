import os
import json
import pickle
import collections
import numpy as np
from s2and.consts import CONFIG

DATA_DIR = CONFIG["main_data_dir"]

OUTPUT_DIR = os.path.join(DATA_DIR, "s2and_mini")
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# excluding MEDLINE because it has no clusters
DATASETS = [
    "aminer",
    "arnetminer",
    "inspire",
    "kisti",
    "pubmed",
    "qian",
    "zbmath",
]

BIG_BLOCK_CUTOFF = 500
TOP_BLOCKS_TO_KEEP = 1000

# load all of the artifacts of each dataset
clusters_all = []
signatures_all = []
X_all = []
keys_all = []
papers_all = []
for dataset in DATASETS:
    print()
    print(f"Loading data from {dataset}...")

    for file_name in os.listdir(os.path.join(DATA_DIR, dataset)):
        file_name = os.path.join(DATA_DIR, dataset, file_name)
        if "specter" in file_name:
            with open(file_name, "rb") as _pickle_file:
                X, keys = pickle.load(_pickle_file)
                X_all.append(X)
                keys_all.append(keys)
        elif "cluster" in file_name:
            with open(file_name) as _json_file:
                clusters = json.load(_json_file)
                new_clusters = {}
                for cluster_id, v in clusters.items():
                    new_cluster_id = f"{dataset}_{cluster_id}"
                    new_v = {
                        "cluster_id": new_cluster_id,
                        "signature_ids": [f"{dataset}_{i}" for i in v["signature_ids"]],
                        "model_version": v["model_version"],
                    }
                    new_clusters[new_cluster_id] = new_v
                clusters_all.append(new_clusters)
        elif "paper" in file_name:
            with open(file_name) as _json_file:
                papers = json.load(_json_file)
                papers_all.append(papers)
        elif "signature" in file_name:
            with open(file_name) as _json_file:
                signatures = json.load(_json_file)
                new_signatures = {}
                for signature_id, v in signatures.items():
                    new_signature_id = f"{dataset}_{signature_id}"
                    new_v = {
                        "author_id": v["author_id"],  # maybe this needs to be prepended by dataset?
                        "paper_id": v["paper_id"],
                        "signature_id": new_signature_id,
                        "author_info": v["author_info"],
                    }
                    new_signatures[new_signature_id] = new_v
                signatures_all.append(new_signatures)
        else:
            print(f"WARNING: Ignoring {file_name} in {dataset}")

print("Finished loading data.  Filtering...")

# the goal is speed so we'll remove the largest blocks
# also only keep top 1000 blocks max
# aminer has 32k, inspire has 15k, and kisti has 7k blocks
for dataset, s, c, p, X, k in zip(DATASETS, signatures_all, clusters_all, papers_all, X_all, keys_all):
    blocks = []
    for v in s.values():
        blocks.append(v["author_info"]["block"])

    vc = collections.Counter(blocks)
    blocks_to_keep = set([k for k, v in sorted(vc.items()) if v <= BIG_BLOCK_CUTOFF][:TOP_BLOCKS_TO_KEEP])

    s_filtered = {k: v for k, v in s.items() if v["author_info"]["block"] in blocks_to_keep}

    # filter the clusters too
    c_filtered = {k: v for k, v in c.items() if np.all([i in s_filtered for i in v["signature_ids"]])}

    # go back through the clusters and find the signatures we'll actually need
    # need to do this because sometimes the block name is just... corrupted
    # e.g. "g miller" for most signatures but "g mller" for one...
    signature_keys_to_keep = set()
    for v in c_filtered.values():
        signature_keys_to_keep.update(v["signature_ids"])

    s_filtered = {k: v for k, v in s.items() if k in signature_keys_to_keep}

    # we don't need all the papers anymore. just the ones in signatures
    # also the references of those
    paper_ids = set([v["paper_id"] for v in s_filtered.values()])
    ref_paper_ids = set()
    for v in p.values():
        if v["references"] is not None:
            ref_paper_ids.update(v["references"])

    p_filtered = {k: v for k, v in p.items() if int(k) in paper_ids or int(k) in ref_paper_ids}

    # filter down the specters to those in papers only since we don't use specters for references
    keys_filtered_flag = np.array([i in paper_ids for i in k.astype(int)])  # type: ignore
    k_filtered = k[keys_filtered_flag]
    X_filtered = X[keys_filtered_flag, :]

    # save all of the data
    data_output_dir = os.path.join(DATA_DIR, "s2and_mini", dataset)
    if not os.path.exists(data_output_dir):
        os.mkdir(data_output_dir)

    with open(os.path.join(data_output_dir, f"{dataset}_clusters.json"), "w") as _json_file:
        json.dump(c_filtered, _json_file)

    with open(os.path.join(data_output_dir, f"{dataset}_signatures.json"), "w") as _json_file:
        json.dump(s_filtered, _json_file)

    with open(os.path.join(data_output_dir, f"{dataset}_papers.json"), "w") as _json_file:
        json.dump(p_filtered, _json_file)

    with open(os.path.join(data_output_dir, f"{dataset}_specter.pickle"), "wb") as _pickle_file:
        pickle.dump((X_filtered, k_filtered), _pickle_file)
