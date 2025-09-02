import argparse
import gzip
import itertools
import json
import logging
import os

import tqdm

import s2and
from s2and.data import ANDData
from s2and.consts import CONFIG

logger = logging.getLogger(__name__)

DATASETS = [
    "aminer",
    "arnetminer",
    "inspire",
    "kisti",
    "pubmed",
    "qian",
    "zbmath",
]


def make_dataset_pairwise_classification_style(args, dataset):
    block_splits = dict()
    (
        block_splits["train"],
        block_splits["dev"],
        block_splits["test"],
    ) = dataset.split_cluster_signatures()

    examples = dict()
    for split_name, split_blocks in block_splits.items():
        examples[split_name] = []
        pbar = tqdm.tqdm(split_blocks.items(), total=len(split_blocks))
        for block_name, block_sigs in pbar:
            for sig_id1, sig_id2 in itertools.combinations(block_sigs, 2):
                sig1 = dataset.signatures[sig_id1]
                sig2 = dataset.signatures[sig_id2]

                label = (
                    1 if (dataset.signature_to_cluster_id[sig_id1] == dataset.signature_to_cluster_id[sig_id2]) else 0
                )
                if label == 0:
                    # If coauthors overlap, the papers might actually share an
                    # author, so we can't set label=0.  We just throw these
                    # examples away.
                    if len(set(sig1.author_info_coauthors).intersection(set(sig2.author_info_coauthors))) != 0:
                        continue

                examples[split_name].append(
                    {
                        "corpus_id_1": str(sig1.paper_id),
                        "corpus_id_2": str(sig2.paper_id),
                        "block_name": block_name,
                        "label": label,
                    }
                )

    return examples


def load_dataset(data_dir, dataset_name, seed, n_jobs=8):
    dataset_dir = os.path.join(data_dir, dataset_name)
    dataset = ANDData(
        signatures=os.path.join(dataset_dir, dataset_name + "_signatures.json"),
        papers=os.path.join(dataset_dir, dataset_name + "_papers.json"),
        name=dataset_name,
        mode="train",
        specter_embeddings=os.path.join(dataset_dir, dataset_name + "_specter.pickle"),
        clusters=os.path.join(dataset_dir, dataset_name + "_clusters.json"),
        block_type="s2",
        n_jobs=n_jobs,
        load_name_counts=False,
        preprocess=False,
        random_seed=seed,
    )

    # Need raw papers for output and ngram ranker
    with open(os.path.join(data_dir, dataset_name, dataset_name + "_papers.json")) as f:
        dataset.raw_papers = json.load(f)

    return dataset


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--data_dir", default=CONFIG["main_data_dir"])
    parser.add_argument("--compression", default="none", choices=["gzip", "none"])
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def paper_id_to_full(paper_id, raw_papers):
    raw_paper = raw_papers[paper_id]
    return {
        "corpus_id": raw_paper["paper_id"],
        "title": raw_paper["title"],
        "abstract": raw_paper["abstract"],
    }


def main():
    args = parse_cli_args()

    open_fn = None
    extension = None
    if args.compression == "gzip":
        open_fn = gzip.open
        extension = ".gz"
    elif args.compression == "none":
        open_fn = open
        extension = ""
    else:
        raise ValueError("Invalid compression {}".format(args.compression))

    os.makedirs(args.output, exist_ok=True)
    with (
        open_fn(os.path.join(args.output, "train.jsonl" + extension), "wt") as train_f,
        open_fn(os.path.join(args.output, "dev.jsonl" + extension), "wt") as dev_f,
        open_fn(os.path.join(args.output, "test.jsonl" + extension), "wt") as test_f,
    ):
        file_objs = {"train": train_f, "dev": dev_f, "test": test_f}

        for dataset_name in DATASETS:
            logger.info("loading {}".format(dataset_name))
            dataset = load_dataset(args.data_dir, dataset_name, args.seed)

            examples = make_dataset_pairwise_classification_style(args, dataset)
            logger.info(
                "made {} examples for {}. Saving...".format(sum(len(x) for x in examples.values()), dataset_name)
            )
            for split_name, file_obj in file_objs.items():
                for row in examples[split_name]:
                    file_obj.write(
                        json.dumps(
                            {
                                "dataset": dataset_name,
                                "paper1": paper_id_to_full(row["corpus_id_1"], dataset.raw_papers),
                                "paper2": paper_id_to_full(row["corpus_id_2"], dataset.raw_papers),
                                "block_id": row["block_name"],
                                "label": row["label"],
                            }
                        )
                        + "\n"
                    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] (%(name)s):  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
        ],
    )
    s2and.logger.handlers = []

    main()
