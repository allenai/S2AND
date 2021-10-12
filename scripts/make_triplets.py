import argparse
import collections
import gzip
import json
import logging
import os

import tqdm
import numpy as np

import s2and
from s2and.data import ANDData
from s2and.consts import CONFIG
from s2and.text import counter_jaccard, cosine_sim, STOPWORDS

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


class NgramJaccardRanker:
    def __init__(self, papers):
        self.paper_ngrams = {k: self._unigrams(v["title"]) + self._unigrams(v["abstract"]) for k, v in papers.items()}

    def _unigrams(self, text):
        if text is None or len(text) == 0:
            return collections.Counter()
        text_split = [word for word in text.lower().split() if word not in STOPWORDS and len(word) > 1]
        unigrams = collections.Counter(text_split)
        return unigrams

    def __call__(self, sig1, sig2):
        paper1 = self.paper_ngrams[str(sig1.paper_id)]
        paper2 = self.paper_ngrams[str(sig2.paper_id)]

        return counter_jaccard(paper1, paper2)


class SpecterRanker:
    def __init__(self, specters):
        self.paper2idx = {paper_id: idx for idx, paper_id in enumerate(specters[1])}
        self.specter = specters[0]

    def __call__(self, sig1, sig2):
        emb1 = self.specter[self.paper2idx[str(sig1.paper_id)]]
        emb2 = self.specter[self.paper2idx[str(sig2.paper_id)]]

        return cosine_sim(emb1, emb2)


def generate_block_triplets(
    dataset, block_sigs, rng, num_queries=None, num_positives=None, num_negatives=None, negative_ranker_fn=None
):
    block_sigs = block_sigs.copy()
    rng.shuffle(block_sigs)

    for query_sig_id in block_sigs[:num_queries]:
        query_sig = dataset.signatures[query_sig_id]

        cluster_sigs = set(dataset.clusters[dataset.signature_to_cluster_id[query_sig_id]]["signature_ids"])

        positives = [dataset.signatures[x] for x in cluster_sigs if x != query_sig_id]
        rng.shuffle(positives)

        negatives = [dataset.signatures[x] for x in block_sigs if x not in cluster_sigs]
        if negative_ranker_fn:
            negatives.sort(reverse=True, key=lambda neg: negative_ranker_fn(query_sig, neg))
        else:
            rng.shuffle(negatives)

        for pos in positives[:num_positives]:
            for neg in negatives[:num_negatives]:
                yield (str(query_sig.paper_id), str(pos.paper_id), str(neg.paper_id))


def make_dataset_triplets(args, dataset):
    rng = np.random.default_rng(args.seed)

    ranker = NgramJaccardRanker(dataset.raw_papers)

    block_splits = dict()
    block_splits["train"], block_splits["dev"], block_splits["test"] = dataset.split_cluster_signatures()

    triplets = dict()
    for split_name, split_blocks in block_splits.items():
        triplets[split_name] = []
        pbar = tqdm.tqdm(split_blocks.items(), total=len(split_blocks))
        for block_name, block_sigs in pbar:
            triplets[split_name].extend(
                generate_block_triplets(
                    dataset,
                    block_sigs,
                    rng,
                    num_queries=args.n_queries_per_block,
                    num_positives=args.n_pos_per_query,
                    num_negatives=args.n_neg_per_query,
                    negative_ranker_fn=ranker,
                )
            )
            pbar.desc = "size={}".format(sum(len(x) for x in triplets.values()))
        if len(triplets[split_name]) > args.max_triplets_per_dataset_split:
            rng.shuffle(triplets[split_name])
            triplets[split_name] = triplets[split_name][: args.max_triplets_per_dataset_split]

    return triplets


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
    parser.add_argument("--max_triplets_per_dataset_split", type=int, default=100000)
    parser.add_argument("--n_queries_per_block", type=int, default=None)
    parser.add_argument("--n_pos_per_query", type=int, default=None)
    parser.add_argument("--n_neg_per_query", type=int, default=None)
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
    with open_fn(os.path.join(args.output, "train.jsonl" + extension), "wt") as train_f, open_fn(
        os.path.join(args.output, "dev.jsonl" + extension), "wt"
    ) as dev_f, open_fn(os.path.join(args.output, "test.jsonl" + extension), "wt") as test_f:
        file_objs = {"train": train_f, "dev": dev_f, "test": test_f}

        for dataset_name in DATASETS:
            logger.info("loading {}".format(dataset_name))
            dataset = load_dataset(args.data_dir, dataset_name, args.seed)

            triplets = make_dataset_triplets(args, dataset)
            logger.info(
                "made {} triplets for {}. Saving...".format(sum(len(x) for x in triplets.values()), dataset_name)
            )
            for split_name, file_obj in file_objs.items():
                for row in triplets[split_name]:
                    file_obj.write(
                        json.dumps(
                            {
                                "dataset": dataset_name,
                                "seed": paper_id_to_full(row[0], dataset.raw_papers),
                                "positive": paper_id_to_full(row[1], dataset.raw_papers),
                                "negative": paper_id_to_full(row[2], dataset.raw_papers),
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
