import argparse
import collections
import gzip
import itertools
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
    dataset,
    block_sigs,
    rng,
    num_queries=None,
    num_positives=None,
    num_negatives=None,
    negative_ranker_fn=None,
):
    block_sigs = block_sigs.copy()
    rng.shuffle(block_sigs)

    for query_sig_id in block_sigs[:num_queries]:
        query_sig = dataset.signatures[query_sig_id]

        cluster_sigs = set(dataset.clusters[dataset.signature_to_cluster_id[query_sig_id]]["signature_ids"])

        positives = [dataset.signatures[x] for x in cluster_sigs if x != query_sig_id]
        rng.shuffle(positives)

        negatives = [
            dataset.signatures[x]
            for x in block_sigs
            if x not in cluster_sigs
            and len(set(query_sig.author_info_coauthors).intersection(set(dataset.signatures[x].author_info_coauthors)))
            == 0
        ]
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
    (
        block_splits["train"],
        block_splits["dev"],
        block_splits["test"],
    ) = dataset.split_cluster_signatures()

    if args.n_random_neg_per_query != 0:
        raise ValueError(
            "TODO: Only hard negatives are supported for this mode.  Need script changes to do random as well."
        )

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
                    num_negatives=args.n_hard_neg_per_query,
                    negative_ranker_fn=ranker,
                )
            )
            pbar.desc = "size={}".format(sum(len(x) for x in triplets.values()))
        if len(triplets[split_name]) > args.max_triplets_per_dataset_split:
            rng.shuffle(triplets[split_name])
            triplets[split_name] = triplets[split_name][: args.max_triplets_per_dataset_split]

    return triplets


def generate_block_rankformat(
    dataset,
    block_sigs,
    rng,
    num_queries=None,
    num_positives=None,
    num_random_negatives=None,
    num_hard_negatives=None,
    negative_ranker_fn=None,
    used_pairs=None,
    blacklisted_papers=None,
):
    block_sigs = block_sigs.copy()
    rng.shuffle(block_sigs)

    if blacklisted_papers is None:
        blacklisted_papers = set()

    if num_hard_negatives != 0 and negative_ranker_fn is None:
        raise ValueError("Asked for hard negatives but no negative_ranker_fn was given")

    for query_sig_id in [x for x in block_sigs if dataset.signatures[x].paper_id not in blacklisted_papers][
        :num_queries
    ]:
        query_sig = dataset.signatures[query_sig_id]

        cluster_sigs = set(
            x
            for x in dataset.clusters[dataset.signature_to_cluster_id[query_sig_id]]["signature_ids"]
            if dataset.signatures[x].paper_id not in blacklisted_papers
        )

        positives = [
            dataset.signatures[x]
            for x in cluster_sigs
            if x != query_sig_id
            and dataset.signatures[x].paper_id not in blacklisted_papers
            and tuple(sorted([query_sig.paper_id, dataset.signatures[x].paper_id])) not in used_pairs
        ]

        negatives = [
            dataset.signatures[x]
            for x in block_sigs
            if x not in cluster_sigs
            and dataset.signatures[x].paper_id not in blacklisted_papers
            and tuple(sorted([query_sig.paper_id, dataset.signatures[x].paper_id])) not in used_pairs
            and len(set(query_sig.author_info_coauthors).intersection(set(dataset.signatures[x].author_info_coauthors)))
            == 0
        ]

        # Filter duplicates with two different authors on the same paper
        new_pos = []
        pos_pids = set()
        for x in positives:
            if x.paper_id in pos_pids:
                continue
            pos_pids.add(x.paper_id)
            new_pos.append(x)
        positives = new_pos

        new_negs = []
        neg_pids = set(pos_pids)
        for x in negatives:
            if x.paper_id in neg_pids:
                continue
            neg_pids.add(x.paper_id)
            new_negs.append(x)
        negatives = new_negs

        # Try to keep ratio for papers that don't have enough negatives
        if len(negatives) < (num_hard_negatives + num_random_negatives):
            s = num_hard_negatives + num_random_negatives
            num_hard_negatives = num_hard_negatives // s
            num_random_negatives = num_random_negatives // s

            # If rounding down reduced total negatives, give the remainder to random
            r = len(negatives) - (num_hard_negatives + num_random_negatives)
            num_random_negatives += r

        hard_negatives = []
        if num_hard_negatives != 0:
            negatives.sort(reverse=True, key=lambda neg: negative_ranker_fn(query_sig, neg))
            hard_negatives = negatives[:num_hard_negatives]
            negatives = negatives[num_hard_negatives:]

        random_negatives = []
        if num_random_negatives != 0:
            rng.shuffle(negatives)
            random_negatives = negatives[:num_random_negatives]
            negatives = negatives[num_random_negatives:]

        negatives = hard_negatives + random_negatives
        rng.shuffle(negatives)

        positives = positives[:num_positives]
        rng.shuffle(positives)

        for x in positives + negatives:
            used_pairs.add(tuple(sorted([query_sig.paper_id, x.paper_id])))

        if len(positives) >= 1 and len(negatives) >= 1:
            yield {
                "query": str(query_sig.paper_id),
                "positives": [str(x.paper_id) for x in positives],
                "negatives": [str(x.paper_id) for x in negatives],
            }


def make_dataset_rankformat(args, dataset, used_pairs=None, test_papers=None):
    rng = np.random.default_rng(args.seed)

    if used_pairs is None:
        used_pairs = set()

    ranker = NgramJaccardRanker(dataset.raw_papers)

    block_splits = dict()
    (
        block_splits["train"],
        block_splits["dev"],
        block_splits["test"],
    ) = dataset.split_cluster_signatures()

    with open(os.path.join(args.output, "block_sigs_{}.json".format(dataset.name)), "x") as f:
        json.dump(block_splits, f)

    triplets = dict()
    for split_name, split_blocks in block_splits.items():
        triplets[split_name] = []
        pbar = tqdm.tqdm(split_blocks.items(), total=len(split_blocks))
        for block_name, block_sigs in pbar:
            triplets[split_name].extend(
                generate_block_rankformat(
                    dataset,
                    block_sigs,
                    rng,
                    num_queries=args.n_queries_per_block,
                    num_positives=args.n_pos_per_query,
                    num_random_negatives=args.n_random_neg_per_query,
                    num_hard_negatives=args.n_hard_neg_per_query,
                    negative_ranker_fn=ranker,
                    used_pairs=used_pairs,
                    blacklisted_papers=(test_papers if split_name != "test" else None),
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
    parser.add_argument("--n_random_neg_per_query", type=int, default=None)
    parser.add_argument("--n_hard_neg_per_query", type=int, default=None)
    parser.add_argument("--data_dir", default=CONFIG["main_data_dir"])
    parser.add_argument("--compression", default="none", choices=["gzip", "none"])
    parser.add_argument("--exclude_dataset", default=[], action="append")
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
        file_objs = collections.OrderedDict([("test", test_f), ("dev", dev_f), ("train", train_f)])
        # file_objs = collections.OrderedDict([("train", train_f), ("dev", dev_f), ("test", test_f)])

        used_query_ids = set()
        used_id_pairs = set()
        test_papers = set()

        datasets = [d for d in DATASETS if d not in args.exclude_dataset]

        # We need to populate the entire test_papers set in advance, because
        # papers can be shared across datasets and we want to bar ALL of the
        # test papers (not just the ones we include in triplet data) in case
        # later we do normal s2and eval using embeddings from the
        # triplet-trained model
        for dataset_name in datasets:
            logger.info("loading {}".format(dataset_name))
            dataset = load_dataset(args.data_dir, dataset_name, args.seed)

            test_blocks = dataset.split_cluster_signatures()[2]
            for sig in itertools.chain(*test_blocks.values()):
                test_papers.add(dataset.signatures[sig].paper_id)

                test_papers |= {
                    dataset.signatures[x].paper_id
                    for x in dataset.clusters[dataset.signature_to_cluster_id[sig]]["signature_ids"]
                }

        logger.info("Reserved {} test papers".format(len(test_papers)))

        for dataset_name in datasets:
            logger.info("loading {}".format(dataset_name))
            dataset = load_dataset(args.data_dir, dataset_name, args.seed)

            triplets = make_dataset_rankformat(args, dataset, used_pairs=used_id_pairs, test_papers=test_papers)
            logger.info(
                "made {} examples for {}. Saving...".format(sum(len(x) for x in triplets.values()), dataset_name)
            )
            for split_name, file_obj in file_objs.items():
                for row in triplets[split_name]:
                    # Filter duplicates (would lose a bit less data doing this
                    # in generate_block_*, but should be small enough diff not
                    # to be worth the trouble)
                    query_record = paper_id_to_full(row["query"], dataset.raw_papers)
                    if query_record["corpus_id"] in used_query_ids:
                        continue
                    used_query_ids.add(query_record["corpus_id"])

                    pos_candidates = [paper_id_to_full(x, dataset.raw_papers) for x in row["positives"]]
                    for x in pos_candidates:
                        x["score"] = 1

                    neg_candidates = [paper_id_to_full(x, dataset.raw_papers) for x in row["negatives"]]
                    for x in neg_candidates:
                        x["score"] = 0

                    file_obj.write(
                        json.dumps(
                            {
                                "dataset": dataset_name,
                                "query": query_record,
                                "candidates": pos_candidates + neg_candidates,
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
