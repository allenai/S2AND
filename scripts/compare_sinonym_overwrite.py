"""
Run the ANDData pipeline twice (with and without Sinonym overwrites) and:
- Print all signatures whose normalized first/middle (without apostrophes) changed
- Compare pairwise feature vectors on the test split using the production featurizer
"""

import os
from typing import Dict, Tuple, List

from s2and.consts import PROJECT_ROOT_PATH, DEFAULT_CHUNK_SIZE
from s2and.data import ANDData, Signature, sinonym_preprocess_papers_parallel
from s2and.featurizer import many_pairs_featurize
from s2and.eval import cluster_eval
import numpy as np
import pickle

N_JOBS = 4


def collect_normalized_first_middle(signatures: Dict[str, Signature]) -> Dict[str, Tuple[str, str]]:
    out: Dict[str, Tuple[str, str]] = {}
    for sig_id, sig in signatures.items():
        out[sig_id] = (
            sig.author_info_first_normalized_without_apostrophe or "",
            sig.author_info_middle_normalized_without_apostrophe or "",
        )
    return out


def build_anddata(dataset_name: str, use_sinonym_overwrite: bool, n_jobs: int = 4) -> ANDData:
    data_root = os.path.join(PROJECT_ROOT_PATH, "data", "s2and_mini")
    anddata = ANDData(
        signatures=os.path.join(data_root, dataset_name, dataset_name + "_signatures.json"),
        papers=os.path.join(data_root, dataset_name, dataset_name + "_papers.json"),
        name=dataset_name,
        mode="train",
        specter_embeddings=os.path.join(data_root, dataset_name, dataset_name + "_specter.pickle"),
        clusters=os.path.join(data_root, dataset_name, dataset_name + "_clusters.json"),
        block_type="s2",
        train_pairs=None,
        val_pairs=None,
        test_pairs=None,
        train_pairs_size=100000,
        val_pairs_size=10000,
        test_pairs_size=10000,
        n_jobs=n_jobs,
        load_name_counts=True,
        preprocess=True,
        random_seed=42,
        name_tuples="filtered",
        use_orcid_id=True,
        use_sinonym_overwrite=use_sinonym_overwrite,
    )
    return anddata


def main() -> None:
    os.environ["OMP_NUM_THREADS"] = str(N_JOBS)

    datasets = [
        # "arnetminer",
        "kisti",
    ]

    for dataset_name in datasets:
        print(f"\n=== Dataset: {dataset_name} ===")

        # With Sinonym overwrites
        anddata_yes = build_anddata(dataset_name, use_sinonym_overwrite=True, n_jobs=N_JOBS)
        fm_yes = collect_normalized_first_middle(anddata_yes.signatures)

        # Without Sinonym overwrites
        anddata_no = build_anddata(dataset_name, use_sinonym_overwrite=False, n_jobs=N_JOBS)
        fm_no = collect_normalized_first_middle(anddata_no.signatures)

        # Compute differences on intersection of signature ids
        def only_space_change(a: str, b: str) -> bool:
            """True if a != b but equal after removing spaces."""
            if a is None or b is None:
                return False
            return a != b and a.replace(" ", "") == b.replace(" ", "")

        changed = []
        for sig_id in sorted(set(fm_no.keys()) & set(fm_yes.keys())):
            first_no, middle_no = fm_no[sig_id]
            first_yes, middle_yes = fm_yes[sig_id]
            if first_no != first_yes or middle_no != middle_yes:
                # Exclude trivial diffs where only first changed by adding/removing spaces
                if only_space_change(first_no or "", first_yes or "") and (middle_no == middle_yes):
                    continue
                changed.append((sig_id, first_no, middle_no, first_yes, middle_yes))

        print(f"Total signatures compared: {len(set(fm_no) & set(fm_yes))}")
        print(f"Changed normalized first/middle: {len(changed)}")
        for sig_id, first_no, middle_no, first_yes, middle_yes in changed:
            print(f"sig={sig_id} | first: '{first_no}' -> '{first_yes}' | middle: '{middle_no}' -> '{middle_yes}'")
            # Provide extra context: paper, authors, and Sinonym parsed tokens for this signature
            try:
                sig_obj_no = anddata_no.signatures[sig_id]
                paper_id = str(sig_obj_no.paper_id)
                paper_no = anddata_no.papers[paper_id]
                paper_yes = anddata_yes.papers[paper_id]
                print(f"  paper_id={paper_id} | title={paper_no.title!r}")
                print(
                    "  authors no:",
                    [f"{a.position}:{a.author_name}" for a in paper_no.authors],
                )
                print(
                    "  authors yes:",
                    [f"{a.position}:{a.author_name}" for a in paper_yes.authors],
                )
                # Run Sinonym preprocessing directly to show parsed tokens
                parsed = sinonym_preprocess_papers_parallel({paper_id: paper_no}, n_jobs=1).get(paper_id, {})
                if parsed:
                    print("  sinonym parsed per position (given_tokens | surname_tokens | middle_tokens?):")
                    for pos, obj in sorted(parsed.items()):
                        try:
                            gt = obj.get("given_tokens") if isinstance(obj, dict) else getattr(obj, "given_tokens", [])
                            st = (
                                obj.get("surname_tokens")
                                if isinstance(obj, dict)
                                else getattr(obj, "surname_tokens", [])
                            )
                            mt = None
                            if isinstance(obj, dict):
                                mt = obj.get("middle_tokens")
                            elif hasattr(obj, "middle_tokens"):
                                mt = getattr(obj, "middle_tokens")
                            print(f"    pos={pos}: {gt} | {st} | {mt}")
                        except Exception as e:
                            print(f"    pos={pos}: <failed to display parsed tokens: {e}>")
                else:
                    print("  sinonym parsed: <none>")
            except Exception as e:
                print(f"  <failed to print extra context for {sig_id}: {e}>")

        # -------- Feature comparison on test split --------
        # Load production clusterer to reuse its featurizer settings
        with open(os.path.join(PROJECT_ROOT_PATH, "data", "production_model_v1.1.pickle"), "rb") as fh:
            prod = pickle.load(fh)
        clusterer = prod["clusterer"]
        clusterer.use_cache = False
        clusterer.n_jobs = N_JOBS

        # Use identical test blocks from the build WITHOUT overwrites (blocks are stable in train mode)
        _, _, test_blocks = anddata_no.split_blocks_helper(anddata_no.get_blocks())

        # Build the exact pair list used by prediction (all unordered pairs per block)
        pairs: List[Tuple[str, str, float]] = []
        pair_blocks: List[str] = []
        for block_key, sigs in test_blocks.items():
            n = len(sigs)
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append((sigs[i], sigs[j], np.nan))
                    pair_blocks.append(block_key)

        if len(pairs) == 0:
            print("No test pairs to compare features.")
            continue

        # Compute features for both datasets with identical featurizer config
        feats_no, _, _ = many_pairs_featurize(
            pairs,
            anddata_no,
            clusterer.featurizer_info,
            n_jobs=clusterer.n_jobs,
            use_cache=False,
            chunk_size=DEFAULT_CHUNK_SIZE,
            nameless_featurizer_info=clusterer.nameless_featurizer_info,
        )
        feats_yes, _, _ = many_pairs_featurize(
            pairs,
            anddata_yes,
            clusterer.featurizer_info,
            n_jobs=clusterer.n_jobs,
            use_cache=False,
            chunk_size=DEFAULT_CHUNK_SIZE,
            nameless_featurizer_info=clusterer.nameless_featurizer_info,
        )

        fnames = clusterer.featurizer_info.get_feature_names()
        assert feats_no.shape == feats_yes.shape, "Feature shapes differ between runs"

        # Compare with NaN-safe equality
        a = feats_no
        b = feats_yes
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        row_changed = ~np.all(same, axis=1)
        changed_indices = np.where(row_changed)[0]
        print(f"Total pairwise comparisons: {a.shape[0]}")
        print(f"Pairs with any feature change: {len(changed_indices)}")

        # Summarize which features changed most
        feat_change_counts = np.sum(~same, axis=0)
        any_changed = np.where(feat_change_counts > 0)[0]
        if len(any_changed) == 0:
            print("No feature differences detected.")
        else:
            print("Changed feature counts:")
            for idx in any_changed:
                print(f"  {idx:02d} {fnames[idx]}: {int(feat_change_counts[idx])}")

        # Show a few example diffs
        max_examples = 25
        if len(changed_indices) > 0:
            print(f"\nExample diffs (up to {max_examples}):")
            for k, ridx in enumerate(changed_indices[:max_examples]):
                sig1, sig2, _ = pairs[ridx]
                sig1_no = anddata_no.signatures[sig1]
                sig2_no = anddata_no.signatures[sig2]
                sig1_yes = anddata_yes.signatures[sig1]
                sig2_yes = anddata_yes.signatures[sig2]
                name1_no = ANDData.get_full_name_for_features(sig1_no)
                name2_no = ANDData.get_full_name_for_features(sig2_no)
                name1_yes = ANDData.get_full_name_for_features(sig1_yes)
                name2_yes = ANDData.get_full_name_for_features(sig2_yes)
                block = pair_blocks[ridx]

                print(f"Pair {k+1}: block={block} | {sig1} [{name1_no}] <-> {sig2} [{name2_no}]")
                print(
                    "    NAME no/yes:",
                    f"[{name1_no}] | [{name1_yes}]",
                    "||",
                    f"[{name2_no}] | [{name2_yes}]",
                )
                print(
                    "    BLOCK no/yes:",
                    repr(sig1_no.author_info_block),
                    "|",
                    repr(sig1_yes.author_info_block),
                    "||",
                    repr(sig2_no.author_info_block),
                    "|",
                    repr(sig2_yes.author_info_block),
                )
                # Show raw first/middle (as stored on signatures) for both runs
                print(
                    "    RAW first_no/yes:",
                    repr(sig1_no.author_info_first),
                    "|",
                    repr(sig1_yes.author_info_first),
                    "||",
                    repr(sig2_no.author_info_first),
                    "|",
                    repr(sig2_yes.author_info_first),
                )
                print(
                    "    RAW middle_no/yes:",
                    repr(sig1_no.author_info_middle),
                    "|",
                    repr(sig1_yes.author_info_middle),
                    "||",
                    repr(sig2_no.author_info_middle),
                    "|",
                    repr(sig2_yes.author_info_middle),
                )
                print(
                    "    first_no/yes:",
                    repr(sig1_no.author_info_first_normalized_without_apostrophe),
                    "|",
                    repr(sig1_yes.author_info_first_normalized_without_apostrophe),
                    "||",
                    repr(sig2_no.author_info_first_normalized_without_apostrophe),
                    "|",
                    repr(sig2_yes.author_info_first_normalized_without_apostrophe),
                )
                print(
                    "    middle_no/yes:",
                    repr(sig1_no.author_info_middle_normalized_without_apostrophe),
                    "|",
                    repr(sig1_yes.author_info_middle_normalized_without_apostrophe),
                    "||",
                    repr(sig2_no.author_info_middle_normalized_without_apostrophe),
                    "|",
                    repr(sig2_yes.author_info_middle_normalized_without_apostrophe),
                )
                diff_cols = np.where(~same[ridx])[0]
                for ci in diff_cols:
                    v0 = a[ridx, ci]
                    v1 = b[ridx, ci]
                    print(f"    {ci:02d} {fnames[ci]}: {v0} -> {v1}")

        # Also compare pairwise classifier probabilities and crossings vs eps
        print("\n--- Pairwise probability comparison ---")
        # p(not same) from the classifier(s)
        if clusterer.nameless_classifier is not None:
            p_no = (
                clusterer.classifier.predict_proba(feats_no)[:, 0]
                + clusterer.nameless_classifier.predict_proba(
                    many_pairs_featurize(
                        pairs,
                        anddata_no,
                        clusterer.featurizer_info,
                        n_jobs=clusterer.n_jobs,
                        use_cache=False,
                        chunk_size=DEFAULT_CHUNK_SIZE,
                        nameless_featurizer_info=clusterer.nameless_featurizer_info,
                    )[
                        2
                    ]  # nameless features
                )[:, 0]
            ) / 2
            p_yes = (
                clusterer.classifier.predict_proba(feats_yes)[:, 0]
                + clusterer.nameless_classifier.predict_proba(
                    many_pairs_featurize(
                        pairs,
                        anddata_yes,
                        clusterer.featurizer_info,
                        n_jobs=clusterer.n_jobs,
                        use_cache=False,
                        chunk_size=DEFAULT_CHUNK_SIZE,
                        nameless_featurizer_info=clusterer.nameless_featurizer_info,
                    )[2]
                )[:, 0]
            ) / 2
        else:
            p_no = clusterer.classifier.predict_proba(feats_no)[:, 0]
            p_yes = clusterer.classifier.predict_proba(feats_yes)[:, 0]

        eps = getattr(clusterer.cluster_model, "eps", 0.5)
        crossed = (p_no < eps) ^ (p_yes < eps)
        print(
            f"Prob delta summary: mean|std={float(np.mean(p_yes-p_no)):.4f}|{float(np.std(p_yes-p_no)):.4f}; "
            f"crossed eps ({eps:.3f}): {int(np.sum(crossed))} of {len(p_no)}"
        )

        # Show top crossing examples by absolute delta
        idx_sorted = np.argsort(-np.abs(p_yes - p_no))
        shown = 0
        for ridx in idx_sorted:
            if not crossed[ridx]:
                continue
            sig1, sig2, _ = pairs[ridx]
            print(
                f"  pair {sig1} <-> {sig2}: p_no={p_no[ridx]:.3f} -> p_yes={p_yes[ridx]:.3f} (Δ={p_yes[ridx]-p_no[ridx]:+.3f})"
            )
            shown += 1
            if shown >= 10:
                break

        # -------- End-to-end clustering comparison on test split --------
        print("\n--- Clustering comparison (test split) ---")

        # Metrics for both runs
        m_no, _ = cluster_eval(anddata_no, clusterer, split="test", use_s2_clusters=False)
        m_yes, _ = cluster_eval(anddata_yes, clusterer, split="test", use_s2_clusters=False)
        print("Metrics no-overwrite:", m_no)
        print("Metrics yes-overwrite:", m_yes)

        # Predicted clusters for both runs using identical test blocks (from no-overwrite)
        pred_no, _ = clusterer.predict(test_blocks, anddata_no, use_s2_clusters=False)
        pred_yes, _ = clusterer.predict(test_blocks, anddata_yes, use_s2_clusters=False)

        # Helper: build signature->cluster map and same-cluster pair set
        def invert_clusters(pred_clusters):
            sig_to_cid = {}
            for cid, sigs in pred_clusters.items():
                for s in sigs:
                    sig_to_cid[s] = cid
            return sig_to_cid

        def same_cluster_pairs(pred_clusters):
            pairs_local = set()
            for sigs in pred_clusters.values():
                n = len(sigs)
                for i in range(n):
                    for j in range(i + 1, n):
                        a, b = sigs[i], sigs[j]
                        if a <= b:
                            pairs_local.add((a, b))
                        else:
                            pairs_local.add((b, a))
            return pairs_local

        sig_to_cid_no = invert_clusters(pred_no)
        sig_to_cid_yes = invert_clusters(pred_yes)
        pairs_no = same_cluster_pairs(pred_no)
        pairs_yes = same_cluster_pairs(pred_yes)

        only_no = pairs_no - pairs_yes
        only_yes = pairs_yes - pairs_no
        inter = pairs_no & pairs_yes

        denom = max(1, len(pairs_no | pairs_yes))
        jacc = len(inter) / denom
        print(
            f"Same-cluster pair counts | no={len(pairs_no)} yes={len(pairs_yes)} inter={len(inter)} jaccard={jacc:.3f}"
        )
        print(f"Pairs only in no (splits in yes): {len(only_no)}")
        print(f"Pairs only in yes (merges in yes): {len(only_yes)}")

        # Show a few illustrative differences
        def describe_pair(sig_a: str, sig_b: str):
            a_no = anddata_no.signatures[sig_a]
            b_no = anddata_no.signatures[sig_b]
            a_yes = anddata_yes.signatures[sig_a]
            b_yes = anddata_yes.signatures[sig_b]
            name_a_no = ANDData.get_full_name_for_features(a_no)
            name_b_no = ANDData.get_full_name_for_features(b_no)
            name_a_yes = ANDData.get_full_name_for_features(a_yes)
            name_b_yes = ANDData.get_full_name_for_features(b_yes)
            print(f"  {sig_a}: [{name_a_no}] |yes-> [{name_a_yes}]  ||  {sig_b}: [{name_b_no}] |yes-> [{name_b_yes}]")
            # Also show first/middle normalized for both
            print(
                "    first_no/yes:",
                repr(a_no.author_info_first_normalized_without_apostrophe),
                "|",
                repr(a_yes.author_info_first_normalized_without_apostrophe),
                "||",
                repr(b_no.author_info_first_normalized_without_apostrophe),
                "|",
                repr(b_yes.author_info_first_normalized_without_apostrophe),
            )
            print(
                "    middle_no/yes:",
                repr(a_no.author_info_middle_normalized_without_apostrophe),
                "|",
                repr(a_yes.author_info_middle_normalized_without_apostrophe),
                "||",
                repr(b_no.author_info_middle_normalized_without_apostrophe),
                "|",
                repr(b_yes.author_info_middle_normalized_without_apostrophe),
            )

        max_show = 10
        if only_no:
            print(f"\nExamples only in no (split in yes): showing up to {max_show}")
            for i, (a, b) in enumerate(list(only_no)[:max_show]):
                describe_pair(a, b)
        if only_yes:
            print(f"\nExamples only in yes (merged in yes): showing up to {max_show}")
            for i, (a, b) in enumerate(list(only_yes)[:max_show]):
                describe_pair(a, b)


if __name__ == "__main__":
    main()
