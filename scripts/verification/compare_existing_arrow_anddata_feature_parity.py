"""Compare existing Arrow feature generation against raw ANDData feature generation.

This is a bounded verification gate for release artifacts that already exist in
``s2and/data`` and original JSON/pickle inputs that exist in ``s2and/data-backup``.
It compares feature matrices from two supported ingestion paths:

- Python featurization over a preprocessed ``ANDData`` subset.
- one retained ``ArrowDataset`` over the existing Arrow bundle.

The target is ingestion/preprocessing parity after the Rust ``from_dataset``
constructor removal.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as infile:
        return json.load(infile)


def _select_signature_ids(
    signatures: Mapping[str, Any],
    *,
    limit: int,
    seed: int,
    sample_mode: str,
) -> list[str]:
    signature_ids = sorted(str(signature_id) for signature_id in signatures)
    if len(signature_ids) <= limit:
        return signature_ids
    if sample_mode == "first":
        return signature_ids[:limit]
    rng = random.Random(seed)
    return sorted(rng.sample(signature_ids, limit))


def _filter_raw_payloads(
    signatures: Mapping[str, Any],
    papers: Mapping[str, Any],
    selected_signature_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
    selected_set = {str(signature_id) for signature_id in selected_signature_ids}
    filtered_signatures = {
        str(signature_id): payload for signature_id, payload in signatures.items() if str(signature_id) in selected_set
    }
    paper_ids = {
        str(payload.get("paper_id", payload.get("paperId")))
        for payload in filtered_signatures.values()
        if isinstance(payload, Mapping) and payload.get("paper_id", payload.get("paperId")) is not None
    }
    filtered_papers = {str(paper_id): payload for paper_id, payload in papers.items() if str(paper_id) in paper_ids}
    missing_papers = sorted(paper_id for paper_id in paper_ids if paper_id not in filtered_papers)
    if missing_papers:
        raise ValueError(f"Raw papers are missing selected paper ids: {missing_papers[:10]}")
    return filtered_signatures, filtered_papers, paper_ids


def _load_specter_subset(path: Path, needed_paper_ids: set[str]) -> dict[str, np.ndarray]:
    with path.open("rb") as infile:
        payload = pickle.load(infile)
    if isinstance(payload, tuple) and len(payload) == 2:
        matrix, paper_ids = payload
        matrix = np.asarray(matrix, dtype=np.float32)
        if len(paper_ids) != matrix.shape[0]:
            raise ValueError(f"SPECTER tuple ids={len(paper_ids)} rows={matrix.shape[0]}")
        return {
            str(paper_id): np.asarray(matrix[index], dtype=np.float32)
            for index, paper_id in enumerate(paper_ids)
            if str(paper_id) in needed_paper_ids
        }
    if not isinstance(payload, Mapping):
        raise ValueError(f"Unsupported SPECTER payload type: {type(payload).__name__}")
    out: dict[str, np.ndarray] = {}
    for paper_id in needed_paper_ids:
        value = payload.get(paper_id)
        if value is None:
            try:
                value = payload.get(int(paper_id))
            except ValueError:
                value = None
        if value is not None:
            out[paper_id] = np.asarray(value, dtype=np.float32)
    return out


def _sample_pairs(signature_ids: Sequence[str], *, pair_count: int, seed: int) -> list[tuple[str, str]]:
    if len(signature_ids) < 2:
        return []
    rng = random.Random(seed)
    unique_pairs: set[tuple[str, str]] = set()
    max_pairs = len(signature_ids) * (len(signature_ids) - 1) // 2
    target = min(pair_count, max_pairs)
    while len(unique_pairs) < target:
        left, right = rng.sample(list(signature_ids), 2)
        left_id, right_id = sorted((str(left), str(right)))
        unique_pairs.add((left_id, right_id))
    return sorted(unique_pairs)


def _feature_matrix(featurizer: Any, pairs: Sequence[tuple[str, str]], *, n_jobs: int) -> np.ndarray:
    index_by_signature_id = {str(signature_id): index for index, signature_id in enumerate(featurizer.signature_ids())}
    missing = sorted(
        {signature_id for pair in pairs for signature_id in pair if str(signature_id) not in index_by_signature_id}
    )
    if missing:
        raise ValueError(f"Featurizer is missing selected signature ids: {missing[:10]}")
    index_pairs = [(index_by_signature_id[str(left)], index_by_signature_id[str(right)]) for left, right in pairs]
    return np.asarray(
        featurizer.featurize_pairs_matrix_indexed(index_pairs, None, int(n_jobs), np.nan),
        dtype=np.float64,
    )


def _numeric_report(left: np.ndarray, right: np.ndarray, *, atol: float, rtol: float) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        return {
            "shape_match": False,
            "left_shape": list(left_array.shape),
            "right_shape": list(right_array.shape),
        }
    left_nan = np.isnan(left_array)
    right_nan = np.isnan(right_array)
    comparable = ~(left_nan | right_nan)
    diff = np.abs(left_array[comparable] - right_array[comparable])
    tolerance = atol + rtol * np.abs(right_array[comparable])
    over_tolerance = diff > tolerance
    mismatch_examples: list[dict[str, Any]] = []
    if np.any(over_tolerance):
        flat_indices = np.flatnonzero(comparable)
        comparable_indices = flat_indices[np.flatnonzero(over_tolerance)[:10]]
        for flat_index in comparable_indices:
            row, column = np.unravel_index(int(flat_index), left_array.shape)
            mismatch_examples.append(
                {
                    "row": int(row),
                    "column": int(column),
                    "anddata": float(left_array[row, column]),
                    "arrow": float(right_array[row, column]),
                    "absdiff": float(abs(left_array[row, column] - right_array[row, column])),
                }
            )
    return {
        "shape_match": True,
        "nan_mismatch_count": int(np.count_nonzero(left_nan != right_nan)),
        "max_absdiff": float(diff.max()) if diff.size else 0.0,
        "mean_absdiff": float(diff.mean()) if diff.size else 0.0,
        "over_tolerance_count": int(np.count_nonzero(over_tolerance)),
        "allclose_equal_nan": bool(np.allclose(left_array, right_array, rtol=rtol, atol=atol, equal_nan=True)),
        "mismatch_examples": mismatch_examples,
    }


def _run_dataset(args: argparse.Namespace, dataset_name: str) -> dict[str, Any]:
    from s2and.arrow_inputs import ArrowDataset
    from s2and.consts import NAME_COUNTS_INDEX_PATH
    from s2and.data import ANDData
    from s2and.feature_port import build_rust_featurizer_from_arrow_dataset
    from s2and.featurizer import FeaturizationInfo, many_pairs_featurize
    from s2and.runtime import RuntimeContext

    raw_dir = args.raw_root / dataset_name
    arrow_dir = args.arrow_root / dataset_name
    signatures = _load_json(raw_dir / f"{dataset_name}_signatures.json")
    papers = _load_json(raw_dir / f"{dataset_name}_papers.json")
    selected_signature_ids = _select_signature_ids(
        signatures,
        limit=int(args.limit_signatures),
        seed=int(args.seed),
        sample_mode=str(args.sample_mode),
    )
    filtered_signatures, filtered_papers, paper_ids = _filter_raw_payloads(signatures, papers, selected_signature_ids)
    specter_suffix = "_specter2.pkl" if args.embedding == "specter2" else "_specter.pickle"
    specter_embeddings = _load_specter_subset(raw_dir / f"{dataset_name}{specter_suffix}", paper_ids)
    if len(specter_embeddings) != len(paper_ids):
        missing_count = len(paper_ids) - len(specter_embeddings)
        if missing_count > 0:
            raise ValueError(f"Raw SPECTER payload is missing {missing_count} selected paper embeddings")

    started = time.perf_counter()
    dataset = ANDData(
        signatures=filtered_signatures,
        papers=filtered_papers,
        name=f"{dataset_name}_existing_arrow_anddata_feature_parity",
        mode="inference",
        clusters=None,
        specter_embeddings=specter_embeddings,
        cluster_seeds=None,
        train_pairs=None,
        val_pairs=None,
        test_pairs=None,
        n_jobs=int(args.n_jobs),
        name_counts_index=NAME_COUNTS_INDEX_PATH,
        preprocess=True,
        random_seed=int(args.seed),
        name_tuples=None,
        use_orcid_id=True,
    )
    anddata_seconds = time.perf_counter() - started

    pairs = _sample_pairs(selected_signature_ids, pair_count=int(args.pair_count), seed=int(args.seed))
    started = time.perf_counter()
    anddata_features, _labels, _nameless_features = many_pairs_featurize(
        [(left, right, 0) for left, right in pairs],
        dataset,
        FeaturizationInfo(),
        n_jobs=int(args.n_jobs),
        chunk_size=max(1, int(args.pair_count)),
        nan_value=np.nan,
        runtime_context=RuntimeContext(
            operation="verification_python_anddata_feature_parity",
            backend="python",
            run_id="verification-python-anddata",
        ),
    )
    python_feature_seconds = time.perf_counter() - started

    with ArrowDataset.open(
        arrow_dir,
        require_specter=True,
        require_name_counts_index=True,
    ) as arrow_dataset:
        started = time.perf_counter()
        arrow_featurizer = build_rust_featurizer_from_arrow_dataset(
            arrow_dataset,
            signature_ids=selected_signature_ids,
            name_tuples=None,
            preprocess=True,
            num_threads=int(args.n_jobs),
        )
        arrow_featurizer_seconds = time.perf_counter() - started

        started = time.perf_counter()
        arrow_features = _feature_matrix(arrow_featurizer, pairs, n_jobs=int(args.n_jobs))
        arrow_feature_seconds = time.perf_counter() - started

    report = _numeric_report(anddata_features, arrow_features, atol=float(args.atol), rtol=float(args.rtol))
    report.update(
        {
            "dataset": dataset_name,
            "signature_count": int(len(selected_signature_ids)),
            "paper_count": int(len(paper_ids)),
            "pair_count": int(len(pairs)),
            "feature_count": int(anddata_features.shape[1]) if anddata_features.ndim == 2 else 0,
            "embedding": str(args.embedding),
            "timings_seconds": {
                "anddata_build": float(anddata_seconds),
                "python_features": float(python_feature_seconds),
                "arrow_featurizer": float(arrow_featurizer_seconds),
                "arrow_features": float(arrow_feature_seconds),
            },
        }
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("s2and/data-backup"))
    parser.add_argument("--arrow-root", type=Path, default=Path("s2and/data"))
    parser.add_argument("--datasets", nargs="+", default=["pubmed", "qian", "zbmath"])
    parser.add_argument("--limit-signatures", type=int, default=64)
    parser.add_argument("--pair-count", type=int, default=128)
    parser.add_argument("--sample-mode", choices=("random", "first"), default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding", choices=("specter", "specter2"), default="specter2")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--allow-mismatch", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, int(args.n_jobs))))
    results = [_run_dataset(args, str(dataset_name)) for dataset_name in args.datasets]
    summary = {
        "raw_root": str(args.raw_root),
        "arrow_root": str(args.arrow_root),
        "datasets": results,
        "all_passed": all(
            bool(result.get("shape_match"))
            and int(result.get("nan_mismatch_count", 0)) == 0
            and int(result.get("over_tolerance_count", 0)) == 0
            for result in results
        ),
        "atol": float(args.atol),
        "rtol": float(args.rtol),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not args.allow_mismatch and not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
