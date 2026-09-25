"""Rust coauthor features on papers with hundreds to thousands of authors.

The featurizer aggregates each paper's author list once and subtracts a
signature's own author per pair. These tests pin the features to the Python
reference on large author lists with repeated names and blocks, and pin build,
featurize and free to stay fast on a 3,000-author paper, where per-signature
coauthor records took O(authors^2) allocations.
"""

from __future__ import annotations

import json
import pickle
import random
import string
import time
from pathlib import Path
from typing import Any

import numpy as np

from s2and.arrow_inputs import RAW_PLANNER_ARROW_BATCH_INDEX_KEYS, RAW_PLANNER_ARROW_KEY_COLUMNS, ArrowDataset
from s2and.data import ANDData
from s2and.feature_port import (
    build_rust_featurizer_from_arrow_dataset,
    build_verified_rust_featurizer,
    featurize_pairs_with_rust,
)
from s2and.feature_schema import FEATURE_SCHEMA
from s2and.featurizer import FeaturizationInfo, many_pairs_featurize
from s2and.name_counts_index import NameCountsIndex
from s2and.name_tuple_artifact import load_name_tuple_artifact
from tests.helpers import write_test_arrow_artifact_manifest
from tests.pairwise_release_helpers import convert_release_arrow, write_pairwise_artifacts

COAUTHOR_COLUMNS = [
    index
    for index, spec in enumerate(FEATURE_SCHEMA)
    if spec.name in {"coauthor_overlap", "coauthor_similarity", "coauthor_match"}
]
FIRSTS = ("Alex", "Avery", "Jian", "Maria", "J", "Li Wei")
LASTS = ("Adams", "Baker", "Wang", "Li", "O'Neil", "Garcia Lopez")


def _letters(index: int, width: int = 3) -> str:
    letters = []
    for _ in range(width):
        index, digit = divmod(index, 26)
        letters.append(string.ascii_lowercase[digit])
    return "".join(letters)


def _author(index: int) -> tuple[str, str]:
    """Pool names repeat within and across papers; every third author has a unique surname."""

    first = FIRSTS[index % len(FIRSTS)]
    last = f"Q{_letters(index)}" if index % 3 == 0 else LASTS[(index // len(FIRSTS)) % len(LASTS)]
    return first, last


def _write_large_author_dataset(data_dir: Path, name: str) -> dict[str, Path]:
    """Two large papers sharing names, small papers from the same name pool, singleton clusters."""

    rng = random.Random(3)
    paper_authors: dict[str, list[int]] = {
        "paper-large-0": list(range(420)),
        "paper-large-1": list(range(200, 450)),
    }
    for paper in range(30):
        paper_authors[f"paper-small-{paper}"] = rng.sample(range(450), rng.randint(1, 8))
    signature_positions: dict[str, list[int]] = {
        "paper-large-0": sorted(rng.sample(range(420), 40)),
        "paper-large-1": sorted(rng.sample(range(250), 25)),
    }
    for paper in range(30):
        signature_positions[f"paper-small-{paper}"] = [0]

    dataset_dir = data_dir / name
    dataset_dir.mkdir(parents=True)
    papers: dict[str, Any] = {}
    signatures: dict[str, Any] = {}
    embeddings: dict[str, np.ndarray] = {}
    for paper_index, (paper_id, authors) in enumerate(paper_authors.items()):
        papers[paper_id] = {
            "paper_id": paper_id,
            "title": f"Collaboration results {paper_index}",
            "abstract": "Measurements.",
            "authors": [
                {"author_name": " ".join(_author(author)), "position": position}
                for position, author in enumerate(authors)
            ],
            "venue": "Physics Letters",
            "journal_name": "Physics Letters",
            "year": 2000 + paper_index % 20,
        }
        embeddings[paper_id] = np.random.default_rng(paper_index).standard_normal(768).astype(np.float32)
        for position in signature_positions[paper_id]:
            first, last = _author(authors[position])
            signature_id = f"{paper_id}-{position}"
            signatures[signature_id] = {
                "signature_id": signature_id,
                "paper_id": paper_id,
                "author_info": {
                    "position": position,
                    "block": f"{first[0].lower()} {last.lower()}",
                    "first": first,
                    "middle": None,
                    "last": last,
                    "suffix": None,
                    "email": None,
                    "affiliations": [],
                },
            }
    clusters = {
        f"cluster-{signature_id}": {
            "cluster_id": f"cluster-{signature_id}",
            "signature_ids": [signature_id],
            "model_version": -1,
        }
        for signature_id in signatures
    }
    for filename, payload in (
        (f"{name}_signatures.json", signatures),
        (f"{name}_papers.json", papers),
        (f"{name}_clusters.json", clusters),
    ):
        (dataset_dir / filename).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with (dataset_dir / f"{name}_specter2.pkl").open("wb") as output:
        pickle.dump(embeddings, output, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "clusters": dataset_dir / f"{name}_clusters.json",
        "papers": dataset_dir / f"{name}_papers.json",
        "signatures": dataset_dir / f"{name}_signatures.json",
        "specter_embeddings": dataset_dir / f"{name}_specter2.pkl",
    }


def test_large_author_list_features_match_python(tmp_path: Path) -> None:
    """Coauthor features equal Python's exactly on repeated names, shared blocks and same-paper pairs."""

    name = "qian"
    files = _write_large_author_dataset(tmp_path / "data", name)
    tuple_path, name_counts_path, _orcid = write_pairwise_artifacts(tmp_path)
    convert_release_arrow(tmp_path / "data", name, tmp_path / "arrow", name_counts_path)
    dataset = ANDData(
        signatures=str(files["signatures"]),
        papers=str(files["papers"]),
        name=name,
        mode="train",
        specter_embeddings=str(files["specter_embeddings"]),
        clusters=str(files["clusters"]),
        train_pairs_size=10,
        val_pairs_size=2,
        random_seed=1111,
        name_counts_index=NameCountsIndex.open(name_counts_path),
        n_jobs=1,
        preprocess=True,
        name_tuples=load_name_tuple_artifact(tuple_path).pairs,
    )
    signature_ids = sorted(dataset.signatures)
    rng = random.Random(5)
    pairs: list[tuple[str, str, int | float]] = []
    for _ in range(500):
        left, right = rng.sample(signature_ids, 2)
        pairs.append((left, right, 0))
    large = [signature_id for signature_id in signature_ids if signature_id.startswith("paper-large-0")]
    pairs += [(large[index], large[index + 1], 0) for index in range(len(large) - 1)]
    info = FeaturizationInfo()
    python, _, _ = many_pairs_featurize(pairs, dataset, info, n_jobs=1, nan_value=np.nan)

    with ArrowDataset.open(tmp_path / "arrow" / name, require_specter=True, require_name_counts_index=True) as arrow:
        featurizer = build_verified_rust_featurizer(
            dataset,
            arrow,
            signature_ids=signature_ids,
            name_counts_manifest_sha256=NameCountsIndex.open(name_counts_path).manifest_sha256,
            num_threads=2,
        )
        rust, _, _ = featurize_pairs_with_rust(
            featurizer,
            pairs,
            info,
            nameless_featurizer_info=None,
            n_jobs=2,
            nan_value=np.nan,
            total_ram_bytes=1_000_000_000,
        )

    assert np.array_equal(rust[:, COAUTHOR_COLUMNS].view(np.uint64), python[:, COAUTHOR_COLUMNS].view(np.uint64))
    np.testing.assert_allclose(rust, python, rtol=0.0, atol=1e-12, equal_nan=True)
    coauthors = python[:, COAUTHOR_COLUMNS]
    assert np.isnan(coauthors).any(), "a single-author paper gives missing coauthor features"
    assert ((coauthors > 0) & (coauthors < 1)).any(axis=0).all(), "every coauthor feature takes fractional values"


def _write_arrow_bundle(root: Path, tables: dict[str, Any]) -> None:
    from s2and.incremental_linking.feature_block import write_arrow_batch_lookup_index, write_arrow_ipc_table

    root.mkdir(parents=True)
    paths: dict[str, str] = {}
    for table_name, table in tables.items():
        arrow_path = root / f"{table_name}.arrow"
        paths[table_name] = write_arrow_ipc_table(table, arrow_path)
        index_key = RAW_PLANNER_ARROW_BATCH_INDEX_KEYS[table_name]
        paths[index_key], _metrics = write_arrow_batch_lookup_index(
            arrow_path,
            root / f"{table_name}.{index_key}.bin",
            key_column=RAW_PLANNER_ARROW_KEY_COLUMNS[table_name],
            table_name=table_name,
        )
    write_test_arrow_artifact_manifest(root, paths)


def test_giant_paper_featurizer_builds_featurizes_and_frees_quickly(tmp_path: Path) -> None:
    """A 3,000-author paper with a signature per author builds, featurizes and frees in well under a second.

    Per-signature coauthor records needed ~9e6 name strings and ~2.7e8 n-gram
    strings here and took tens of seconds to build and free.
    """

    import pyarrow as pa

    authors = 3000
    firsts = ["Anna"] * authors
    lasts = [f"Q{_letters(index)}" for index in range(authors)]
    signature_ids = [str(index) for index in range(authors)]
    _write_arrow_bundle(
        tmp_path / "arrow",
        {
            "signatures": pa.table(
                {
                    "signature_id": pa.array(signature_ids, type=pa.string()),
                    "paper_id": pa.array(["giant"] * authors, type=pa.string()),
                    "author_first": pa.array(firsts, type=pa.string()),
                    "author_middle": pa.array([""] * authors, type=pa.string()),
                    "author_last": pa.array(lasts, type=pa.string()),
                    "author_suffix": pa.array([""] * authors, type=pa.string()),
                    "author_affiliations": pa.array([[]] * authors, type=pa.list_(pa.string())),
                    "author_position": pa.array(range(authors), type=pa.int64()),
                    "author_block": pa.array([f"a {last.lower()}" for last in lasts], type=pa.string()),
                }
            ),
            "papers": pa.table(
                {
                    "paper_id": pa.array(["giant"], type=pa.string()),
                    "title": pa.array(["Observation of a new particle"], type=pa.string()),
                    "venue": pa.array([""], type=pa.string()),
                    "journal_name": pa.array([""], type=pa.string()),
                }
            ),
            "paper_authors": pa.table(
                {
                    "paper_id": pa.array(["giant"] * authors, type=pa.string()),
                    "position": pa.array(range(authors), type=pa.int64()),
                    "author_name": pa.array([f"Anna {last}" for last in lasts], type=pa.string()),
                }
            ),
        },
    )
    pairs = [(index, (index + 1) % authors) for index in range(authors)]
    with ArrowDataset.open(tmp_path / "arrow") as arrow:
        started = time.perf_counter()
        featurizer = build_rust_featurizer_from_arrow_dataset(arrow, name_tuples=frozenset(), num_threads=2)
        index = {signature_id: position for position, signature_id in enumerate(featurizer.signature_ids())}
        features = np.asarray(
            featurizer.featurize_pairs_matrix_indexed(
                [(index[str(left)], index[str(right)]) for left, right in pairs], COAUTHOR_COLUMNS, 2, np.nan
            )
        )
        del featurizer
        elapsed = time.perf_counter() - started

    # Distinct names and blocks: each side drops its own author, so |A & B| = n - 2 and |A | B| = n.
    expected_jaccard = (authors - 2) / authors
    overlap, similarity, match = features.T
    assert np.all(overlap == expected_jaccard)
    assert np.all(match == expected_jaccard)
    assert np.all(similarity == 1.0), "tens of thousands of shared n-grams exceed the 5000 denominator cap"
    assert elapsed < 3.0, f"giant-paper featurizer took {elapsed:.2f}s to build, featurize and free"
