"""Tiny real pairwise-release inputs: JSON dataset, runtime artifacts and provenance-bound Arrow."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from s2and._sha256 import sha256_file
from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from s2and.name_tuple_artifact import load_name_tuple_artifact
from scripts.convert_to_arrow import benchmark_dataset_sources, convert_runtime_dataset_to_arrow
from scripts.production.counts.generate_orcid_name_prefix_counts import write_publication
from scripts.production.generate_canonical_name_tuples import regenerate

PAIRWISE_LAST_NAMES = (
    "Adams",
    "Baker",
    "Clark",
    "Davis",
    "Evans",
    "Foster",
    "Green",
    "Harris",
    "Irwin",
    "Jones",
)
# Plan roles whose files determine Rust pair features.
ARROW_SOURCE_ROLES = ("signatures", "papers", "specter_embeddings")


def write_pairwise_dataset(data_dir: Path, name: str = "qian") -> dict[str, Path]:
    """Write a 40-signature benchmark-layout JSON dataset with two clusters per block."""

    dataset_dir = data_dir / name
    dataset_dir.mkdir(parents=True)
    signatures: dict[str, dict[str, Any]] = {}
    papers: dict[str, dict[str, Any]] = {}
    clusters: dict[str, dict[str, Any]] = {}
    embeddings: dict[str, np.ndarray] = {}

    for block_index, last_name in enumerate(PAIRWISE_LAST_NAMES):
        block = f"a {last_name.lower()}"
        for author_index in range(4):
            signature_id = f"signature-{block_index}-{author_index}"
            paper_id = f"paper-{block_index}-{author_index}"
            cluster_index = author_index // 2
            first_name = "Alex" if cluster_index == 0 else "Avery"
            affiliation = f"Institute {cluster_index}"
            coauthor = f"Coauthor {cluster_index}"
            signatures[signature_id] = {
                "signature_id": signature_id,
                "paper_id": paper_id,
                "author_info": {
                    "position": 0,
                    "block": block,
                    "first": first_name,
                    "middle": None,
                    "last": last_name,
                    "suffix": None,
                    "email": f"{first_name.lower()}@example.org",
                    "affiliations": [affiliation],
                },
            }
            papers[paper_id] = {
                "paper_id": paper_id,
                "title": f"Topic {cluster_index} study {block_index}",
                "abstract": f"Deterministic abstract for topic {cluster_index}.",
                "authors": [
                    {"author_name": f"{first_name} {last_name}", "position": 0},
                    {"author_name": coauthor, "position": 1},
                ],
                "venue": f"Venue {cluster_index}",
                "journal_name": f"Journal {cluster_index}",
                "year": 2010 + block_index,
            }
            vector = np.zeros(768, dtype=np.float32)
            vector[cluster_index] = 1.0
            vector[10 + block_index] = 0.1
            vector[100 + author_index] = 0.01
            embeddings[paper_id] = vector
            cluster_id = f"cluster-{block_index}-{cluster_index}"
            clusters.setdefault(
                cluster_id,
                {
                    "cluster_id": cluster_id,
                    "signature_ids": [],
                    "model_version": -1,
                },
            )["signature_ids"].append(signature_id)

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


def write_pairwise_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write canonical name tuples, a name-count index and ORCID prefix counts for the tiny dataset."""

    tuple_source = tmp_path / "raw_name_tuples.txt"
    tuple_source.write_text("William,Bill\n", encoding="utf-8")
    tuple_path = tmp_path / "canonical_name_tuples.txt"
    regenerate(str(tuple_source), str(tuple_path))
    name_tuples = load_name_tuple_artifact(tuple_path)

    first_counts = {"alex": 20, "avery": 20}
    last_counts = {last_name.lower(): 4 for last_name in PAIRWISE_LAST_NAMES}
    first_last_counts = {
        f"{first} {last_name.lower()}": 2 for first in first_counts for last_name in PAIRWISE_LAST_NAMES
    }
    last_first_initial_counts = {f"{last_name.lower()} a": 4 for last_name in PAIRWISE_LAST_NAMES}
    name_counts_path, _ = write_name_counts_index(
        tmp_path / "name-counts",
        (
            first_counts,
            last_counts,
            first_last_counts,
            last_first_initial_counts,
        ),
    )

    orcid_counts_path = tmp_path / "orcid-prefix-counts"
    write_publication(
        {},
        output_dir=orcid_counts_path,
        name_tuples=name_tuples,
    )
    return tuple_path, Path(name_counts_path), orcid_counts_path


def convert_release_arrow(source_root: Path, name: str, arrow_root: Path, name_counts_path: Path) -> Path:
    """Convert one JSON dataset with the production converter, bound to ``name_counts_path``."""

    convert_runtime_dataset_to_arrow(
        sources=benchmark_dataset_sources(source_root, name),
        output_dir=arrow_root / name,
        root_manifest_dir=arrow_root,
        name_counts_index_root=name_counts_path.parent,
        n_jobs=1,
        overwrite=True,
        skip_name_counts_index=False,
        selected_embedding="specter2",
        validate=True,
    )
    return arrow_root / name


def source_sha256(files: dict[str, Path]) -> dict[str, str]:
    """Return the digests of the plan roles that bind an Arrow conversion."""

    return {role: sha256_file(files[role]) for role in ARROW_SOURCE_ROLES}
