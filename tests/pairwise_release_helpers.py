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
        name_counts_index_root=name_counts_path,
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


def write_prepared_pairwise_root(tmp_path: Path) -> Path:
    """Write a prepared pairwise input tree with release/evaluation Arrow tables and augmented splits."""
    import shutil

    root = tmp_path / "prepared"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(exist_ok=True)
    _, index, _ = write_pairwise_artifacts(artifacts)
    shutil.copytree(index, root / "name_counts_global" / "name_counts_index")
    model = root / "populations" / "model_source"
    evaluation = root / "populations" / "evaluation_source"
    write_pairwise_dataset(model)
    shutil.copytree(model / "qian", evaluation / "qian")
    sidecar = "restored_unknown_signature_ids_20261009.json"
    for source in (model / "qian", evaluation / "qian"):
        (source / sidecar).write_text("[]", encoding="utf-8")
    for filename in ("test_pairs.json", "test_blocks.json"):
        (evaluation / "qian" / filename).write_text("{}", encoding="utf-8")
    write_pairwise_dataset(root, "augmented")
    for folder, source in (("release", model), ("evaluation", evaluation)):
        arrow = root / "arrow" / folder
        convert_release_arrow(source, "qian", arrow, index)
        convert_release_arrow(root, "augmented", arrow, index)
    for filename, first, second in (("train_pairs.csv", "0-0", "0-1"), ("val_pairs.csv", "0-2", "0-3")):
        (root / "augmented" / filename).write_text(
            f"signature_id_1,signature_id_2,label\nsignature-{first},signature-{second},1\n",
            encoding="utf-8",
        )
    (root / "augmented" / "test_pairs.json").write_text(
        json.dumps([{"signature_id_1": "signature-1-0", "signature_id_2": "signature-1-1", "label": 1}]),
        encoding="utf-8",
    )
    return root


def write_linker_arrow_collection_root(
    root: Path,
    *,
    dataset: str,
    replay: bool,
    name_counts_source: Path | None = None,
) -> None:
    """Write a tiny generic Arrow collection root (benchmark or replay layout) with batch lookup indexes."""
    import shutil

    import pyarrow as pa

    from s2and.arrow_inputs import ARROW_COLLECTION_KIND, build_arrow_artifact_manifest, write_arrow_artifact_manifest
    from s2and.consts import PUBLIC_DATA_FORMAT_VERSION
    from s2and.incremental_linking.feature_block import write_arrow_batch_lookup_index, write_arrow_ipc_table
    from tests.helpers import tiny_name_counts_tuple

    root.mkdir(parents=True)
    (root / "LICENSE.txt").write_text("fixture license\n", encoding="utf-8")
    if name_counts_source is None:
        write_name_counts_index(root, tiny_name_counts_tuple())
    else:
        shutil.copytree(name_counts_source, root / "name_counts_index")
    dataset_root = root / "datasets" / dataset if replay else root / dataset

    table_specs = {
        "signatures": (
            pa.table(
                {
                    "signature_id": pa.array(["s1"], type=pa.string()),
                    "paper_id": pa.array(["p1"], type=pa.string()),
                    "author_first": pa.array(["Ada"], type=pa.string()),
                    "author_middle": pa.array([""], type=pa.string()),
                    "author_last": pa.array(["Lovelace"], type=pa.string()),
                    "author_suffix": pa.array([""], type=pa.string()),
                    "author_affiliations": pa.array([[]], type=pa.list_(pa.string())),
                    "author_position": pa.array([0], type=pa.int64()),
                }
            ),
            "signature_id",
        ),
        "papers": (
            pa.table(
                {
                    "paper_id": pa.array(["p1"], type=pa.string()),
                    "title": pa.array(["Notes"], type=pa.string()),
                    "venue": pa.array([""], type=pa.string()),
                    "journal_name": pa.array([""], type=pa.string()),
                }
            ),
            "paper_id",
        ),
        "paper_authors": (
            pa.table(
                {
                    "paper_id": pa.array(["p1"], type=pa.string()),
                    "position": pa.array([0], type=pa.int64()),
                    "author_name": pa.array(["Ada Lovelace"], type=pa.string()),
                }
            ),
            "paper_id",
        ),
        "specter": (
            pa.table(
                {
                    "paper_id": pa.array(["p1"], type=pa.string()),
                    "embedding": pa.FixedSizeListArray.from_arrays(
                        pa.array([1.0, 0.0], type=pa.float32()),
                        2,
                    ),
                }
            ),
            "paper_id",
        ),
    }
    paths: dict[str, str] = {}
    index_keys = {
        "signatures": "signatures_batch_index",
        "papers": "papers_batch_index",
        "paper_authors": "paper_authors_batch_index",
        "specter": "specter_batch_index",
    }
    for table_name, (table, key_column) in table_specs.items():
        filename = "specter2.arrow" if table_name == "specter" else f"{table_name}.arrow"
        table_path = dataset_root / filename
        paths[table_name] = write_arrow_ipc_table(table, table_path)
        index_key = index_keys[table_name]
        index_path = dataset_root / f"{table_name}.{index_key}.bin"
        paths[index_key], _metrics = write_arrow_batch_lookup_index(
            table_path,
            index_path,
            key_column=key_column,
            table_name=table_name,
        )
    paths["name_counts_index"] = str(root / "name_counts_index")
    dataset_manifest = build_arrow_artifact_manifest(paths, dataset_root)
    manifest_path = write_arrow_artifact_manifest(dataset_manifest, dataset_root)
    relative_manifest = manifest_path.relative_to(root).as_posix()
    root_manifest = {
        "kind": ARROW_COLLECTION_KIND,
        "format_version": PUBLIC_DATA_FORMAT_VERSION,
        "dataset_manifests": {
            dataset: {
                "path": relative_manifest,
                "sha256": sha256_file(manifest_path),
            }
        },
    }
    (root / "manifest.json").write_text(json.dumps(root_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_linker_replay_inputs(root: Path, *, leak: bool, dataset: str = "replay_demo") -> None:
    """Write featureless Parquet label tables, candidate members, split assignments, and ``bundle.json``."""
    import pandas as pd

    label_paths = {
        "train_path": "labels/train.parquet",
        "classic_gate_source_path": "labels/calibration.parquet",
        "s2and_eval_path": "labels/s2and_eval.parquet",
        "hwang_eval_path": "labels/hwang_eval.parquet",
    }
    row_ids = {
        "train_path": ("train:q1", "train:b1"),
        "classic_gate_source_path": ("cal:q1", "cal:b1"),
        "s2and_eval_path": ("s2and:q1", "s2and:b1"),
        "hwang_eval_path": ("hwang:q1", "cal:b1" if leak else "hwang:b1"),
    }
    for key, relative_path in label_paths.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        query_group_id, base_group_id = row_ids[key]
        pd.DataFrame(
            {
                "query_group_id": [query_group_id],
                "base_group_id": [base_group_id],
                "dataset": [dataset],
                "query_view": ["full"],
                "candidate_component_key": ["component-1"],
                "retrieval_rank": [1],
                "label": [1],
            }
        ).to_parquet(path, index=False)

    candidate_path = root / "components" / f"{dataset}_members.parquet"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "candidate_component_key": ["component-1"],
            "member_index": [0],
            "signature_id": ["s1"],
        }
    ).to_parquet(candidate_path, index=False)

    assignments = pd.DataFrame(
        {
            "query_group_id": ["cal:q1", "s2and:q1", "hwang:q1"],
            "source_key": ["s2and_eval", "s2and_eval", "hwang_eval"],
            "split": ["calibration_fit", "calibration_fit", "test"],
            "base_group_id": ["cal:b1", "s2and:b1", "cal:b1" if leak else "hwang:b1"],
        }
    )
    assignments_path = root / "splits" / "assignments.csv"
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(assignments_path, index=False)
    (root / "splits" / "summary.json").write_text('{\n  "row_count": 3\n}\n', encoding="utf-8")
    pd.DataFrame({"base_group_id": ["cal:b1"]}).to_csv(root / "splits" / "internal_eval.csv", index=False)

    bundle = {
        "schema": "s2and_linker_replay_arrow_bundle_v1",
        "bundle_name": "mini-linker-source",
        "assets": {
            "candidate_members": {
                "root": "components",
                "datasets": {dataset: f"components/{dataset}_members.parquet"},
            },
            "featureless_rows": {"root": "labels", "files": label_paths},
            "splits": {
                "root": "splits",
                "assignments_path": "splits/assignments.csv",
                "summary_path": "splits/summary.json",
            },
        },
        "models": {
            "classic": {
                "classic_gate_source_path": label_paths["classic_gate_source_path"],
                "s2and_eval_path": label_paths["s2and_eval_path"],
                "hwang_eval_path": label_paths["hwang_eval_path"],
                "classic_gate_internal_eval_base_groups_path": "splits/internal_eval.csv",
                "stratified_eval_test_split": {
                    "assignments_path": "splits/assignments.csv",
                    "summary_path": "splits/summary.json",
                    "calibration_fit_split": "calibration_fit",
                    "test_split": "test",
                },
                "promoted_stratified_gate": {
                    "calibration_splits": ["calibration_fit"],
                    "test_split": "test",
                },
            }
        },
        "expected_metrics": {},
    }
    (root / "bundle.json").write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_singleton_orcid_views(
    source_root: Path, *, view_count: int, positive_view_count: int = 0, candidates_per_view: int = 1
) -> None:
    """Populate ``labels/s2and_eval.parquet`` with synthetic singleton-ORCID review views."""
    import pandas as pd

    path = source_root / "labels" / "s2and_eval.parquet"
    template = pd.read_parquet(path).assign(supervision_type="negative_singleton_orcid")
    rows = pd.concat(
        [
            template,
            *(
                template.assign(query_group_id=f"singleton:{index}", base_group_id=f"singleton:{index}")
                for index in range(1, view_count)
            ),
        ],
        ignore_index=True,
    )
    rows["label"] = [1] * positive_view_count + [0] * (view_count - positive_view_count - 1) + [None]
    if candidates_per_view > 1:
        rows = pd.concat(
            [
                rows,
                *(
                    rows.assign(candidate_component_key=f"candidate-{rank}", retrieval_rank=rank, label=0)
                    for rank in range(2, candidates_per_view + 1)
                ),
            ],
            ignore_index=True,
        )
    rows.to_parquet(path, index=False)


def link_directory_or_junction(path: Path, target: Path) -> None:
    """Create a Windows directory junction (privilege-free) or POSIX directory symlink."""
    import os

    import pytest

    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target), str(path))
    else:
        try:
            path.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"Directory symlink is unavailable: {exc}")
