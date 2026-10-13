"""Exercise accepted names, corrected gold, and augmented split boundaries."""

from __future__ import annotations

import copy
import csv
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from scripts.production.data._pairwise_augmentation import (
    build_augmentation,
    finalize_augmentation,
    make_variant,
    vector_action,
)
from scripts.production.data.prepare_pairwise_inputs import (
    dataset_for_split,
    members,
    prepare_benchmark,
    read,
    sha,
    write,
)


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    legacy = tmp_path / "legacy"
    output = tmp_path / "prepared"
    signatures: dict[str, Any] = {}
    clusters: dict[str, Any] = {}
    papers: dict[str, Any] = {}
    for block in range(40):
        pid = str(block)
        for position in (2, 0, 3, 1):
            sid = str(block * 4 + position)
            signatures[sid] = {
                "signature_id": sid, "paper_id": pid,
                "author_info": {
                    "first": "Jae-Hoon", "middle": "M", "last": "van der Lee", "suffix": None,
                    "position": position, "block": f"b{block}", "affiliations": ["Institute"], "emails": [],
                },
            }
        for group in range(2):
            cid = f"c{block}-{group}"
            clusters[cid] = {"signature_ids": [str(block * 4 + group * 2 + index) for index in range(2)]}
        papers[pid] = {
            "paper_id": pid, "title": "Current title", "abstract": "", "references": ["ref"],
            "venue": "Venue", "journal_name": "Journal", "year": 2020,
            "authors": [{"author_name": "Author", "position": index} for index in range(4)],
        }
    for root in (source, legacy):
        directory = root / "qian"
        write(directory / "qian_signatures.json", signatures)
        write(directory / "qian_clusters.json", clusters)
        write(directory / "qian_papers.json", papers)
        write(directory / "qian_added_signatures.json", [])
        write(directory / "unknown.json", ["0"])
        with (directory / "qian_specter2.pkl").open("wb") as stream:
            pickle.dump((np.ones((40, 768), dtype=np.float32), np.asarray(list(papers))), stream, protocol=4)
    write(legacy / "augmented" / "augmented_papers.json", {})
    return source, legacy, output


def prepare(inputs: tuple[Path, Path, Path], **options: Any) -> dict:
    source, legacy, output = inputs
    with threadpool_limits(limits=1):
        return prepare_benchmark(
            "qian", source_root=source, fixed_pair_root=legacy, output_root=output,
            unknown_filename="unknown.json", evaluation_pairs=10, **options,
        )


def augment(inputs: tuple[Path, Path, Path], destination: Path, **options: Any) -> dict:
    pytest.importorskip("ijson")
    _, legacy, prepared = inputs
    with threadpool_limits(limits=1):
        return build_augmentation(
            prepared_root=prepared, legacy_root=legacy, output=destination, datasets=("qian",),
            unknown_filename="unknown.json", train_candidates=100, train_pool_cap=100,
            evaluation_candidates=10, train_positive_cap=5, evaluation_positive_cap=2,
            **options,
        )


def test_names_order_and_nested_real_split(inputs: tuple[Path, Path, Path]) -> None:
    report = prepare(inputs, seed=1234)
    source, _, output = inputs
    original = read(source / "qian" / "qian_signatures.json")
    prepared = read(output / "benchmark_source" / "qian" / "qian_signatures.json")
    assert list(prepared) == list(original)
    assert prepared["2"]["author_info"]["first"] == "Jae-Hoon"
    assert prepared["2"]["author_info"]["middle"] == "M"
    assert report["canonical_names"]["changed_signatures"] == 0
    populations = read(output / "populations" / "identities" / "qian.json")
    model = read(output / "populations" / "model_source" / "qian" / "qian_signatures.json")
    clusters = read(output / "populations" / "model_source" / "qian" / "qian_clusters.json")
    with threadpool_limits(limits=1):
        inner = dataset_for_split("qian", model, clusters, seed=1234).split_cluster_signatures()
    for actual, key in zip(inner, ("train", "validation", "unused_internal_test"), strict=True):
        assert actual == populations[key]
    external = {sid for block in populations["external_test"].values() for sid in block}
    assert not external & model.keys()
    assert external | model.keys() == original.keys()
    assert list(model) == [sid for sid in original if sid not in external]


def test_correction_restores_unknown_and_keeps_known_move(inputs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    source, _, output = inputs
    original_path = source / "qian" / "qian_clusters.json"
    staged = copy.deepcopy(read(original_path))
    staged["c0-1"]["signature_ids"].remove("2")
    staged["c0-0"]["signature_ids"].append("2")
    staged["c0-1"]["signature_ids"].remove("3")
    stage_path = tmp_path / "corrections" / "qian" / "qian_clusters.json"
    write(stage_path, staged)
    receipt = tmp_path / "receipt.json"
    write(receipt, {"datasets": {"qian": {
        "before_sha256": sha(original_path), "staged_sha256": sha(stage_path),
        "unlabeled_ids": ["3"], "rows": 1, "moved": 1,
    }}})
    prepare(inputs, correction_receipt=receipt, correction_root=tmp_path / "corrections")
    current = members(read(output / "benchmark_source" / "qian" / "qian_clusters.json"))
    assert current["2"] == "c0-0"
    assert current["3"] == "c0-1"
    assert read(output / "benchmark_source" / "qian" / "unknown.json") == ["0", "3"]


def test_variant_ownership_and_real_pair_guard(inputs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    prepare(inputs)
    destination = tmp_path / "augmented"
    report = augment(inputs, destination, omission_rate=0)
    assert report["unique_embedding_texts_to_refresh"] == 0
    signatures = read(destination / "augmented_signatures.json")
    papers = read(destination / "augmented_papers.json")
    owners = {row["paper_id"] for row in signatures.values()}
    assert len(owners) == len(signatures) == len(papers)
    assert owners == papers.keys()
    assert all(len(paper["authors"]) == 4 for paper in papers.values())
    assert all(row["author_info"]["first"] == "Jae-Hoon" for row in signatures.values())
    _, _, prepared = inputs
    result = finalize_augmentation(
        root=destination, prepared_root=prepared, datasets=("qian",), unknown_filename="unknown.json",
    )
    assert result["vector_rows"] == len(signatures)


@pytest.mark.parametrize("mutation", ["heldout", "unknown", "label", "source"])
def test_finalizer_rejects_bad_training_pairs(
    inputs: tuple[Path, Path, Path], tmp_path: Path, mutation: str,
) -> None:
    prepare(inputs)
    destination = tmp_path / "augmented"
    augment(inputs, destination, omission_rate=0)
    _, _, prepared = inputs
    pair_path = destination / "train_pairs.csv"
    with pair_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if mutation == "heldout":
        test_path = destination / "test_pairs.csv"
        with test_path.open(newline="") as stream:
            test_rows = list(csv.DictReader(stream))
        rows[0] = test_rows.pop(0)
        with test_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("signature_id_1", "signature_id_2", "label"))
            writer.writeheader()
            writer.writerows(test_rows)
        with pytest.raises(ValueError, match="Signature leakage|Population leakage"):
            _replace_and_finalize(rows, pair_path, destination, prepared)
    elif mutation == "unknown":
        unknown_path = prepared / "benchmark_source" / "qian" / "unknown.json"
        write(unknown_path, [rows[0]["signature_id_1"].partition("___")[2]])
        with pytest.raises(ValueError, match="Unknown augmentation member"):
            _replace_and_finalize(rows, pair_path, destination, prepared)
    elif mutation == "source":
        population_path = prepared / "populations" / "identities" / "qian.json"
        populations = read(population_path)
        missing = rows[0]["signature_id_1"].partition("___")[2]
        populations["train"] = {
            block: [sid for sid in ids if sid != missing] for block, ids in populations["train"].items()
        }
        write(population_path, populations)
        with pytest.raises(ValueError, match="Population leakage"):
            _replace_and_finalize(rows, pair_path, destination, prepared)
    else:
        rows[0]["label"] = str(1 - int(rows[0]["label"]))
        with pytest.raises(ValueError, match="Stale augmentation label"):
            _replace_and_finalize(rows, pair_path, destination, prepared)


def _replace_and_finalize(rows: list[dict], pair_path: Path, destination: Path, prepared: Path) -> None:
    with pair_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("signature_id_1", "signature_id_2", "label"))
        writer.writeheader()
        writer.writerows(rows)
    finalize_augmentation(root=destination, prepared_root=prepared, datasets=("qian",), unknown_filename="unknown.json")


def test_missing_blank_vectors_remain_absent(inputs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    source, _, prepared = inputs
    paper_path = source / "qian" / "qian_papers.json"
    papers = read(paper_path)
    for paper in papers.values():
        paper["title"] = ""
    write(paper_path, papers)
    with (source / "qian" / "qian_specter2.pkl").open("wb") as stream:
        pickle.dump((np.empty((0, 768), dtype=np.float32), np.asarray([], dtype="U1")), stream, protocol=4)
    prepare(inputs)
    destination = tmp_path / "augmented"
    report = augment(inputs, destination, omission_rate=0)
    assert report["unique_embedding_texts_to_refresh"] == 0
    assert read(destination / "vector_ids.json") == []
    result = finalize_augmentation(
        root=destination, prepared_root=prepared, datasets=("qian",), unknown_filename="unknown.json",
    )
    assert result["vector_rows"] == 0
    assert result["absent_blank_vectors"] == report["signature_count"]


def test_dropped_last_text_does_not_queue_separator_only_vectors(
    inputs: tuple[Path, Path, Path], tmp_path: Path,
) -> None:
    source, _, _ = inputs
    paper_path = source / "qian" / "qian_papers.json"
    papers = read(paper_path)
    for paper in papers.values():
        paper.update(title="", abstract="Only source text")
    write(paper_path, papers)
    prepare(inputs)
    destination = tmp_path / "augmented"
    report = augment(inputs, destination, omission_rate=1)
    assert report["unique_embedding_texts_to_refresh"] == 0
    assert read(destination / "vector_ids.json") == []
    assert report["datasets"]["qian"]["corruptions"]["absent_blank_vectors"] == report["signature_count"]


def test_variants_keep_separate_author_rows_and_raw_compound_names() -> None:
    paper = {
        "paper_id": "p", "title": "Title", "abstract": "Abstract", "references": [],
        "venue": None, "journal_name": None,
        "authors": [{"position": 0}, {"position": 1}],
    }
    signatures: list[dict[str, Any]] = [
        {"signature_id": str(index), "paper_id": "p", "author_info": {
            "first": "Jae-Hoon", "middle": "M", "last": "van der Lee", "position": index, "affiliations": [],
        }}
        for index in range(2)
    ]
    variants = [
        make_variant("qian", str(index), signature, paper, old_paper=paper, cached_paper=None,
                     rng=random.Random(1111), omission_rate=1)
        for index, signature in enumerate(signatures)
    ]
    assert variants[0][0]["paper_id"] != variants[1][0]["paper_id"]
    assert variants[0][1]["authors"] == [{"position": 0}]
    assert variants[1][1]["authors"] == [{"position": 1}]
    assert len(paper["authors"]) == 2
    assert all(signature["author_info"]["first"] == "Jae-Hoon" for signature in signatures)
    full, _, _ = make_variant(
        "qian", "0", signatures[0], paper, old_paper=paper, cached_paper=None,
        rng=random.Random(1111), omission_rate=0,
    )
    assert full["author_info"]["first"] == "Jae-Hoon"
    assert full["author_info"]["middle"] == "M"


@pytest.mark.parametrize(
    ("title", "abstract", "has_vector", "expected"),
    [("", "", False, "absent"), (" ", "", True, "absent"),
     ("Title", "Abstract", False, "refresh"), ("Title", "Abstract", True, "reuse"),
     ("Title", "", True, "refresh")],
)
def test_vector_policy_without_stream_dependencies(title: str, abstract: str, has_vector: bool, expected: str) -> None:
    assert vector_action(
        {"title": title, "abstract": abstract}, {"title": "Title", "abstract": "Abstract"},
        has_source_vector=has_vector,
    ) == expected
