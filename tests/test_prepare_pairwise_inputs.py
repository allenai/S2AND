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

from s2and._sha256 import sha256_file
from scripts.production.data._pairwise_augmentation import (
    build_augmentation,
    finalize_augmentation,
    make_variant,
    vector_action,
)
from scripts.production.data.prepare_pairwise_inputs import (
    ModelExclusions,
    dataset_for_split,
    load_model_exclusions,
    main,
    members,
    prepare_benchmark,
    read,
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
                "signature_id": sid,
                "paper_id": pid,
                "author_info": {
                    "first": "Jae-Hoon",
                    "middle": "M",
                    "last": "van der Lee",
                    "suffix": None,
                    "position": position,
                    "block": f"b{block}",
                    "affiliations": ["Institute"],
                    "emails": [],
                },
            }
        for group in range(2):
            cid = f"c{block}-{group}"
            clusters[cid] = {"signature_ids": [str(block * 4 + group * 2 + index) for index in range(2)]}
        papers[pid] = {
            "paper_id": pid,
            "title": "Current title",
            "abstract": "",
            "references": ["ref"],
            "venue": "Venue",
            "journal_name": "Journal",
            "year": 2020,
            "authors": [{"author_name": "Author", "position": index} for index in range(4)],
        }
    for root in (source, legacy):
        directory = root / "qian"
        write(directory / "qian_signatures.json", signatures)
        write(directory / "qian_clusters.json", clusters)
        write(directory / "qian_papers.json", papers)
        write(directory / "qian_added_signatures.json", [])
        write(directory / "restored_unknown_signature_ids_test.json", ["0"])
        with (directory / "qian_specter2.pkl").open("wb") as stream:
            pickle.dump((np.ones((40, 768), dtype=np.float32), np.asarray(list(papers))), stream, protocol=4)
    write(legacy / "augmented" / "augmented_papers.json", {})
    return source, legacy, output


def prepare(inputs: tuple[Path, Path, Path], **options: Any) -> dict:
    source, legacy, output = inputs
    with threadpool_limits(limits=1):
        return prepare_benchmark(
            "qian",
            source_root=source,
            fixed_pair_root=legacy,
            output_root=output,
            unknown_filename="restored_unknown_signature_ids_test.json",
            evaluation_pairs=10,
            **options,
        )


def augment(inputs: tuple[Path, Path, Path], destination: Path, **options: Any) -> dict:
    pytest.importorskip("ijson")
    _, legacy, prepared = inputs
    with threadpool_limits(limits=1):
        return build_augmentation(
            prepared_root=prepared,
            legacy_root=legacy,
            output=destination,
            datasets=("qian",),
            unknown_filename="restored_unknown_signature_ids_test.json",
            train_candidates=100,
            train_pool_cap=100,
            evaluation_candidates=10,
            train_positive_cap=5,
            evaluation_positive_cap=2,
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
    assert set(populations) == {"train", "validation", "unused_internal_test", "external_test"}
    assert "model_exclusions" not in report
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


@pytest.mark.parametrize(
    "filename",
    [
        "unknown.json",
        "../restored_unknown_signature_ids_test.json",
        "nested/restored_unknown_signature_ids_test.json",
        "nested\\restored_unknown_signature_ids_test.json",
        "C:restored_unknown_signature_ids_test.json",
        "restored_unknown_signature_ids_*.json",
        "restored_unknown_signature_ids_.json",
    ],
)
def test_unknown_filename_fails_before_output_creation(inputs: tuple[Path, Path, Path], filename: str) -> None:
    source, legacy, output = inputs
    with pytest.raises(ValueError, match="Unknown-signature filename"):
        prepare_benchmark(
            "qian", source_root=source, fixed_pair_root=legacy, output_root=output, unknown_filename=filename
        )
    assert not output.exists()


@pytest.mark.parametrize("stage", ["benchmarks", "augmentation", "finalize"])
def test_cli_rejects_unknown_filename_before_any_stage(tmp_path: Path, stage: str) -> None:
    output = tmp_path / "output"
    options = {
        "benchmarks": ["--source-root", "absent", "--fixed-pair-root", "absent", "--output", str(output)],
        "augmentation": ["--prepared-root", "absent", "--output", str(output)],
        "finalize": ["--prepared-root", "absent", "--root", str(output)],
    }
    with pytest.raises(SystemExit) as error:
        main([stage, *options[stage], "--unknown-filename", "unknown.json"])
    assert error.value.code == 2
    assert not output.exists()


def test_preparation_rejects_multiple_sidecars_before_output_creation(inputs: tuple[Path, Path, Path]) -> None:
    source, _, output = inputs
    write(source / "qian" / "restored_unknown_signature_ids_other.json", [])
    with pytest.raises(ValueError, match="Multiple"):
        prepare(inputs)
    assert not output.exists()


def test_reserved_scope_stays_out_of_pairs_and_eps(inputs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    source, legacy, baseline = inputs
    signatures_path = source / "qian" / "qian_signatures.json"
    clusters_path = source / "qian" / "qian_clusters.json"
    signatures, clusters = read(signatures_path), read(clusters_path)
    for row in signatures.values():
        row["author_info"]["email"] = None
    with threadpool_limits(limits=1):
        dataset = dataset_for_split("qian", signatures, clusters)
        outer_train, outer_val, outer_test = dataset.split_cluster_signatures()
    selected = [block for block in (*outer_train, *outer_val) if block != "b0"][:5]
    reserved_block, author_block, paper_block, explicit_author_block, explicit_paper_block = selected
    blocks = dataset.get_blocks()
    labels = members(clusters)
    reserved_member = blocks[reserved_block][0]
    author_member = blocks[author_block][0]
    old_cluster = labels[author_member]
    clusters[old_cluster]["signature_ids"].remove(author_member)
    clusters[labels[reserved_member]]["signature_ids"].append(author_member)
    signatures[blocks[paper_block][0]]["paper_id"] = signatures[reserved_member]["paper_id"]
    write(signatures_path, signatures)
    write(clusters_path, clusters)
    originals = {path: path.read_bytes() for path in (signatures_path, clusters_path)}
    prepare(inputs)
    exclusion_path = tmp_path / "exclusions.json"
    write(
        exclusion_path,
        {
            "qian": {
                "reserved_block_ids": [reserved_block, next(iter(outer_test))],
                "reserved_author_ids": [labels[blocks[explicit_author_block][0]]],
                "reserved_query_paper_ids": [str(signatures[blocks[explicit_paper_block][0]]["paper_id"])],
                "rationale": "Reserve fresh incremental query scope.",
                "reference": "fresh_inventory.json",
            }
        },
    )
    output = tmp_path / "reserved"
    assert (
        main(
            [
                "benchmarks",
                "--source-root",
                str(source),
                "--fixed-pair-root",
                str(legacy),
                "--output",
                str(output),
                "--datasets",
                "qian",
                "--unknown-filename",
                "restored_unknown_signature_ids_test.json",
                "--evaluation-pairs",
                "10",
                "--model-exclusions",
                str(exclusion_path),
            ]
        )
        == 0
    )
    summary = read(output / "population_summary.json")
    assert summary["model_exclusions"]["sha256"] == sha256_file(exclusion_path)
    report = summary["datasets"]["qian"]
    populations = read(output / "populations" / "identities" / "qian.json")
    previous = read(baseline / "populations" / "identities" / "qian.json")
    assert populations["external_test"] == previous["external_test"]
    for path, original in originals.items():
        assert path.read_bytes() == original
    for filename in ("qian_signatures.json", "qian_clusters.json", "restored_unknown_signature_ids_test.json"):
        assert (output / "benchmark_source" / "qian" / filename).read_bytes() == (
            baseline / "benchmark_source" / "qian" / filename
        ).read_bytes()
    for path in (baseline / "populations" / "evaluation_source" / "qian").iterdir():
        assert (output / "populations" / "evaluation_source" / "qian" / path.name).read_bytes() == path.read_bytes()
    model_root = output / "populations" / "model_source" / "qian"
    model, model_clusters = read(model_root / "qian_signatures.json"), read(model_root / "qian_clusters.json")
    reserved_ids = {sid for ids in populations["reserved_incremental"].values() for sid in ids}
    assert set(blocks[reserved_block]) <= reserved_ids
    assert author_member in reserved_ids
    assert blocks[paper_block][0] in reserved_ids
    assert not set(blocks[author_block][1:]) & reserved_ids
    assert not set(blocks[paper_block][1:]) & reserved_ids
    assert not model.keys() & reserved_ids
    assert model.keys() | reserved_ids == read(baseline / "populations/model_source/qian/qian_signatures.json").keys()
    assert not model_clusters.keys() & set(report["model_exclusions"]["reserved_author_ids"])
    assert not {str(row["paper_id"]) for row in model.values()} & set(report["model_exclusions"]["reserved_paper_ids"])
    assert report["model_exclusions"]["excluded_development_signatures"] == len(reserved_ids)
    assert report["model_exclusions"]["scope_in_external_test_signatures"] >= len(next(iter(outer_test.values())))
    with threadpool_limits(limits=1):
        model_dataset = dataset_for_split("qian", model, model_clusters)
        train, eps_validation, unused = model_dataset.split_cluster_signatures()
        pairs = model_dataset.pair_sampling(10, [], train)
    for actual, key in zip(
        (train, eps_validation, unused), ("train", "validation", "unused_internal_test"), strict=True
    ):
        assert actual == populations[key]
        assert not reserved_ids & {sid for ids in actual.values() for sid in ids}
    assert not reserved_ids & {sid for left, right, _ in pairs for sid in (left, right)}


@pytest.mark.parametrize(
    ("exclusions", "message"),
    [
        (ModelExclusions(reserved_block_ids=("absent",)), "Reserved blocks"),
        (ModelExclusions(reserved_author_ids=("absent",)), "Reserved authors"),
        (ModelExclusions(reserved_query_paper_ids=("absent",)), "Reserved query papers"),
    ],
)
def test_reserved_scope_rejects_absent_ids(
    inputs: tuple[Path, Path, Path], exclusions: ModelExclusions, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare(inputs, model_exclusions=exclusions)


def test_model_exclusions_reject_bad_inventory(tmp_path: Path) -> None:
    path = tmp_path / "exclusions.json"
    for value in (
        [],
        {"qian": {"reserved_block_ids": "b0"}},
        {"qian": {"reserved_block_ids": ["b0", "b0"]}},
        {"qian": {"reserved_author_ids": [123]}},
        {"qian": {"reserved_block_id": ["b0"]}},
    ):
        write(path, value)
        with pytest.raises(ValueError):
            load_model_exclusions(path)


def test_corrected_source_does_not_require_addition_history(inputs: tuple[Path, Path, Path]) -> None:
    source, _, output = inputs
    (source / "qian" / "qian_added_signatures.json").unlink()
    prepare(inputs)
    assert not (output / "benchmark_source" / "qian" / "qian_added_signatures.json").exists()
    actual = read(output / "benchmark_source" / "qian" / "qian_signatures.json")
    original = read(source / "qian" / "qian_signatures.json")
    assert list(actual) == list(original)
    assert all(actual[sid]["author_info"]["first"] == row["author_info"]["first"] for sid, row in original.items())


def test_augmentation_uses_corrected_source_without_legacy(inputs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    pytest.importorskip("ijson")
    prepare(inputs)
    _, _, prepared = inputs
    destination = tmp_path / "augmented_current"
    with threadpool_limits(limits=1):
        report = build_augmentation(
            prepared_root=prepared,
            output=destination,
            datasets=("qian",),
            unknown_filename="restored_unknown_signature_ids_test.json",
            train_candidates=100,
            train_pool_cap=100,
            evaluation_candidates=10,
            train_positive_cap=5,
            evaluation_positive_cap=2,
        )
    assert report["datasets"]["qian"]
    finalize_augmentation(
        root=destination,
        prepared_root=prepared,
        datasets=("qian",),
        unknown_filename="restored_unknown_signature_ids_test.json",
    )


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
    write(
        receipt,
        {
            "datasets": {
                "qian": {
                    "before_sha256": sha256_file(original_path),
                    "staged_sha256": sha256_file(stage_path),
                    "unlabeled_ids": ["3"],
                    "rows": 1,
                    "moved": 1,
                }
            }
        },
    )
    prepare(inputs, correction_receipt=receipt, correction_root=tmp_path / "corrections")
    current = members(read(output / "benchmark_source" / "qian" / "qian_clusters.json"))
    assert current["2"] == "c0-0"
    assert current["3"] == "c0-1"
    assert read(output / "benchmark_source" / "qian" / "restored_unknown_signature_ids_test.json") == ["0", "3"]


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
        root=destination,
        prepared_root=prepared,
        datasets=("qian",),
        unknown_filename="restored_unknown_signature_ids_test.json",
    )
    assert result["vector_rows"] == len(signatures)


@pytest.mark.parametrize("mutation", ["heldout", "unknown", "label", "source"])
def test_finalizer_rejects_bad_training_pairs(
    inputs: tuple[Path, Path, Path],
    tmp_path: Path,
    mutation: str,
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
        unknown_path = prepared / "benchmark_source" / "qian" / "restored_unknown_signature_ids_test.json"
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
    finalize_augmentation(
        root=destination,
        prepared_root=prepared,
        datasets=("qian",),
        unknown_filename="restored_unknown_signature_ids_test.json",
    )


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
        root=destination,
        prepared_root=prepared,
        datasets=("qian",),
        unknown_filename="restored_unknown_signature_ids_test.json",
    )
    assert result["vector_rows"] == 0
    assert result["absent_blank_vectors"] == report["signature_count"]


def test_dropped_last_text_does_not_queue_separator_only_vectors(
    inputs: tuple[Path, Path, Path],
    tmp_path: Path,
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
        "paper_id": "p",
        "title": "Title",
        "abstract": "Abstract",
        "references": [],
        "venue": None,
        "journal_name": None,
        "authors": [{"position": 0}, {"position": 1}],
    }
    signatures: list[dict[str, Any]] = [
        {
            "signature_id": str(index),
            "paper_id": "p",
            "author_info": {
                "first": "Jae-Hoon",
                "middle": "M",
                "last": "van der Lee",
                "position": index,
                "affiliations": [],
            },
        }
        for index in range(2)
    ]
    variants = [
        make_variant(
            "qian",
            str(index),
            signature,
            paper,
            old_paper=paper,
            cached_paper=None,
            rng=random.Random(1111),
            omission_rate=1,
        )
        for index, signature in enumerate(signatures)
    ]
    assert variants[0][0]["paper_id"] != variants[1][0]["paper_id"]
    assert variants[0][1]["authors"] == [{"position": 0}]
    assert variants[1][1]["authors"] == [{"position": 1}]
    assert len(paper["authors"]) == 2
    assert all(signature["author_info"]["first"] == "Jae-Hoon" for signature in signatures)
    full, _, _ = make_variant(
        "qian",
        "0",
        signatures[0],
        paper,
        old_paper=paper,
        cached_paper=None,
        rng=random.Random(1111),
        omission_rate=0,
    )
    assert full["author_info"]["first"] == "Jae-Hoon"
    assert full["author_info"]["middle"] == "M"


@pytest.mark.parametrize(
    ("title", "abstract", "has_vector", "expected"),
    [
        ("", "", False, "absent"),
        (" ", "", True, "absent"),
        ("Title", "Abstract", False, "refresh"),
        ("Title", "Abstract", True, "reuse"),
        ("Title", "", True, "refresh"),
    ],
)
def test_vector_policy_without_stream_dependencies(title: str, abstract: str, has_vector: bool, expected: str) -> None:
    assert (
        vector_action(
            {"title": title, "abstract": abstract},
            {"title": "Title", "abstract": "Abstract"},
            has_source_vector=has_vector,
        )
        == expected
    )


@pytest.mark.parametrize("cache_title", ["French translation of version 3.0", "unicode standard version"])
def test_translation_cache_preserves_version_numbers(cache_title: str) -> None:
    original = {
        "paper_id": "p",
        "title": "Unicode Standard Version 8.0",
        "abstract": "",
        "references": [],
        "venue": None,
        "journal_name": None,
        "authors": [],
    }
    signature = {
        "signature_id": "s",
        "paper_id": "p",
        "author_info": {
            "first": "Jae-Hoon",
            "middle": None,
            "last": "Lee",
            "position": 0,
            "affiliations": [],
        },
    }
    old = {**original, "title": "Unicode Standard Version 3.0"}
    _, variant, counts = make_variant(
        "qian",
        "s",
        signature,
        original,
        old_paper=old,
        cached_paper={"title": cache_title},
        rng=random.Random(1111),
        omission_rate=0,
    )
    assert variant["title"] == original["title"]
    assert not counts["cached_translation_applied"]
    if cache_title.startswith("French"):
        assert counts["cached_translation_source_change"] == 1
    else:
        assert not counts["cached_translation_source_change"]


@pytest.mark.parametrize(
    ("old_title", "current_title"),
    [("Actual Title", "actual title"), ("Formula x+y", "Formula x-y")],
)
def test_translation_cache_requires_exact_source_title(old_title: str, current_title: str) -> None:
    paper = {
        "paper_id": "p",
        "title": current_title,
        "abstract": "",
        "references": [],
        "venue": None,
        "journal_name": None,
        "authors": [],
    }
    signature = {
        "signature_id": "s",
        "paper_id": "p",
        "author_info": {
            "first": "Jae-Hoon",
            "middle": None,
            "last": "Lee",
            "position": 0,
            "affiliations": [],
        },
    }
    _, variant, counts = make_variant(
        "qian",
        "s",
        signature,
        paper,
        old_paper={**paper, "title": old_title},
        cached_paper={"title": "French translation"},
        rng=random.Random(1111),
        omission_rate=0,
    )
    assert variant["title"] == current_title
    assert counts["cached_translation_source_change"] == 1
    assert not counts["cached_translation_applied"]
