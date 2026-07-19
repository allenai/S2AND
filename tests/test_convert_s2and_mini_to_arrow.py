from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scripts.convert_to_arrow as convert_to_arrow
from s2and.consts import NORMALIZATION_VERSION
from s2and.incremental_linking.feature_block import write_arrow_ipc_table
from scripts.convert_to_arrow import RuntimeDatasetSources


def _fake_sources(tmp_path: Path, dataset: str) -> RuntimeDatasetSources:
    source_dir = tmp_path / "source" / dataset
    return RuntimeDatasetSources(
        dataset=dataset,
        source_dir=source_dir,
        signatures_path=source_dir / f"{dataset}_signatures.json",
        papers_path=source_dir / f"{dataset}_papers.json",
    )


def _write_signatures_table(pa: Any, path: Path, signature_ids: list[str | None], paper_ids: list[str]) -> None:
    row_count = len(signature_ids)
    write_arrow_ipc_table(
        pa.table(
            {
                "signature_id": pa.array(signature_ids, type=pa.string()),
                "paper_id": pa.array(paper_ids, type=pa.string()),
                "author_first": pa.array(["Ada"] * row_count, type=pa.string()),
                "author_middle": pa.array([""] * row_count, type=pa.string()),
                "author_last": pa.array(["Lovelace"] * row_count, type=pa.string()),
                "author_suffix": pa.array([""] * row_count, type=pa.string()),
                "author_affiliations": pa.array([["Analytical Engine"]] * row_count, type=pa.list_(pa.string())),
                "author_orcid": pa.array([""] * row_count, type=pa.string()),
                "author_position": pa.array(list(range(row_count)), type=pa.int64()),
            }
        ),
        path,
    )


def _write_papers_table(pa: Any, path: Path, paper_ids: list[str]) -> None:
    row_count = len(paper_ids)
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(paper_ids, type=pa.string()),
                "title": pa.array(["Notes"] * row_count, type=pa.string()),
                "venue": pa.array(["Proceedings"] * row_count, type=pa.string()),
                "journal_name": pa.array(["Journal"] * row_count, type=pa.string()),
            }
        ),
        path,
    )


def _write_paper_authors_table(pa: Any, path: Path, paper_ids: list[str], author_names: list[str]) -> None:
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(paper_ids, type=pa.string()),
                "position": pa.array(list(range(len(paper_ids))), type=pa.int64()),
                "author_name": pa.array(author_names, type=pa.string()),
            }
        ),
        path,
    )


def test_arrow_artifact_generation_excludes_request_local_sidecars(tmp_path: Path) -> None:
    signatures_path = tmp_path / "signatures.arrow"
    signatures_path.write_bytes(b"signatures")
    sidecar_paths = {
        "query_signatures": tmp_path / "query-signatures.arrow",
        "cluster_seeds": tmp_path / "cluster-seeds.arrow",
        "cluster_seed_disallows": tmp_path / "cluster-seed-disallows.arrow",
        "altered_cluster_signatures": tmp_path / "altered-cluster-signatures.arrow",
    }
    for key, path in sidecar_paths.items():
        path.write_bytes(key.encode("utf-8"))
    paths = {
        "signatures": str(signatures_path),
        **{key: str(path) for key, path in sidecar_paths.items()},
    }

    first = convert_to_arrow._arrow_artifact_generation(paths, tmp_path)
    for path in sidecar_paths.values():
        path.write_bytes(b"request-local replacement")
    second = convert_to_arrow._arrow_artifact_generation(paths, tmp_path)

    assert set(first["files"]) == {"signatures"}
    assert second == first


def test_benchmark_parser_requires_explicit_dataset_selection(tmp_path: Path) -> None:
    parser = convert_to_arrow._build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            [
                "benchmark",
                "--source-root",
                str(tmp_path / "source"),
                "--output-root",
                str(tmp_path / "out"),
            ]
        )

    assert excinfo.value.code == 2


def test_linker_replay_parser_requires_explicit_dataset_selection(tmp_path: Path) -> None:
    parser = convert_to_arrow._build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            [
                "linker-replay",
                "--raw-root",
                str(tmp_path / "raw"),
                "--embeddings-root",
                str(tmp_path / "embeddings"),
                "--output-root",
                str(tmp_path / "out"),
            ]
        )

    assert excinfo.value.code == 2


def test_name_counts_subcommand_is_validation_only(tmp_path: Path) -> None:
    parser = convert_to_arrow._build_parser()

    args = parser.parse_args(["validate-name-counts-index", "--output-root", str(tmp_path)])

    assert args.func is convert_to_arrow._run_validate_name_counts_index
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["name-counts-index", "--output-root", str(tmp_path)])
    assert excinfo.value.code == 2


def test_benchmark_main_run_full_discovers_datasets_only_when_explicit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_convert_runtime_dataset_to_arrow(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"dataset": kwargs["sources"].dataset}

    monkeypatch.setattr(convert_to_arrow, "discover_benchmark_datasets", lambda _source_root: ["first", "second"])
    monkeypatch.setattr(
        convert_to_arrow, "benchmark_dataset_sources", lambda _source_root, dataset: _fake_sources(tmp_path, dataset)
    )
    monkeypatch.setattr(convert_to_arrow, "convert_runtime_dataset_to_arrow", fake_convert_runtime_dataset_to_arrow)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_to_arrow.py",
            "benchmark",
            "--source-root",
            str(tmp_path / "source"),
            "--output-root",
            str(tmp_path / "out"),
            "--run-full",
            "--skip-validation",
        ],
    )

    convert_to_arrow.main()

    assert [call["sources"].dataset for call in calls] == ["first", "second"]


def test_root_manifest_upsert_keeps_dataset_order_stable(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    for dataset_name in ("b", "a"):
        dataset_dir = output_root / dataset_name
        dataset_dir.mkdir()
        (dataset_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "dataset": dataset_name,
                    "conversion_kind": "table-runtime",
                    "source_dir": str(tmp_path / "source" / dataset_name),
                    "signature_count": 2,
                    "paper_count": 3,
                    "paper_embedding_count": 99,
                    "cluster_seeds_require_count": 99,
                    "cluster_seeds_disallow_count": 0,
                    "altered_cluster_signature_count": 0,
                    "paths": {
                        "signatures": "signatures.arrow",
                        "papers": "papers.arrow",
                        "paper_authors": "paper_authors.arrow",
                        "specter": "specter.arrow",
                        "cluster_seeds": "cluster_seeds.arrow",
                        "name_counts_index": "../name_counts_index",
                        "signatures_batch_index": "signatures.signatures_batch_index.bin",
                    },
                    "validation": {
                        "specter_count": 2,
                        "missing_specter_paper_count": 1,
                        "cluster_seed_count": 1,
                        "cluster_seed_disallow_count": 0,
                        "altered_cluster_signature_count": 0,
                    },
                }
            ),
            encoding="utf-8",
        )
        convert_to_arrow._upsert_root_manifest(output_root, dataset_name=dataset_name, dataset_dir=dataset_dir)

    dataset_dir = output_root / "a"
    convert_to_arrow._upsert_root_manifest(output_root, dataset_name="a", dataset_dir=dataset_dir)

    root_manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert root_manifest["datasets"] == ["a", "b"]
    assert root_manifest["output_root"] == str(output_root)
    assert [entry["dataset"] for entry in root_manifest["dataset_manifests"]] == ["a", "b"]
    assert root_manifest["generator"]["script"] == "scripts/convert_to_arrow.py"
    assert isinstance(root_manifest["generated_at_utc"], str)
    assert root_manifest["audit"] == {
        "dataset_count": 2,
        "datasets_with_missing_manifests": [],
        "total_signature_count": 4,
        "total_paper_count": 6,
        "total_embedding_row_count": 4,
        "total_missing_embedding_count": 2,
        "total_batch_index_count": 2,
    }
    assert root_manifest["validation_command_cwd"] == str(convert_to_arrow._PROJECT_ROOT)
    first_entry = root_manifest["dataset_manifests"][0]
    assert first_entry["manifest_exists"] is True
    assert first_entry["manifest_size_bytes"] > 0
    assert len(first_entry["manifest_sha256"]) == 64
    assert str(first_entry["audit"]["source_id"]).replace("\\", "/").endswith("source/a")
    assert first_entry["audit"]["embedding_row_count"] == 2
    assert first_entry["audit"]["cluster_seed_count"] == 1
    assert first_entry["audit"]["sidecar_keys"] == [
        "cluster_seeds",
        "name_counts_index",
        "signatures_batch_index",
    ]
    validation_command = (
        "uv run python scripts/convert_to_arrow.py validate --dataset-dir "
        "{dataset_dir} --require-embeddings --require-name-counts-index"
    )
    dataset_dir_a = convert_to_arrow._manifest_relative_path(output_root / "a", convert_to_arrow._PROJECT_ROOT).replace(
        "\\", "/"
    )
    dataset_dir_b = convert_to_arrow._manifest_relative_path(output_root / "b", convert_to_arrow._PROJECT_ROOT).replace(
        "\\", "/"
    )
    assert root_manifest["validation_commands"] == [
        validation_command.format(dataset_dir=dataset_dir_a),
        validation_command.format(dataset_dir=dataset_dir_b),
    ]


def test_root_manifest_upsert_preserves_existing_output_root_label(tmp_path: Path) -> None:
    output_root = tmp_path / "release"
    output_root.mkdir()
    dataset_dir = output_root / "qian"
    dataset_dir.mkdir()
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "qian",
                "conversion_kind": "table-runtime",
                "signature_count": 7,
                "paper_count": 5,
                "paths": {
                    "signatures": "signatures.arrow",
                    "papers": "papers.arrow",
                    "paper_authors": "paper_authors.arrow",
                },
            }
        ),
        encoding="utf-8",
    )
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": convert_to_arrow.ROOT_MANIFEST_SCHEMA,
                "output_root": "s3://ai2-s2-research-public/s2and-release-arrow",
                "datasets": [],
                "dataset_manifests": [],
            }
        ),
        encoding="utf-8",
    )

    convert_to_arrow._upsert_root_manifest(output_root, dataset_name="qian", dataset_dir=dataset_dir)

    root_manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert root_manifest["output_root"] == "s3://ai2-s2-research-public/s2and-release-arrow"


def test_refresh_root_manifest_enriches_release_and_replay_entries(tmp_path: Path) -> None:
    release_root = tmp_path / "release"
    release_root.mkdir()
    dataset_dir = release_root / "qian"
    dataset_dir.mkdir()
    dataset_manifest = {
        "dataset": "qian",
        "conversion_kind": "table-runtime",
        "signature_count": 7,
        "paper_count": 5,
        "paths": {
            "signatures": "signatures.arrow",
            "papers": "papers.arrow",
            "paper_authors": "paper_authors.arrow",
            "specter2": "specter2.arrow",
            "name_counts_index": "../name_counts_index",
            "signatures_batch_index": "signatures.signatures_batch_index.bin",
        },
        "validation": {"specter_count": 5, "missing_specter_paper_count": 0},
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(dataset_manifest), encoding="utf-8")

    replay_root = release_root / "s2and_and_big_blocks_linker_dataset_20260525"
    replay_dataset_dir = replay_root / "datasets" / "pubmed"
    replay_dataset_dir.mkdir(parents=True)
    replay_dataset_manifest = {
        "dataset": "pubmed",
        "conversion_kind": "table-runtime",
        "signature_count": 11,
        "paper_count": 9,
        "paths": {
            "signatures": "signatures.arrow",
            "papers": "papers.arrow",
            "paper_authors": "paper_authors.arrow",
            "specter2": "specter2.arrow",
            "name_counts_index": "../../../name_counts_index",
        },
        "validation": {"specter_count": 8, "missing_specter_paper_count": 1},
    }
    (replay_dataset_dir / "manifest.json").write_text(json.dumps(replay_dataset_manifest), encoding="utf-8")
    (replay_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "inference_arrow_bundle_v1",
                "datasets": ["pubmed"],
                "dataset_manifests": [
                    {
                        "dataset": "pubmed",
                        "dataset_dir": "datasets/pubmed",
                        "manifest_path": "datasets/pubmed/manifest.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (release_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "inference_arrow_bundle_v1",
                "output_root": "s3://ai2-s2-research-public/s2and-release-arrow",
                "datasets": ["qian"],
                "dataset_manifests": [
                    {
                        "dataset": "qian",
                        "dataset_dir": "qian",
                        "manifest_path": "qian/manifest.json",
                    }
                ],
                "replay_bundles": [
                    {
                        "bundle": "s2and_and_big_blocks_linker_dataset_20260525",
                        "manifest_path": "s2and_and_big_blocks_linker_dataset_20260525/manifest.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    args = convert_to_arrow._build_parser().parse_args(
        [
            "refresh-root-manifest",
            "--output-root",
            str(release_root),
        ]
    )
    args.func(args)

    root_manifest = json.loads((release_root / "manifest.json").read_text(encoding="utf-8"))
    assert root_manifest["output_root"] == "s3://ai2-s2-research-public/s2and-release-arrow"
    assert len(root_manifest["dataset_manifests"][0]["manifest_sha256"]) == 64
    assert root_manifest["audit"]["total_signature_count"] == 7
    replay_entry = root_manifest["replay_bundles"][0]
    assert len(replay_entry["manifest_sha256"]) == 64
    assert replay_entry["audit"]["total_signature_count"] == 11
    assert root_manifest["replay_audit"] == {
        "bundle_count": 1,
        "bundles_with_missing_manifests": [],
        "total_dataset_count": 1,
    }
    qian_dir = convert_to_arrow._manifest_relative_path(release_root / "qian", convert_to_arrow._PROJECT_ROOT).replace(
        "\\", "/"
    )
    replay_pubmed_dir = convert_to_arrow._manifest_relative_path(
        release_root / "s2and_and_big_blocks_linker_dataset_20260525" / "datasets" / "pubmed",
        convert_to_arrow._PROJECT_ROOT,
    ).replace("\\", "/")
    assert root_manifest["validation_commands"] == [
        (
            "uv run python scripts/convert_to_arrow.py validate --dataset-dir "
            f"{qian_dir} "
            "--require-embeddings --require-name-counts-index"
        ),
        (
            "uv run python scripts/convert_to_arrow.py validate --dataset-dir "
            f"{replay_pubmed_dir} "
            "--require-embeddings --require-name-counts-index"
        ),
    ]


def test_linker_replay_main_writes_datasets_under_release_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_linker_sources(_raw_root: Path, _embeddings_root: Path, dataset: str) -> RuntimeDatasetSources:
        return RuntimeDatasetSources(
            dataset=dataset,
            source_dir=tmp_path / "raw" / dataset,
            signatures_path=tmp_path / "raw" / dataset / "signatures.json",
            papers_path=tmp_path / "raw" / dataset / "papers.json",
            specter2_path=tmp_path / "embeddings" / dataset / "specter2.pkl",
        )

    def fake_convert_runtime_dataset_to_arrow(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"dataset": kwargs["sources"].dataset}

    monkeypatch.setattr(convert_to_arrow, "linker_replay_dataset_sources", fake_linker_sources)
    monkeypatch.setattr(convert_to_arrow, "convert_runtime_dataset_to_arrow", fake_convert_runtime_dataset_to_arrow)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_to_arrow.py",
            "linker-replay",
            "--raw-root",
            str(tmp_path / "raw"),
            "--embeddings-root",
            str(tmp_path / "embeddings"),
            "--output-root",
            str(tmp_path / "linker_replay_20260513"),
            "--datasets",
            "pubmed",
            "--skip-validation",
        ],
    )

    convert_to_arrow.main()

    assert calls[0]["output_dir"] == tmp_path / "linker_replay_20260513" / "datasets" / "pubmed"
    assert calls[0]["root_manifest_dir"] == tmp_path / "linker_replay_20260513"
    assert calls[0]["selected_embedding"] == "specter2"


def test_linker_replay_main_run_full_discovers_datasets_only_when_explicit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_linker_sources(_raw_root: Path, _embeddings_root: Path, dataset: str) -> RuntimeDatasetSources:
        return RuntimeDatasetSources(
            dataset=dataset,
            source_dir=tmp_path / "raw" / dataset,
            signatures_path=tmp_path / "raw" / dataset / "signatures.json",
            papers_path=tmp_path / "raw" / dataset / "papers.json",
            specter2_path=tmp_path / "embeddings" / dataset / "specter2.pkl",
        )

    def fake_convert_runtime_dataset_to_arrow(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"dataset": kwargs["sources"].dataset}

    monkeypatch.setattr(
        convert_to_arrow, "discover_linker_replay_datasets", lambda _raw_root, _embeddings_root: ["pubmed"]
    )
    monkeypatch.setattr(convert_to_arrow, "linker_replay_dataset_sources", fake_linker_sources)
    monkeypatch.setattr(convert_to_arrow, "convert_runtime_dataset_to_arrow", fake_convert_runtime_dataset_to_arrow)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_to_arrow.py",
            "linker-replay",
            "--raw-root",
            str(tmp_path / "raw"),
            "--embeddings-root",
            str(tmp_path / "embeddings"),
            "--output-root",
            str(tmp_path / "linker_replay_20260513"),
            "--run-full",
            "--skip-validation",
        ],
    )

    convert_to_arrow.main()

    assert [call["sources"].dataset for call in calls] == ["pubmed"]


def test_validate_manifest_require_embeddings_reports_missing_specter_rows(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")

    signatures_path = tmp_path / "signatures.arrow"
    papers_path = tmp_path / "papers.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    specter_path = tmp_path / "specter.arrow"
    _write_signatures_table(pa, signatures_path, ["s1", "s2"], ["p1", "p2"])
    _write_papers_table(pa, papers_path, ["p1", "p2"])
    _write_paper_authors_table(pa, paper_authors_path, ["p1", "p2"], ["Ada Lovelace", "Bob Smith"])
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(pa.array([0.1, 0.2], type=pa.float32()), 2),
            }
        ),
        specter_path,
    )

    manifest = {
        "normalization_version": NORMALIZATION_VERSION,
        "paths": {
            "signatures": str(signatures_path),
            "papers": str(papers_path),
            "paper_authors": str(paper_authors_path),
            "specter": str(specter_path),
        },
        "signature_count": 2,
        "paper_count": 2,
    }

    metrics = convert_to_arrow.validate_arrow_dataset_manifest(
        manifest,
        require_embeddings=True,
        require_name_counts_index=False,
    )

    assert metrics["specter_count"] == 1
    assert metrics["missing_specter_paper_count"] == 1
    assert metrics["missing_specter_paper_examples"] == ["p2"]


def test_validate_arrow_dataset_manifest_accepts_specter2_only_manifest(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    signatures_path = tmp_path / "signatures.arrow"
    papers_path = tmp_path / "papers.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    specter2_path = tmp_path / "specter2.arrow"
    _write_signatures_table(pa, signatures_path, ["s1"], ["p1"])
    _write_papers_table(pa, papers_path, ["p1"])
    _write_paper_authors_table(pa, paper_authors_path, ["p1"], ["Ada Lovelace"])
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(pa.array([0.1, 0.2], type=pa.float32()), 2),
            }
        ),
        specter2_path,
    )

    metrics = convert_to_arrow.validate_arrow_dataset_manifest(
        {
            "normalization_version": NORMALIZATION_VERSION,
            "paths": {
                "signatures": str(signatures_path),
                "papers": str(papers_path),
                "paper_authors": str(paper_authors_path),
                "specter2": str(specter2_path),
            },
        },
        require_embeddings=True,
        require_name_counts_index=False,
    )

    assert metrics["specter_count"] == 1


@pytest.mark.parametrize(
    ("table_name", "missing_column", "require_embeddings"),
    [
        ("signatures", "author_first", False),
        ("papers", "journal_name", False),
        ("paper_authors", "author_name", False),
        ("specter2", "embedding", True),
    ],
)
def test_validate_arrow_dataset_manifest_rejects_missing_contract_required_columns(
    tmp_path: Path,
    table_name: str,
    missing_column: str,
    require_embeddings: bool,
) -> None:
    pa = pytest.importorskip("pyarrow")
    signatures_path = tmp_path / "signatures.arrow"
    papers_path = tmp_path / "papers.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    specter2_path = tmp_path / "specter2.arrow"
    _write_signatures_table(pa, signatures_path, ["s1"], ["p1"])
    _write_papers_table(pa, papers_path, ["p1"])
    _write_paper_authors_table(pa, paper_authors_path, ["p1"], ["Ada Lovelace"])
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(pa.array([0.1, 0.2], type=pa.float32()), 2),
            }
        ),
        specter2_path,
    )

    if table_name == "signatures":
        write_arrow_ipc_table(
            pa.table(
                {
                    "signature_id": pa.array(["s1"], type=pa.string()),
                    "paper_id": pa.array(["p1"], type=pa.string()),
                    "author_middle": pa.array([""], type=pa.string()),
                    "author_last": pa.array(["Lovelace"], type=pa.string()),
                    "author_suffix": pa.array([""], type=pa.string()),
                    "author_affiliations": pa.array([["Analytical Engine"]], type=pa.list_(pa.string())),
                    "author_orcid": pa.array([""], type=pa.string()),
                    "author_position": pa.array([0], type=pa.int64()),
                }
            ),
            signatures_path,
        )
    elif table_name == "papers":
        write_arrow_ipc_table(
            pa.table(
                {
                    "paper_id": pa.array(["p1"], type=pa.string()),
                    "title": pa.array(["Notes"], type=pa.string()),
                    "venue": pa.array(["Proceedings"], type=pa.string()),
                }
            ),
            papers_path,
        )
    elif table_name == "paper_authors":
        write_arrow_ipc_table(
            pa.table(
                {
                    "paper_id": pa.array(["p1"], type=pa.string()),
                    "position": pa.array([0], type=pa.int64()),
                }
            ),
            paper_authors_path,
        )
    elif table_name == "specter2":
        write_arrow_ipc_table(pa.table({"paper_id": pa.array(["p1"], type=pa.string())}), specter2_path)
    else:
        raise AssertionError(f"unexpected table_name: {table_name}")

    with pytest.raises(KeyError, match=missing_column):
        convert_to_arrow.validate_arrow_dataset_manifest(
            {
                "normalization_version": NORMALIZATION_VERSION,
                "paths": {
                    "signatures": str(signatures_path),
                    "papers": str(papers_path),
                    "paper_authors": str(paper_authors_path),
                    "specter2": str(specter2_path),
                },
            },
            require_embeddings=require_embeddings,
            require_name_counts_index=False,
        )


@pytest.mark.parametrize(
    ("signature_id", "author_name", "message"),
    [
        (None, "Ada Lovelace", "signatures.signature_id contains null value"),
        ("s1", "", "paper_authors.author_name contains empty value"),
    ],
)
def test_validate_arrow_dataset_manifest_rejects_null_or_empty_required_strings(
    tmp_path: Path,
    signature_id: str | None,
    author_name: str,
    message: str,
) -> None:
    pa = pytest.importorskip("pyarrow")
    signatures_path = tmp_path / "signatures.arrow"
    papers_path = tmp_path / "papers.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    _write_signatures_table(pa, signatures_path, [signature_id], ["p1"])
    _write_papers_table(pa, papers_path, ["p1"])
    _write_paper_authors_table(pa, paper_authors_path, ["p1"], [author_name])

    with pytest.raises(ValueError, match=message):
        convert_to_arrow.validate_arrow_dataset_manifest(
            {
                "normalization_version": NORMALIZATION_VERSION,
                "paths": {
                    "signatures": str(signatures_path),
                    "papers": str(papers_path),
                    "paper_authors": str(paper_authors_path),
                },
            },
            require_embeddings=False,
            require_name_counts_index=False,
        )


def test_validate_manifest_can_require_complete_specter_rows(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")

    signatures_path = tmp_path / "signatures.arrow"
    papers_path = tmp_path / "papers.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    specter_path = tmp_path / "specter.arrow"
    _write_signatures_table(pa, signatures_path, ["s1", "s2"], ["p1", "p2"])
    _write_papers_table(pa, papers_path, ["p1", "p2"])
    _write_paper_authors_table(pa, paper_authors_path, ["p1", "p2"], ["Ada Lovelace", "Bob Smith"])
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(pa.array([0.1, 0.2], type=pa.float32()), 2),
            }
        ),
        specter_path,
    )

    manifest = {
        "normalization_version": NORMALIZATION_VERSION,
        "paths": {
            "signatures": str(signatures_path),
            "papers": str(papers_path),
            "paper_authors": str(paper_authors_path),
            "specter": str(specter_path),
        },
        "signature_count": 2,
        "paper_count": 2,
    }

    with pytest.raises(ValueError, match="require_complete_embeddings=True.*p2"):
        convert_to_arrow.validate_arrow_dataset_manifest(
            manifest,
            require_embeddings=True,
            require_name_counts_index=False,
            require_complete_embeddings=True,
        )


def test_write_specter_arrow_reports_zero_size_vectors(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    pytest.importorskip("pyarrow")
    source_path = tmp_path / "specter.pkl"
    output_path = tmp_path / "specter.arrow"
    with source_path.open("wb") as outfile:
        pickle.dump(
            {
                "p1": np.array([0.1, 0.2], dtype=np.float32),
                "p2": np.array([], dtype=np.float32),
                "p3": np.array([0.3, 0.4], dtype=np.float32),
            },
            outfile,
        )

    with caplog.at_level("WARNING", logger="scripts.convert_to_arrow"):
        report = convert_to_arrow._write_specter_arrow(
            source_path=source_path,
            output_path=output_path,
            needed_paper_ids={"p1", "p2"},
            overwrite=True,
        )

    assert report["row_count"] == 1
    assert report["dropped_empty_embedding_count"] == 1
    assert "zero-size vectors" in caplog.text


def test_extra_specter_physical_layout_omits_nonportable_batch_index_path(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    specter_path = tmp_path / "specter2.arrow"
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1", "p2"], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(
                    pa.array([0.1, 0.2, 0.3, 0.4], type=pa.float32()),
                    2,
                ),
            }
        ),
        specter_path,
    )
    paths = {"specter2": str(specter_path)}
    raw_planner_index_metrics: dict[str, Any] = {}
    physical_layout: dict[str, Any] = {"tables": {}}

    convert_to_arrow._add_extra_specter_index_and_layout(
        paths=paths,
        raw_planner_index_metrics=raw_planner_index_metrics,
        physical_layout=physical_layout,
        table_key="specter2",
        output_dir=tmp_path,
        overwrite=True,
    )

    layout = physical_layout["tables"]["specter2"]
    assert layout["batch_index_path_key"] == "specter2_batch_index"
    assert "batch_index_path" not in layout
    assert Path(paths["specter2_batch_index"]).exists()


def test_validate_arrow_dataset_dir_resolves_relative_manifest_paths() -> None:
    pytest.importorskip("pyarrow")
    fixture = Path("tests/fixtures/arrow/pubmed_specter2/pubmed")

    metrics = convert_to_arrow.validate_arrow_dataset_dir(
        fixture,
        require_embeddings=True,
        require_name_counts_index=True,
    )

    assert metrics["signature_count"] == 2871
    assert metrics["name_counts_index_present"] is True


def test_validate_arrow_dataset_manifest_rejects_incomplete_name_counts_index(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    signatures_path = tmp_path / "signatures.arrow"
    papers_path = tmp_path / "papers.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    name_counts_index = tmp_path / "name_counts_index"
    name_counts_index.mkdir()
    (name_counts_index / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "name_counts_index_v1",
                "normalization_version": NORMALIZATION_VERSION,
                "source_provenance": {"normalization_version": NORMALIZATION_VERSION},
                "files": {"first": {"path": "missing-first.bin"}},
            }
        ),
        encoding="utf-8",
    )
    _write_signatures_table(pa, signatures_path, ["s1"], ["p1"])
    _write_papers_table(pa, papers_path, ["p1"])
    _write_paper_authors_table(pa, paper_authors_path, ["p1"], ["Ada Lovelace"])

    with pytest.raises(ValueError, match="missing files.first.path target"):
        convert_to_arrow.validate_arrow_dataset_manifest(
            {
                "normalization_version": NORMALIZATION_VERSION,
                "paths": {
                    "signatures": str(signatures_path),
                    "papers": str(papers_path),
                    "paper_authors": str(paper_authors_path),
                    "name_counts_index": str(name_counts_index),
                },
            },
            require_embeddings=False,
            require_name_counts_index=True,
        )


def test_validate_arrow_dataset_manifest_rejects_integer_id_columns(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    signatures_path = tmp_path / "signatures.arrow"
    papers_path = tmp_path / "papers.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    write_arrow_ipc_table(
        pa.table(
            {
                "signature_id": pa.array([1], type=pa.int64()),
                "paper_id": pa.array(["p1"], type=pa.string()),
            }
        ),
        signatures_path,
    )
    write_arrow_ipc_table(pa.table({"paper_id": pa.array(["p1"], type=pa.string())}), papers_path)
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "position": pa.array([0], type=pa.int64()),
                "author_name": pa.array(["Ada Lovelace"], type=pa.string()),
            }
        ),
        paper_authors_path,
    )

    with pytest.raises(ValueError, match="expected string"):
        convert_to_arrow.validate_arrow_dataset_manifest(
            {
                "normalization_version": NORMALIZATION_VERSION,
                "paths": {
                    "signatures": str(signatures_path),
                    "papers": str(papers_path),
                    "paper_authors": str(paper_authors_path),
                },
            },
            require_embeddings=False,
            require_name_counts_index=False,
        )


def test_validate_arrow_dataset_manifest_requires_batch_index_sidecar(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    signatures_path = tmp_path / "signatures.arrow"
    papers_path = tmp_path / "papers.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    _write_signatures_table(pa, signatures_path, ["s1"], ["p1"])
    _write_papers_table(pa, papers_path, ["p1"])
    _write_paper_authors_table(pa, paper_authors_path, ["p1"], ["Ada Lovelace"])
    manifest = {
        "normalization_version": NORMALIZATION_VERSION,
        "paths": {
            "signatures": str(signatures_path),
            "papers": str(papers_path),
            "paper_authors": str(paper_authors_path),
        },
        "physical_layout": {
            "tables": {
                "signatures": {
                    "key": "signature_id",
                    "batch_index_path_key": "signatures_batch_index",
                    "batch_index_present": True,
                    "max_record_batch_rows": 16384,
                    "actual_max_batch_rows": 1,
                }
            }
        },
    }

    with pytest.raises(FileNotFoundError, match="signatures_batch_index"):
        convert_to_arrow.validate_arrow_dataset_manifest(
            manifest,
            require_embeddings=False,
            require_name_counts_index=False,
        )
