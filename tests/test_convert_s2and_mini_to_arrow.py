from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

import scripts.convert_to_arrow as convert_to_arrow
from s2and.arrow_inputs import ARROW_COLLECTION_KIND
from s2and.consts import PUBLIC_DATA_FORMAT_VERSION
from s2and.incremental_linking.feature_block import (
    write_arrow_ipc_table,
    write_raw_arrow_batch_lookup_indexes,
)
from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from scripts.convert_to_arrow import RuntimeDatasetSources
from tests.helpers import tiny_name_counts_tuple, write_test_arrow_artifact_manifest


def _fake_sources(tmp_path: Path, dataset: str) -> RuntimeDatasetSources:
    source_dir = tmp_path / "source" / dataset
    return RuntimeDatasetSources(
        dataset=dataset,
        source_dir=source_dir,
        signatures_path=source_dir / f"{dataset}_signatures.json",
        papers_path=source_dir / f"{dataset}_papers.json",
    )


def _minimal_tables(row_count: int = 1) -> dict[str, pa.Table]:
    paper_ids = [f"p{i + 1}" for i in range(row_count)]
    return {
        "signatures": pa.table(
            {
                "signature_id": pa.array([f"s{i + 1}" for i in range(row_count)], type=pa.string()),
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
        "papers": pa.table(
            {
                "paper_id": pa.array(paper_ids, type=pa.string()),
                "title": pa.array(["Notes"] * row_count, type=pa.string()),
                "venue": pa.array(["Proceedings"] * row_count, type=pa.string()),
                "journal_name": pa.array(["Journal"] * row_count, type=pa.string()),
            }
        ),
        "paper_authors": pa.table(
            {
                "paper_id": pa.array(paper_ids, type=pa.string()),
                "position": pa.array(list(range(row_count)), type=pa.int64()),
                "author_name": pa.array(["Ada Lovelace"] * row_count, type=pa.string()),
            }
        ),
    }


def _write_tables(tmp_path: Path, tables: dict[str, pa.Table]) -> dict[str, str]:
    paths = {name: str(tmp_path / f"{name}.arrow") for name in tables}
    for name, table in tables.items():
        write_arrow_ipc_table(table, Path(paths[name]))
    return paths


def test_join_canonical_benchmark_names_replaces_only_name_fields() -> None:
    signatures = {
        "s2": {
            "signature_id": "s2",
            "paper_id": "p2",
            "author_info": {"first": "Jean Marie", "middle": None, "last": "Müller", "block": "keep"},
        },
        "s1": {
            "signature_id": "s1",
            "paper_id": "p1",
            "author_info": {"first": "Ada", "middle": None, "last": "Lovelace", "block": "keep"},
        },
    }
    original = json.loads(json.dumps(signatures))

    joined, report = convert_to_arrow.join_canonical_benchmark_names(
        signatures,
        [
            {"signature_id": "s1", "first": "ada", "middle": "", "last": "lovelace"},
            {"signature_id": "s2", "first": "jean-marie", "middle": "", "last": "muller"},
        ],
    )

    assert signatures == original
    assert list(joined) == ["s1", "s2"]
    assert joined["s2"]["author_info"] == {
        "first": "jean-marie",
        "middle": "",
        "last": "muller",
        "block": "keep",
    }
    assert joined["s2"]["paper_id"] == "p2"
    assert report == {
        "rows": 2,
        "changed_signatures": 2,
        "field_changes": {"first": 2, "middle": 2, "last": 2},
    }


def test_join_canonical_benchmark_names_rejects_duplicate_missing_or_extra_ids() -> None:
    signatures = {
        "s1": {
            "signature_id": "s1",
            "paper_id": "p1",
            "author_info": {"first": "Ada", "middle": None, "last": "Lovelace"},
        }
    }
    cases = (
        ("duplicate", ["s1", "s1"], "duplicate signature_id"),
        ("missing", [], "missing=['s1']"),
        ("extra", ["s1", "s2"], "extra=['s2']"),
    )
    for case_id, canonical_ids, message in cases:
        canonical_rows = [
            {"signature_id": signature_id, "first": "ada", "middle": "", "last": "lovelace"}
            for signature_id in canonical_ids
        ]

        try:
            convert_to_arrow.join_canonical_benchmark_names(signatures, canonical_rows)
        except ValueError as error:
            assert message in str(error), f"{case_id}: {error}"
        else:
            raise AssertionError(f"{case_id}: invalid canonical IDs were accepted")


def test_arrow_manifest_files_exclude_request_local_sidecars(tmp_path: Path) -> None:
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

    first = convert_to_arrow.build_arrow_artifact_manifest(paths, tmp_path)["files"]
    for path in sidecar_paths.values():
        path.write_bytes(b"request-local replacement")
    second = convert_to_arrow.build_arrow_artifact_manifest(paths, tmp_path)["files"]

    assert set(first) == {"signatures"}
    assert second == first


def test_run_full_discovers_datasets_only_when_explicit(
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
    assert [call["selected_embedding"] for call in calls] == ["specter2", "specter2"]

    calls.clear()

    def fake_linker_sources(_raw_root: Path, _embeddings_root: Path, dataset: str) -> RuntimeDatasetSources:
        return RuntimeDatasetSources(
            dataset=dataset,
            source_dir=tmp_path / "raw" / dataset,
            signatures_path=tmp_path / "raw" / dataset / "signatures.json",
            papers_path=tmp_path / "raw" / dataset / "papers.json",
            specter2_path=tmp_path / "embeddings" / dataset / "specter2.pkl",
        )

    monkeypatch.setattr(
        convert_to_arrow, "discover_linker_replay_datasets", lambda _raw_root, _embeddings_root: ["pubmed"]
    )
    monkeypatch.setattr(convert_to_arrow, "linker_replay_dataset_sources", fake_linker_sources)
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
    assert calls[0]["selected_embedding"] == "specter2"


def test_runtime_conversion_requires_selected_specter2_before_writing(tmp_path: Path) -> None:
    output_dir = tmp_path / "out" / "dummy"

    with pytest.raises(FileNotFoundError, match="requires SPECTER2"):
        convert_to_arrow.convert_runtime_dataset_to_arrow(
            sources=_fake_sources(tmp_path, "dummy"),
            output_dir=output_dir,
            root_manifest_dir=tmp_path / "out",
            name_counts_index_root=None,
            n_jobs=1,
            overwrite=False,
            skip_name_counts_index=True,
        )

    assert not output_dir.exists()


def test_runtime_conversion_defaults_to_one_physical_specter2_table(tmp_path: Path) -> None:
    source_dir = tmp_path / "source" / "dummy"
    source_dir.mkdir(parents=True)
    signatures_path = source_dir / "dummy_signatures.json"
    papers_path = source_dir / "dummy_papers.json"
    specter1_path = source_dir / "dummy_specter.pickle"
    specter2_path = source_dir / "dummy_specter2.pkl"
    signatures_path.write_text(
        json.dumps(
            {
                "s1": {
                    "signature_id": "s1",
                    "paper_id": "p1",
                    "author_info": {
                        "position": 0,
                        "block": "a lovelace",
                        "first": "Ada",
                        "middle": None,
                        "last": "Lovelace",
                        "suffix": None,
                        "affiliations": [],
                        "email": None,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    papers_path.write_text(
        json.dumps(
            {
                "p1": {
                    "paper_id": "p1",
                    "title": "Notes on the Analytical Engine",
                    "abstract": None,
                    "authors": [{"position": 0, "author_name": "Ada Lovelace"}],
                    "venue": None,
                    "journal_name": None,
                    "year": 1843,
                }
            }
        ),
        encoding="utf-8",
    )
    with specter1_path.open("wb") as stream:
        pickle.dump({"p1": np.asarray([1.0, 1.0], dtype=np.float32)}, stream)
    with specter2_path.open("wb") as stream:
        pickle.dump({"p1": np.asarray([2.0, 2.0], dtype=np.float32)}, stream)

    output_root = tmp_path / "out"
    output_dir = output_root / "dummy"
    manifest = convert_to_arrow.convert_runtime_dataset_to_arrow(
        sources=RuntimeDatasetSources(
            dataset="dummy",
            source_dir=source_dir,
            signatures_path=signatures_path,
            papers_path=papers_path,
            specter_path=specter1_path,
            specter2_path=specter2_path,
        ),
        output_dir=output_dir,
        root_manifest_dir=output_root,
        name_counts_index_root=None,
        n_jobs=1,
        overwrite=False,
        skip_name_counts_index=True,
    )

    assert Path(manifest["paths"]["specter"]).name == "specter2.arrow"
    assert (output_dir / "specter2.arrow").is_file()
    assert not (output_dir / "specter.arrow").exists()


def test_root_manifest_upsert_keeps_dataset_order_stable(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    for dataset_name in ("b", "a"):
        dataset_dir = output_root / dataset_name
        dataset_dir.mkdir()
        (dataset_dir / "manifest.json").write_text(
            json.dumps({"dataset": dataset_name}),
            encoding="utf-8",
        )
        convert_to_arrow._upsert_root_manifest(output_root, dataset_name=dataset_name, dataset_dir=dataset_dir)

    dataset_dir = output_root / "a"
    convert_to_arrow._upsert_root_manifest(output_root, dataset_name="a", dataset_dir=dataset_dir)

    root_manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert root_manifest["kind"] == ARROW_COLLECTION_KIND
    assert root_manifest["format_version"] == PUBLIC_DATA_FORMAT_VERSION
    assert set(root_manifest) == {"kind", "format_version", "dataset_manifests"}
    assert list(root_manifest["dataset_manifests"]) == ["a", "b"]
    for dataset_name, binding in root_manifest["dataset_manifests"].items():
        manifest_path = output_root / binding["path"]
        assert set(binding) == {"path", "sha256"}
        assert binding["path"] == f"{dataset_name}/manifest.json"
        assert binding["sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def test_generic_root_updates_reject_published_roots(tmp_path: Path) -> None:
    output_root = tmp_path / "release"
    output_root.mkdir()
    dataset_dir = output_root / "qian"
    dataset_dir.mkdir()
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"dataset": "qian"}),
        encoding="utf-8",
    )
    convert_to_arrow._write_root_manifest(
        output_root,
        dataset_manifests={"qian": "qian/manifest.json"},
        release_version="1.3",
    )
    original_manifest = (output_root / "manifest.json").read_bytes()

    for operation in (
        lambda: convert_to_arrow._validate_existing_root_manifest(output_root / "manifest.json"),
        lambda: convert_to_arrow._upsert_root_manifest(
            output_root,
            dataset_name="qian",
            dataset_dir=dataset_dir,
        ),
    ):
        with pytest.raises(ValueError, match="cannot modify a published root"):
            operation()

    assert (output_root / "manifest.json").read_bytes() == original_manifest


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


def test_validate_manifest_require_embeddings_reports_missing_specter_rows(tmp_path: Path) -> None:
    tables = _minimal_tables(row_count=2)
    tables["specter"] = pa.table(
        {
            "paper_id": pa.array(["p1"], type=pa.string()),
            "embedding": pa.FixedSizeListArray.from_arrays(pa.array([0.1, 0.2], type=pa.float32()), 2),
        }
    )
    paths, _metrics = write_raw_arrow_batch_lookup_indexes(_write_tables(tmp_path, tables), tmp_path)

    metrics = convert_to_arrow.validate_arrow_dataset_manifest(
        {"paths": paths},
        require_embeddings=True,
        require_name_counts_index=False,
    )
    assert metrics["specter_count"] == 1
    assert metrics["missing_specter_paper_count"] == 1
    assert metrics["missing_specter_paper_examples"] == ["p2"]

    with pytest.raises(ValueError, match="require_complete_embeddings=True.*p2"):
        convert_to_arrow.validate_arrow_dataset_manifest(
            {"paths": paths},
            require_embeddings=True,
            require_name_counts_index=False,
            require_complete_embeddings=True,
        )


def test_validate_arrow_dataset_manifest_rejects_malformed_optional_column(tmp_path: Path) -> None:
    tables = _minimal_tables()
    tables["papers"] = tables["papers"].append_column("language_reliability", pa.array([0.75], type=pa.float32()))
    paths = _write_tables(tmp_path, tables)

    with pytest.raises(ValueError, match="language_reliability.*expected float64"):
        convert_to_arrow.validate_arrow_dataset_manifest(
            {"paths": paths},
            require_embeddings=False,
            require_name_counts_index=False,
        )


@pytest.mark.parametrize(
    ("table_name", "column_name"),
    [("signatures", "signature_id"), ("signatures", "author_position"), ("paper_authors", "author_name")],
)
def test_validate_arrow_dataset_manifest_rejects_null_required_values(
    tmp_path: Path, table_name: str, column_name: str
) -> None:
    tables = _minimal_tables()
    table = tables[table_name]
    tables[table_name] = table.set_column(
        table.schema.get_field_index(column_name),
        column_name,
        pa.array([None], type=table.schema.field(column_name).type),
    )
    paths = _write_tables(tmp_path, tables)

    with pytest.raises(ValueError, match=rf"{table_name}\.{column_name} contains null value"):
        convert_to_arrow.validate_arrow_dataset_manifest(
            {"paths": paths},
            require_embeddings=False,
            require_name_counts_index=False,
        )


def test_validate_arrow_dataset_manifest_accepts_blank_paper_author_names(tmp_path: Path) -> None:
    tables = _minimal_tables(row_count=2)
    tables["paper_authors"] = tables["paper_authors"].set_column(2, "author_name", pa.array(["", "   "]))
    paths, _metrics = write_raw_arrow_batch_lookup_indexes(_write_tables(tmp_path, tables), tmp_path)
    metrics = convert_to_arrow.validate_arrow_dataset_manifest(
        {"paths": paths},
        require_embeddings=False,
        require_name_counts_index=False,
    )

    assert metrics["paper_author_count"] == 2


def test_write_specter_arrow_reports_zero_size_vectors(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
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


def test_validate_arrow_dataset_dir_resolves_relative_manifest_paths(tmp_path: Path) -> None:
    tables = _minimal_tables()
    tables["specter"] = pa.table(
        {
            "paper_id": pa.array(["p1"], type=pa.string()),
            "embedding": pa.FixedSizeListArray.from_arrays(pa.array([0.1, 0.2], type=pa.float32()), 2),
        }
    )
    paths, _metrics = write_raw_arrow_batch_lookup_indexes(_write_tables(tmp_path, tables), tmp_path)
    paths["name_counts_index"], _metrics = write_name_counts_index(
        tmp_path / "name_counts_index",
        tiny_name_counts_tuple(),
    )
    write_test_arrow_artifact_manifest(tmp_path, paths)

    metrics = convert_to_arrow.validate_arrow_dataset_dir(
        tmp_path,
        require_embeddings=True,
        require_name_counts_index=True,
    )
    assert metrics["signature_count"] == 1
    assert metrics["name_counts_index_present"] is True


def test_validate_arrow_dataset_manifest_rejects_incomplete_name_counts_index(tmp_path: Path) -> None:
    paths = _write_tables(tmp_path, _minimal_tables())
    paths["name_counts_index"], _metrics = write_name_counts_index(tmp_path, tiny_name_counts_tuple())
    (Path(paths["name_counts_index"]) / "first.bin").unlink()

    with pytest.raises(ValueError, match=r"files\.first target"):
        convert_to_arrow.validate_arrow_dataset_manifest(
            {"paths": paths},
            require_embeddings=False,
            require_name_counts_index=True,
        )


def test_validate_arrow_dataset_manifest_requires_batch_index_sidecar(tmp_path: Path) -> None:
    paths = _write_tables(tmp_path, _minimal_tables())
    with pytest.raises(FileNotFoundError, match="signatures_batch_index"):
        convert_to_arrow.validate_arrow_dataset_manifest(
            {"paths": paths},
            require_embeddings=False,
            require_name_counts_index=False,
        )
