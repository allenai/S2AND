from __future__ import annotations

import json
import sys

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")

from scripts.verification import compare_full_predict_arrow_parity as parity_module  # noqa: E402
from scripts.verification.compare_full_predict_arrow_parity import (  # noqa: E402
    _assert_exact,
    _build_arg_parser,
    _cluster_partition,
    _fixture_meta_path,
    _load_cluster_seeds_require,
    _numeric_report,
    _write_raw_planner_indexes_and_layout,
)


def test_assert_exact_rejects_constraint_index_mismatch_with_equal_values() -> None:
    report = {
        "distance_comparison": {},
        "feature_constraint_comparison": {
            "feature_matrix": {
                "allclose_equal_nan": True,
                "nan_mismatch_count": 0,
            },
            "constraints": {
                "left_indices_equal": False,
                "right_indices_equal": True,
                "values_equal": True,
            },
        },
        "clusters_exact_match": True,
    }

    with pytest.raises(AssertionError, match="constraint index mismatch"):
        _assert_exact(report)


def test_numeric_report_uses_configured_nan_mismatch_policy() -> None:
    left = np.asarray([1.0, np.nan])
    right = np.asarray([1.0, 2.0])

    assert _numeric_report(left, right, treat_nan_as_mismatch=True)["nan_mismatch_count"] == 1
    assert _numeric_report(left, right, treat_nan_as_mismatch=False)["nan_mismatch_count"] == 0


def test_cluster_partition_ignores_cluster_labels_and_member_order() -> None:
    incumbent = {"a": ["s1", "s2"], "b": ["s3"]}
    arrow = {"cluster_7": ["s3"], "cluster_9": ["s2", "s1"]}

    assert _cluster_partition(incumbent) == _cluster_partition(arrow)
    assert _cluster_partition(incumbent) != _cluster_partition({"a": ["s1"], "b": ["s2", "s3"]})


def test_parity_parser_compares_features_by_default(tmp_path) -> None:
    args = _build_arg_parser().parse_args(
        [
            "--fixture-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--output-json",
            str(tmp_path / "out.json"),
            "--model-path",
            str(tmp_path / "model"),
            "--block-size",
            "2",
        ]
    )

    assert args.compare_features is True

    args = _build_arg_parser().parse_args(
        [
            "--fixture-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--output-json",
            str(tmp_path / "out.json"),
            "--model-path",
            str(tmp_path / "model"),
            "--block-size",
            "2",
            "--no-compare-features",
        ]
    )

    assert args.compare_features is False


def test_parity_main_writes_output_json_before_asserting_mismatch(tmp_path, monkeypatch) -> None:
    report = {
        "distance_comparison": {"block-a": {"allclose_equal_nan": False}},
        "clusters_exact_match": True,
    }
    output_json = tmp_path / "report.json"
    monkeypatch.setattr(parity_module, "run", lambda _args: report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_full_predict_arrow_parity.py",
            "--fixture-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--output-json",
            str(output_json),
            "--model-path",
            str(tmp_path / "model"),
            "--block-size",
            "2",
        ],
    )

    with pytest.raises(AssertionError, match="distance mismatch"):
        parity_module.main()

    assert json.loads(output_json.read_text(encoding="utf-8")) == report


def test_parity_fixture_meta_paths_resolve_relative_to_fixture_dir(tmp_path) -> None:
    seed_path = tmp_path / "seeds.json"
    seed_path.write_text('{"s1": "c1", "s2": "c2"}\n', encoding="utf-8")
    meta = {
        "paths": {
            "signatures": "signatures.json",
            "cluster_seeds_require": "seeds.json",
        }
    }

    assert _fixture_meta_path(meta, tmp_path, "signatures") == tmp_path / "signatures.json"
    assert _load_cluster_seeds_require(meta, tmp_path, ["s1"], enabled=True) == {"s1": "c1"}


def _write_ipc(path, table) -> str:
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    return str(path)


def test_parity_arrow_writer_adds_current_raw_planner_indexes(tmp_path) -> None:
    arrow_paths = {
        "signatures": _write_ipc(
            tmp_path / "signatures.arrow",
            pa.table({"signature_id": pa.array(["s1"], type=pa.string())}),
        ),
        "papers": _write_ipc(
            tmp_path / "papers.arrow",
            pa.table({"paper_id": pa.array(["p1"], type=pa.string())}),
        ),
        "paper_authors": _write_ipc(
            tmp_path / "paper_authors.arrow",
            pa.table({"paper_id": pa.array(["p1"], type=pa.string())}),
        ),
        "specter": _write_ipc(
            tmp_path / "specter.arrow",
            pa.table({"paper_id": pa.array(["p1"], type=pa.string())}),
        ),
    }

    indexed_paths, index_metrics, physical_layout = _write_raw_planner_indexes_and_layout(arrow_paths, tmp_path)

    assert indexed_paths["signatures_batch_index"].endswith("signatures.signatures_batch_index.bin")
    assert index_metrics["signatures_batch_index"]["schema_version"] == "arrow_batch_lookup_index"
    assert index_metrics["signatures_batch_index"]["magic"] == "S2ABI002"
    assert physical_layout["schema"] == "s2and_arrow_physical_v1"
