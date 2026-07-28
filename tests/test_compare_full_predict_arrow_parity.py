from __future__ import annotations

import json
import sys
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")

from s2and.arrow_inputs import ArrowDataset  # noqa: E402
from scripts.verification import compare_full_predict_arrow_parity as parity_module  # noqa: E402
from scripts.verification.compare_full_predict_arrow_parity import (  # noqa: E402
    _assert_exact,
    _cluster_partition,
    _feature_constraint_report,
    _fixture_meta_path,
    _load_cluster_seeds_require,
    _numeric_report,
    _write_raw_planner_indexes_and_layout,
)


def _parity_argv(
    tmp_path,
    *,
    fixture_dir=None,
    model_dir=None,
) -> list[str]:
    return [
        "compare_full_predict_arrow_parity.py",
        "--fixture-dir",
        str(fixture_dir or tmp_path),
        "--evaluation-plan",
        str(tmp_path / "evaluation_plan.json"),
        "--output-dir",
        str(tmp_path / "out"),
        "--output-json",
        str(tmp_path / "report.json"),
        "--model-path",
        str(model_dir or tmp_path / "model"),
        "--run-binding",
        str(tmp_path / "run_binding.json"),
        "--block-size",
        "2",
    ]


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


def test_numeric_report_requires_exact_integer_reference_values() -> None:
    report = _numeric_report(
        np.asarray([[1.0, 1.25]]),
        np.asarray([[1.0 + 5e-7, 1.25 + 5e-7]]),
        treat_nan_as_mismatch=True,
        atol=1e-6,
        exact_integer_reference=True,
    )

    assert report["allclose_equal_nan"] is False
    assert report["exact_integer_mismatch_count"] == 1


def test_feature_constraint_report_compares_real_tiny_python_and_arrow_paths(
    tmp_path,
    monkeypatch,
) -> None:
    from s2and.feature_port import _get_rust_featurizer
    from tests.helpers import build_arrow_training_dataset, build_dummy_dataset

    monkeypatch.setenv("S2AND_BACKEND", "python")
    python_dataset = build_dummy_dataset("full-predict-parity-report", name_counts_index=True)
    arrow_dataset = build_arrow_training_dataset(python_dataset, tmp_path)
    arrow_featurizer = _get_rust_featurizer(arrow_dataset)
    signature_ids = list(python_dataset.signatures)[:4]

    report = _feature_constraint_report(
        python_dataset,
        arrow_featurizer,
        signature_ids,
        n_jobs=1,
    )

    assert report["pair_count"] == 6
    assert report["feature_matrix"]["allclose_equal_nan"] is True
    assert report["feature_matrix"]["nan_mismatch_count"] == 0
    assert report["constraints"]["left_indices_equal"] is True
    assert report["constraints"]["right_indices_equal"] is True
    assert report["constraints"]["values_equal"] is True


def test_cluster_partition_ignores_cluster_labels_and_member_order() -> None:
    incumbent = {"a": ["s1", "s2"], "b": ["s3"]}
    arrow = {"cluster_7": ["s3"], "cluster_9": ["s2", "s1"]}

    assert _cluster_partition(incumbent) == _cluster_partition(arrow)
    assert _cluster_partition(incumbent) != _cluster_partition({"a": ["s1"], "b": ["s2", "s3"]})


def test_parity_main_rejects_mismatch_without_report(tmp_path, monkeypatch) -> None:
    report = {
        "distance_comparison": {"block-a": {"allclose_equal_nan": False}},
        "clusters_exact_match": True,
    }
    output_json = tmp_path / "report.json"
    monkeypatch.setattr(parity_module, "run", lambda _args: report)
    monkeypatch.setattr(sys, "argv", _parity_argv(tmp_path))

    with pytest.raises(AssertionError, match="distance mismatch"):
        parity_module.main()

    assert not output_json.exists()


def test_parity_main_writes_fresh_report_on_success(tmp_path, monkeypatch) -> None:
    report = {
        "distance_comparison": {},
        "clusters_exact_match": True,
    }
    output_json = tmp_path / "report.json"
    monkeypatch.setattr(parity_module, "run", lambda _args: report)
    monkeypatch.setattr(sys, "argv", _parity_argv(tmp_path))

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


def test_release_parity_inputs_are_content_and_workload_bound(tmp_path) -> None:
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    signatures = fixture_dir / "signatures.json"
    papers = fixture_dir / "papers.json"
    signatures.write_text('{"s1": {}}\n', encoding="utf-8")
    papers.write_text("{}\n", encoding="utf-8")
    (fixture_dir / "meta.json").write_text(
        json.dumps(
            {
                "block": "reviewed",
                "dataset": "dummy",
                "paths": {
                    "papers": papers.name,
                    "signatures": signatures.name,
                },
            }
        ),
        encoding="utf-8",
    )
    workload = {
        "block_size": 2,
        "compare_features": True,
        "include_specter": False,
        "n_jobs": 1,
        "total_ram_bytes": 1_000_000,
        "use_cluster_seeds": False,
    }
    plan = {
        "block": "reviewed",
        "dataset": "dummy",
        "files": {
            "papers": (papers.resolve(), sha256(papers.read_bytes()).hexdigest()),
            "signatures": (signatures.resolve(), sha256(signatures.read_bytes()).hexdigest()),
        },
        "fixture_dir": fixture_dir.resolve(),
        "workload": workload,
    }
    args = SimpleNamespace(
        block_size=2,
        compare_features=True,
        fixture_dir=fixture_dir,
        n_jobs=1,
        no_specter=True,
        total_ram_bytes=1_000_000,
        use_cluster_seeds=False,
    )

    assert parity_module._validated_release_parity_inputs(args, plan)["dataset"] == "dummy"
    signatures.write_text('{"changed": {}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="signatures"):
        parity_module._validated_release_parity_inputs(args, plan)


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
    assert index_metrics["signatures_batch_index"]["magic"] == "S2ABI002"
    assert physical_layout["tables"]["signatures"]["batch_index_present"] is True


def test_parity_arrow_writer_publishes_validator_compatible_manifest(tmp_path) -> None:
    from tests.helpers import write_minimal_arrow_prediction_bundle

    write_minimal_arrow_prediction_bundle(tmp_path)
    with ArrowDataset.open(tmp_path) as dataset:
        assert dataset.root == tmp_path.resolve()
