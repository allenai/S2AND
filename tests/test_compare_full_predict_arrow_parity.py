from __future__ import annotations

import argparse
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
    _feature_constraint_report,
    _fixture_meta_path,
    _jsonable,
    _load_cluster_seeds_require,
    _numeric_report,
    _requested_name_counts_index,
    _write_arrow_artifact_manifest,
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


def test_feature_constraint_report_compares_python_anddata_to_rust_arrow(monkeypatch) -> None:
    signature_ids = ["s1", "s2", "s3"]
    expected_signature_pairs = [("s1", "s2"), ("s1", "s3"), ("s2", "s3")]
    expected_features = np.asarray([[1.25, np.nan], [2.0, 3.0], [4.0, 5.0]])
    python_constraint_values = {
        ("s1", "s2"): None,
        ("s1", "s3"): 0.0,
        ("s2", "s3"): 10000.0,
    }

    class FakeDataset:
        def __init__(self) -> None:
            self.constraint_pairs: list[tuple[str, str]] = []

        def get_constraint(self, left: str, right: str, **_kwargs):
            self.constraint_pairs.append((left, right))
            return python_constraint_values[(left, right)]

    class FakeArrowFeaturizer:
        def signature_ids(self):
            return signature_ids

        def featurize_pairs_matrix_indexed(self, pairs, selected_indices, num_threads, nan_value):
            assert pairs == [(0, 1), (0, 2), (1, 2)]
            assert selected_indices is None
            assert num_threads == 2
            assert np.isnan(nan_value)
            return expected_features + np.asarray([[5e-7, 0.0], [0.0, 0.0], [0.0, 0.0]])

    def fake_many_pairs_featurize(pairs, dataset, _info, **kwargs):
        assert dataset is python_dataset
        assert [(left, right) for left, right, _label in pairs] == expected_signature_pairs
        assert kwargs["runtime_context"].backend == "python"
        return expected_features, np.zeros(len(pairs)), None

    def fake_get_constraints(block_indices, **kwargs):
        assert block_indices == [0, 1, 2]
        assert kwargs["featurizer"] is arrow_featurizer
        return [0, 0, 1], [1, 2, 2], [None, 0.0, 10000.0]

    python_dataset = FakeDataset()
    arrow_featurizer = FakeArrowFeaturizer()
    monkeypatch.setattr("s2and.featurizer.many_pairs_featurize", fake_many_pairs_featurize)
    monkeypatch.setattr(
        "s2and.rust_calls.get_constraints_block_upper_triangle_indexed_rust",
        fake_get_constraints,
    )

    report = _feature_constraint_report(
        python_dataset,
        arrow_featurizer,
        signature_ids,
        n_jobs=2,
    )

    assert report["reference_backend"] == "python_anddata"
    assert report["candidate_backend"] == "rust_arrow"
    assert report["feature_matrix"]["allclose_equal_nan"] is True
    assert report["feature_matrix"]["atol"] == 1e-6
    assert report["constraints"]["values_equal"] is True
    assert python_dataset.constraint_pairs == expected_signature_pairs


def test_cluster_partition_ignores_cluster_labels_and_member_order() -> None:
    incumbent = {"a": ["s1", "s2"], "b": ["s3"]}
    arrow = {"cluster_7": ["s3"], "cluster_9": ["s2", "s1"]}

    assert _cluster_partition(incumbent) == _cluster_partition(arrow)
    assert _cluster_partition(incumbent) != _cluster_partition({"a": ["s1"], "b": ["s2", "s3"]})


def test_jsonable_converts_validated_arrow_mapping() -> None:
    import s2and.arrow_inputs as arrow_inputs_module
    from s2and.arrow_inputs import ValidatedArrowInputs
    from s2and.consts import NORMALIZATION_VERSION

    paths = ValidatedArrowInputs._from_verified(
        paths={"signatures": "signatures.arrow"},
        generation_id="generation",
        normalization_version=NORMALIZATION_VERSION,
        capability=arrow_inputs_module._VERIFIED_ARROW_INPUTS_CAPABILITY,  # noqa: SLF001
    )

    assert _jsonable(paths) == {"signatures": "signatures.arrow"}


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
    assert args.name_counts_index is None
    assert args.name_artifact_dir is None

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


def test_parity_parser_rejects_two_name_count_sources(tmp_path) -> None:
    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args(
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
                "--name-artifact-dir",
                str(tmp_path / "legacy"),
                "--name-counts-index",
                str(tmp_path / "index"),
            ]
        )


def test_parity_legacy_name_artifact_dir_resolves_its_index(tmp_path) -> None:
    assert _requested_name_counts_index(
        argparse.Namespace(name_artifact_dir=tmp_path / "legacy", name_counts_index=None)
    ) == (tmp_path / "legacy" / "name_counts_index")
    assert _requested_name_counts_index(
        argparse.Namespace(name_artifact_dir=None, name_counts_index=tmp_path / "direct")
    ) == (tmp_path / "direct")


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


def test_parity_arrow_writer_publishes_validator_compatible_manifest(tmp_path) -> None:
    from s2and.arrow_inputs import validate_arrow_prediction_artifacts
    from s2and.consts import NORMALIZATION_VERSION

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
    }
    arrow_paths, _index_metrics, _physical_layout = _write_raw_planner_indexes_and_layout(
        arrow_paths,
        tmp_path,
    )
    arrow_paths["manifest"] = str(
        _write_arrow_artifact_manifest(
            arrow_paths,
            tmp_path,
            normalization_version=NORMALIZATION_VERSION,
        )
    )

    validated = validate_arrow_prediction_artifacts(
        arrow_paths,
        require_specter=False,
        require_name_counts_index=False,
        expected_normalization_version=NORMALIZATION_VERSION,
        context="full prediction parity regression test",
    )

    assert validated.generation_id
    assert validated.normalization_version == NORMALIZATION_VERSION
