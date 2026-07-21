from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from scripts._rust_suite import largest_block_cmd


def _comparison_result(*, backend: str, input_format: str, input_digest: str = "same") -> dict:
    return {
        "backend": backend,
        "input_format": input_format,
        "dataset": "qian",
        "block_key": "a smith",
        "original_block_size": 2,
        "effective_block_size": 2,
        "num_pairs": 1,
        "input_signature_ids_digest": input_digest,
        "cluster_membership_digest": "clusters",
        "signature_to_cluster_fingerprint": {"s1": "cluster", "s2": "cluster"},
        "quality_metrics": None,
        "anddata_build_seconds": 0.1,
        "arrow_block_load_seconds": 0.05,
        "warm_rust_featurizer_seconds": 0.0,
        "predict_seconds": 1.0,
        "total_seconds": 1.1,
        "peak_rss_gb": 0.2,
        "num_clusters": 1,
    }


def _compare_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    values = {
        "dataset": "qian",
        "block": "a smith",
        "timeout_hours": 0.1,
        "model_path": str(tmp_path / "model"),
        "data_root": str(tmp_path / "json"),
        "input_format": "json",
        "max_block_size": 10,
        "n_jobs": 1,
        "quality_check": False,
        "require_rust_release": False,
        "arrow_data_root": str(tmp_path / "arrow"),
        "specter_suffix": "_specter2.pkl",
        "write_json": "",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_bounded_block_signature_ids_are_order_identical_across_input_representations() -> None:
    assert largest_block_cmd._bounded_block_signature_ids(["s3", "s1", "s2"], 2) == ["s1", "s2"]
    assert largest_block_cmd._bounded_block_signature_ids(["s2", "s3", "s1"], 0) == ["s1", "s2", "s3"]


def test_compare_rejects_unbounded_runs_before_launching_subprocess(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        largest_block_cmd,
        "_run_single_subprocess",
        lambda **_kwargs: pytest.fail("comparison subprocess should not launch"),
    )

    with pytest.raises(ValueError, match="positive --max-block-size"):
        largest_block_cmd._compare_runs(_compare_args(tmp_path, max_block_size=0))


def test_compare_rejects_nonidentical_signature_sequence() -> None:
    python_result = _comparison_result(backend="python", input_format="json", input_digest="python")
    rust_result = _comparison_result(backend="rust", input_format="arrow", input_digest="rust")

    with pytest.raises(AssertionError, match="comparison inputs differ"):
        largest_block_cmd._assert_comparison_inputs_identical(
            python_result,
            rust_result,
            max_block_size=10,
        )


def test_compare_rejects_vacuous_same_route_results() -> None:
    first = _comparison_result(backend="rust", input_format="arrow")
    second = _comparison_result(backend="rust", input_format="arrow")

    with pytest.raises(AssertionError, match="Python/JSON versus Rust/Arrow"):
        largest_block_cmd._assert_comparison_inputs_identical(
            first,
            second,
            max_block_size=10,
        )


def test_removed_json_constraint_sampling_option_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["largest_block_cmd.py", "--model-path", "missing-model", "--constraint-sample", "1"],
    )

    with pytest.raises(SystemExit, match="2"):
        largest_block_cmd.main()

    assert "unrecognized arguments: --constraint-sample 1" in capsys.readouterr().err
