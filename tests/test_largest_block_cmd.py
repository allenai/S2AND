from __future__ import annotations

import argparse
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
        "constraint_parity": None,
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
        "constraint_sample": 0,
        "n_jobs": 1,
        "quality_check": False,
        "constraint_sample_seed": 42,
        "require_rust_release": False,
        "arrow_data_root": str(tmp_path / "arrow"),
        "specter_suffix": "_specter2.pkl",
        "write_json": "",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_run_single_arrow_rejects_python_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires --backend rust"):
        largest_block_cmd._run_single(
            backend="python",
            dataset_name="qian",
            block_key="a smith",
            n_jobs=1,
            profile_output_path=str(tmp_path / "profile.txt"),
            model_path=str(tmp_path / "model"),
            input_format="arrow",
        )


def test_bounded_block_signature_ids_are_order_identical_across_input_representations() -> None:
    assert largest_block_cmd._bounded_block_signature_ids(["s3", "s1", "s2"], 2) == ["s1", "s2"]
    assert largest_block_cmd._bounded_block_signature_ids(["s2", "s3", "s1"], 0) == ["s1", "s2", "s3"]


def test_run_single_json_rejects_rust_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires --input-format arrow"):
        largest_block_cmd._run_single(
            backend="rust",
            dataset_name="qian",
            block_key="a smith",
            n_jobs=1,
            profile_output_path=str(tmp_path / "profile.txt"),
            model_path=str(tmp_path / "model"),
            input_format="json",
        )


def test_compare_routes_python_json_against_rust_arrow(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict] = []

    def fake_run_single_subprocess(**kwargs):
        calls.append(kwargs)
        return _comparison_result(backend=kwargs["backend"], input_format=kwargs["input_format"])

    monkeypatch.setattr(largest_block_cmd, "_run_single_subprocess", fake_run_single_subprocess)

    largest_block_cmd._compare_runs(_compare_args(tmp_path))

    assert [(call["backend"], call["input_format"]) for call in calls] == [
        ("python", "json"),
        ("rust", "arrow"),
    ]
    assert calls[1]["arrow_data_root"] == str(tmp_path / "arrow")
    assert calls[1]["specter_suffix"] == "_specter2.pkl"
    assert all(call["max_block_size"] == 10 for call in calls)


def test_compare_rejects_unbounded_runs_before_launching_subprocess(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        largest_block_cmd,
        "_run_single_subprocess",
        lambda **_kwargs: pytest.fail("comparison subprocess should not launch"),
    )

    with pytest.raises(ValueError, match="positive --max-block-size"):
        largest_block_cmd._compare_runs(_compare_args(tmp_path, max_block_size=0))


def test_compare_requires_explicit_current_arrow_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        largest_block_cmd,
        "_run_single_subprocess",
        lambda **_kwargs: pytest.fail("comparison subprocess should not launch"),
    )

    with pytest.raises(ValueError, match="explicit --arrow-data-root"):
        largest_block_cmd._compare_runs(_compare_args(tmp_path, arrow_data_root=""))


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


def test_sample_unique_pair_indices_never_duplicates_pairs() -> None:
    pairs = largest_block_cmd._sample_unique_pair_indices(6, 12, largest_block_cmd.random.Random(7))  # noqa: SLF001

    assert len(pairs) == 12
    assert len(set(pairs)) == 12
    assert all(0 <= i < j < 6 for i, j in pairs)


def test_sample_unique_pair_indices_returns_all_pairs_when_sample_exceeds_population() -> None:
    pairs = largest_block_cmd._sample_unique_pair_indices(4, 99, largest_block_cmd.random.Random(7))  # noqa: SLF001

    assert pairs == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
