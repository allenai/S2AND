from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from scripts._rust_suite import largest_block_cmd


class _FakeRSSMonitor:
    peak_gb = 0.25

    def __init__(self, *, interval_seconds: float) -> None:
        del interval_seconds

    def __enter__(self) -> _FakeRSSMonitor:
        return self

    def __exit__(self, *_args) -> None:
        return None


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
        "subblocking_threshold": largest_block_cmd.DEFAULT_SUBBLOCKING_THRESHOLD,
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


def test_single_subprocess_forwards_subblocking_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = list(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(largest_block_cmd.subprocess, "run", fake_run)
    monkeypatch.setattr(largest_block_cmd, "extract_marked_json_payload", lambda *_args, **_kwargs: {})

    largest_block_cmd._run_single_subprocess(
        backend="rust",
        dataset_name="qian",
        block_key="a smith",
        n_jobs=1,
        profile_output_path=str(tmp_path / "profile.txt"),
        model_path=str(tmp_path / "model"),
        data_root=str(tmp_path / "data"),
        max_block_size=0,
        run_label="rust",
        timeout_seconds=10,
        quality_check=False,
        emit_signature_map=False,
        require_rust_release=False,
        input_format="arrow",
        arrow_data_root=str(tmp_path / "arrow"),
        subblocking_threshold=321,
    )

    threshold_index = captured["cmd"].index("--subblocking-threshold")
    assert captured["cmd"][threshold_index + 1] == "321"


def test_json_result_reports_subblocking_threshold_as_not_applied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import s2and.data as data_module
    import s2and.model as model_module
    import s2and.production_model as production_model_module

    fake_data = SimpleNamespace(clusters=None, get_blocks=lambda: {"a smith": ["s1", "s2"]})
    fake_clusterer = SimpleNamespace(
        classifier=object(),
        nameless_classifier=object(),
        predict_helper=lambda *_args, **_kwargs: ({"cluster": ["s1", "s2"]}, None),
    )

    monkeypatch.setattr(data_module, "ANDData", lambda **_kwargs: fake_data)
    monkeypatch.setattr(production_model_module, "load_production_model", lambda _path: fake_clusterer)
    monkeypatch.setattr(model_module, "_ensure_lightgbm_fitted", lambda _classifier: None)
    monkeypatch.setattr(largest_block_cmd, "_check_paths", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(largest_block_cmd, "_write_profile_output", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(largest_block_cmd, "ProcessTreeRSSMonitor", _FakeRSSMonitor)
    monkeypatch.setattr(largest_block_cmd, "build_run_metadata", lambda **_kwargs: {})

    result = largest_block_cmd._run_single(
        backend="python",
        dataset_name="qian",
        block_key="a smith",
        n_jobs=1,
        profile_output_path=str(tmp_path / "profile.txt"),
        model_path=str(tmp_path / "model"),
        data_root=str(tmp_path / "data"),
        subblocking_threshold=321,
    )

    assert result["subblocking_threshold"] is None


@pytest.mark.parametrize(
    ("configured_threshold", "expected_applied_threshold"),
    [(321, 321), (0, None)],
)
def test_arrow_result_reports_only_applied_subblocking_threshold(
    configured_threshold: int,
    expected_applied_threshold: int | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import s2and.model as model_module
    import s2and.production_model as production_model_module

    captured: dict[str, int | None] = {}

    def predict_from_arrow_paths(*_args, **kwargs):
        captured["batching_threshold"] = kwargs["batching_threshold"]
        return {"cluster": ["s1", "s2"]}, None

    fake_clusterer = SimpleNamespace(
        classifier=object(),
        nameless_classifier=object(),
        predict_from_arrow_paths=predict_from_arrow_paths,
    )
    fake_eval_prod_models = ModuleType("scripts.eval_prod_models")
    fake_eval_prod_models.resolve_arrow_dataset_paths = lambda *_args: {
        "signatures": "signatures.arrow",
        "clusters": "clusters.arrow",
    }
    fake_eval_prod_models.read_arrow_s2_blocks = lambda _path: {"a smith": ["s1", "s2"]}
    fake_eval_prod_models.read_signature_to_cluster_id = lambda _path: {}

    monkeypatch.setitem(sys.modules, "scripts.eval_prod_models", fake_eval_prod_models)
    monkeypatch.setattr(production_model_module, "load_production_model", lambda _path: fake_clusterer)
    monkeypatch.setattr(model_module, "_ensure_lightgbm_fitted", lambda _classifier: None)
    monkeypatch.setattr(largest_block_cmd, "_write_profile_output", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(largest_block_cmd, "ProcessTreeRSSMonitor", _FakeRSSMonitor)
    monkeypatch.setattr(largest_block_cmd, "build_run_metadata", lambda **_kwargs: {})
    monkeypatch.setattr(largest_block_cmd, "collect_rust_extension_identity", lambda **_kwargs: None)

    result = largest_block_cmd._run_single_arrow(
        backend="rust",
        dataset_name="qian",
        block_key="a smith",
        n_jobs=1,
        profile_output_path=str(tmp_path / "profile.txt"),
        model_path=str(tmp_path / "model"),
        arrow_data_root=str(tmp_path / "arrow"),
        specter_suffix="_specter2.pkl",
        max_block_size=0,
        run_label="rust",
        quality_check=False,
        emit_signature_map=False,
        require_rust_release=False,
        subblocking_threshold=configured_threshold,
    )

    assert captured["batching_threshold"] == expected_applied_threshold
    assert result["subblocking_threshold"] == expected_applied_threshold
