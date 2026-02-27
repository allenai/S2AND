from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from s2and.consts import PROJECT_ROOT_PATH


def _load_module():
    module_path = Path(PROJECT_ROOT_PATH) / "scripts" / "rust_suite.py"
    spec = importlib.util.spec_from_file_location("rust_suite", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _touch(path: Path, content: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_effective_train_pairs_size_modes():
    module = _load_module()

    assert module._effective_train_pairs_size(10000, "scaled") == 10000
    assert module._effective_train_pairs_size(0, "scaled") == 1
    assert module._effective_train_pairs_size(10000, "exact_internal") == 100000
    assert module._effective_train_pairs_size(120000, "exact_internal") == 120000

    with pytest.raises(ValueError, match="Unknown train_pairs_size_mode"):
        module._effective_train_pairs_size(1000, "invalid")


def test_resolve_dataset_file_uses_fallback_candidate(tmp_path: Path):
    module = _load_module()
    data_dir = tmp_path / "data"
    _touch(data_dir / "demo" / "signatures.json")

    resolved = module._resolve_dataset_file(str(data_dir), "demo", ["demo_signatures.json", "signatures.json"])
    assert resolved is not None
    assert resolved.endswith("signatures.json")


def test_resolve_dataset_file_raises_when_required_missing(tmp_path: Path):
    module = _load_module()
    data_dir = tmp_path / "data"
    (data_dir / "demo").mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Missing required dataset file"):
        module._resolve_dataset_file(str(data_dir), "demo", ["missing.json"])


def test_build_anddata_kwargs_clustered_dataset(tmp_path: Path):
    module = _load_module()
    data_dir = tmp_path / "data"
    dataset_name = "kisti"
    _touch(data_dir / dataset_name / f"{dataset_name}_signatures.json")
    _touch(data_dir / dataset_name / f"{dataset_name}_papers.json")
    _touch(data_dir / dataset_name / f"{dataset_name}_clusters.json")
    _touch(data_dir / dataset_name / f"{dataset_name}_specter.pickle", content="x")

    kwargs = module._build_anddata_kwargs(
        data_dir=str(data_dir),
        dataset_name=dataset_name,
        n_jobs=4,
        random_seed=1,
        n_train_pairs=10000,
        n_val_test_size=500,
        name_counts={"first_dict": {}, "last_dict": {}, "first_last_dict": {}, "last_first_initial_dict": {}},
        train_pairs_size_mode="scaled",
    )

    assert kwargs["clusters"].endswith(f"{dataset_name}_clusters.json")
    assert kwargs["train_pairs"] is None
    assert kwargs["val_pairs"] is None
    assert kwargs["test_pairs"] is None
    assert kwargs["train_pairs_size"] == 10000
    assert kwargs["n_jobs"] == 4


def test_build_anddata_kwargs_pairwise_only_dataset(tmp_path: Path):
    module = _load_module()
    data_dir = tmp_path / "data"
    dataset_name = "medline"
    _touch(data_dir / dataset_name / f"{dataset_name}_signatures.json")
    _touch(data_dir / dataset_name / f"{dataset_name}_papers.json")
    _touch(data_dir / dataset_name / f"{dataset_name}_specter.pickle", content="x")
    _touch(data_dir / dataset_name / "train_pairs.csv", content="sig1,sig2,label\n")
    _touch(data_dir / dataset_name / "test_pairs.csv", content="sig1,sig2,label\n")
    # intentionally do not create val_pairs.csv to validate optional behavior

    kwargs = module._build_anddata_kwargs(
        data_dir=str(data_dir),
        dataset_name=dataset_name,
        n_jobs=2,
        random_seed=7,
        n_train_pairs=10000,
        n_val_test_size=1000,
        name_counts={"first_dict": {}, "last_dict": {}, "first_last_dict": {}, "last_first_initial_dict": {}},
        train_pairs_size_mode="exact_internal",
    )

    assert kwargs["clusters"] is None
    assert kwargs["train_pairs"].endswith("train_pairs.csv")
    assert kwargs["val_pairs"] is None
    assert kwargs["test_pairs"].endswith("test_pairs.csv")
    # exact_internal keeps transfer_experiment_internal.py behavior
    assert kwargs["train_pairs_size"] == 100000


def test_process_tree_rss_monitor_sample_updates_peak(monkeypatch):
    module = _load_module()
    monitor = module.ProcessTreeRSSMonitor(interval_seconds=1.0)
    gib = 1024**3
    samples = [2 * gib, 5 * gib]

    def _fake_tree_rss():
        return samples.pop(0)

    monkeypatch.setattr(monitor, "_tree_rss_bytes", _fake_tree_rss)

    first_gb = monitor.sample_gb()
    second_gb = monitor.sample_gb()

    assert first_gb == pytest.approx(2.0)
    assert second_gb == pytest.approx(5.0)
    assert monitor.peak_rss_bytes == 5 * gib


def test_build_run_metadata_handles_missing_git(monkeypatch):
    module = _load_module()

    def _raise_file_not_found(*_args, **_kwargs):
        raise FileNotFoundError("git missing")

    monkeypatch.setattr(module.subprocess, "run", _raise_file_not_found)
    metadata = module._build_run_metadata()

    assert metadata["git_commit"] is None
    assert metadata["git_branch"] is None
    assert metadata["git_dirty"] is None
    assert metadata["script"].endswith("rust_suite.py")
    assert isinstance(metadata["env"], dict)


def test_workload_id_stability():
    module = _load_module()
    workload = module._build_workload(
        datasets=["kisti", "arnetminer", "zbmath"],
        target="kisti",
        n_jobs=4,
        n_train_pairs=10000,
        n_iter=5,
        random_seed=1,
        train_pairs_size_mode="scaled",
    )
    same_workload = module._build_workload(
        datasets=["kisti", "arnetminer", "zbmath"],
        target="kisti",
        n_jobs=4,
        n_train_pairs=10000,
        n_iter=5,
        random_seed=1,
        train_pairs_size_mode="scaled",
    )
    changed_workload = module._build_workload(
        datasets=["kisti", "arnetminer", "zbmath"],
        target="kisti",
        n_jobs=4,
        n_train_pairs=12000,
        n_iter=5,
        random_seed=1,
        train_pairs_size_mode="scaled",
    )
    assert module._workload_id(workload) == module._workload_id(same_workload)
    assert module._workload_id(workload) != module._workload_id(changed_workload)


def test_preset_resolution_smoke_defaults_and_overrides():
    module = _load_module()
    transfer_module = module._load_internal_module("transfer_mini")

    default_args = argparse.Namespace(
        preset="smoke",
        datasets=None,
        target=None,
        n_jobs=None,
        n_train_pairs=None,
        n_iter=None,
        random_seed=None,
        train_pairs_size_mode=None,
    )
    smoke_workload, _ = transfer_module._resolve_workload(default_args)
    assert smoke_workload["datasets"] == ["kisti"]
    assert smoke_workload["target"] == "kisti"
    assert smoke_workload["n_jobs"] == 2
    assert smoke_workload["n_train_pairs"] == 300
    assert smoke_workload["n_iter"] == 1

    override_args = argparse.Namespace(
        preset="full",
        datasets=["kisti", "zbmath"],
        target="zbmath",
        n_jobs=3,
        n_train_pairs=777,
        n_iter=2,
        random_seed=9,
        train_pairs_size_mode="exact_internal",
    )
    full_override_workload, _ = transfer_module._resolve_workload(override_args)
    assert full_override_workload["datasets"] == ["kisti", "zbmath"]
    assert full_override_workload["target"] == "zbmath"
    assert full_override_workload["n_jobs"] == 3
    assert full_override_workload["n_train_pairs"] == 777
    assert full_override_workload["n_iter"] == 2
    assert full_override_workload["random_seed"] == 9
    assert full_override_workload["train_pairs_size_mode"] == "exact_internal"


def test_gate_mode_rejects_mismatched_workload_ids(tmp_path: Path):
    module = _load_module()
    transfer_module = module._load_internal_module("transfer_mini")

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    baseline_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "run_label": "rust",
                        "workload_id": "aaa",
                        "total_seconds": 10.0,
                        "peak_rss_gb": 1.0,
                        "b3": [0.9, 0.9, 0.9],
                        "stage_timings": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    current_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "run_label": "rust",
                        "workload_id": "bbb",
                        "total_seconds": 9.0,
                        "peak_rss_gb": 0.9,
                        "b3": [0.9, 0.9, 0.9],
                        "stage_timings": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    args = argparse.Namespace(
        baseline_json=str(baseline_path),
        current_json=str(current_path),
        gate_run_label="rust",
        max_runtime_regression_fraction=0.05,
        max_peak_rss_regression_fraction=0.05,
        max_b3_f1_drop=0.001,
        max_stage_regression_fraction=0.1,
        write_json="",
    )
    with pytest.raises(RuntimeError, match="Workload mismatch"):
        transfer_module._gate(args)
