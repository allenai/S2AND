from __future__ import annotations

import importlib.util
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
