from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from s2and.consts import PROJECT_ROOT_PATH


def _load_profile_module():
    module_path = Path(PROJECT_ROOT_PATH) / "scripts" / "rust_suite.py"
    spec = importlib.util.spec_from_file_location("rust_suite", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_s2and_modules(monkeypatch, project_root: Path) -> None:
    consts_mod = ModuleType("s2and.consts")
    consts_mod.PROJECT_ROOT_PATH = str(project_root)

    class _DummyANDData:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    def _dummy_cluster_eval(anddata, clusterer, split, use_s2_clusters):
        del anddata, clusterer, split, use_s2_clusters
        return (
            {
                "B3 (P, R, F1)": (0.9, 0.9, 0.9),
                "Cluster (P, R F1)": (0.8, 0.8, 0.8),
                "Cluster Macro (P, R, F1)": (0.7, 0.7, 0.7),
            },
            None,
        )

    def _dummy_load_model(_path):
        return {"clusterer": SimpleNamespace(use_cache=False, n_jobs=1)}

    data_mod = ModuleType("s2and.data")
    data_mod.ANDData = _DummyANDData

    eval_mod = ModuleType("s2and.eval")
    eval_mod.cluster_eval = _dummy_cluster_eval

    serialization_mod = ModuleType("s2and.serialization")
    serialization_mod.load_pickle_with_verified_label_encoder_compat = _dummy_load_model

    monkeypatch.setitem(sys.modules, "s2and.consts", consts_mod)
    monkeypatch.setitem(sys.modules, "s2and.data", data_mod)
    monkeypatch.setitem(sys.modules, "s2and.eval", eval_mod)
    monkeypatch.setitem(sys.modules, "s2and.serialization", serialization_mod)


def test_single_run_sets_env_and_returns_result(tmp_path, monkeypatch):
    module = _load_profile_module()
    _install_fake_s2and_modules(monkeypatch, tmp_path)

    dataset_dir = tmp_path / "data" / "s2and_mini" / "kisti"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["signatures.json", "papers.json", "clusters.json", "specter.pickle"]:
        (dataset_dir / f"kisti_{suffix}").write_text("{}", encoding="utf-8")

    profile_output_path = tmp_path / "profile.txt"
    result = module._single_run(
        backend="rust",
        dataset_name="kisti",
        n_jobs=2,
        profile_output_path=str(profile_output_path),
    )

    assert result["backend"] == "rust"
    assert result["run_metadata"]["script"].endswith("rust_suite.py")
    assert isinstance(result["run_metadata"]["env"], dict)


def test_run_single_subprocess_passes_flags_and_parses_result(monkeypatch):
    module = _load_profile_module()
    captured_cmd: list[str] = []
    payload = {
        "backend": "rust",
        "backend_label": "rust_from_dataset",
        "dataset": "kisti",
        "n_jobs": 4,
        "total_latency_seconds": 1.0,
        "anddata_build_seconds": 0.5,
        "prediction_seconds": 0.5,
        "peak_rss_gb": 1.0,
        "b3": [0.9, 0.9, 0.9],
        "cluster": [0.8, 0.8, 0.8],
        "cluster_macro": [0.7, 0.7, 0.7],
        "profile_output_path": "dummy.txt",
        "raw_cluster_metrics": {},
    }

    def _fake_run(cmd, capture_output, text, check):
        del capture_output, text, check
        captured_cmd.extend(cmd)
        stdout = (
            f"{module.RESULT_JSON_START}\n"
            f"{json.dumps(payload)}\n"
            f"{module.RESULT_JSON_END}\n"
        )
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    module._run_single_subprocess(
        script_path=Path("scripts/rust_suite.py"),
        backend="rust",
        dataset_name="kisti",
        n_jobs=4,
        profile_output_path="scratch/profile.txt",
        model_path="data/production_model_v1.2.pickle",
        data_root="data",
        specter_file="kisti_specter2.pkl",
        rust_warm_featurizer_before_predict=1,
        single_write_json="scratch/single.json",
        run_label="rust_from_dataset",
    )

    assert "--model-path" in captured_cmd
    assert "--data-root" in captured_cmd
    assert "--specter-file" in captured_cmd
    assert "--rust-warm-featurizer-before-predict" in captured_cmd
    assert "--single-write-json" in captured_cmd


def test_build_data_paths_honors_custom_root_and_specter(tmp_path):
    module = _load_profile_module()
    paths = module._build_data_paths(str(tmp_path), "inventors_s2and", "data", "inventors_s2and_specter2.pkl")
    assert paths["signatures"] == str(tmp_path / "data" / "inventors_s2and" / "inventors_s2and_signatures.json")
    assert paths["specter"] == str(tmp_path / "data" / "inventors_s2and" / "inventors_s2and_specter2.pkl")


def test_single_run_warms_featurizer_when_requested(tmp_path, monkeypatch):
    module = _load_profile_module()
    _install_fake_s2and_modules(monkeypatch, tmp_path)

    feature_port_mod = ModuleType("s2and.feature_port")
    warm_calls: list[object] = []

    def _warm_rust_featurizer(dataset):
        warm_calls.append(dataset)

    feature_port_mod.warm_rust_featurizer = _warm_rust_featurizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "s2and.feature_port", feature_port_mod)

    dataset_dir = tmp_path / "data" / "s2and_mini" / "kisti"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["signatures.json", "papers.json", "clusters.json", "specter.pickle"]:
        (dataset_dir / f"kisti_{suffix}").write_text("{}", encoding="utf-8")

    profile_output_path = tmp_path / "profile.txt"
    result = module._single_run(
        backend="rust",
        dataset_name="kisti",
        n_jobs=1,
        profile_output_path=str(profile_output_path),
        rust_warm_featurizer_before_predict=1,
    )

    assert len(warm_calls) == 1
    assert result["rust_warm_featurizer_before_predict"] == 1
    assert result["rust_warm_featurizer_seconds"] >= 0.0
    assert result["run_metadata"]["script"].endswith("rust_suite.py")


def test_build_run_metadata_handles_missing_git(monkeypatch):
    module = _load_profile_module()

    def _raise_file_not_found(*_args, **_kwargs):
        raise FileNotFoundError("git missing")

    monkeypatch.setattr(module.subprocess, "run", _raise_file_not_found)
    metadata = module._build_run_metadata()

    assert metadata["git_commit"] is None
    assert metadata["git_branch"] is None
    assert metadata["git_dirty"] is None
    assert metadata["script"].endswith("rust_suite.py")
