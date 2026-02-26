from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from s2and.consts import PROJECT_ROOT_PATH


def _load_compare_module():
    module_path = Path(PROJECT_ROOT_PATH) / "scripts" / "compare_python_vs_rust.py"
    spec = importlib.util.spec_from_file_location("compare_python_vs_rust", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_language_feature_indices_detect_expected_columns():
    module = _load_compare_module()
    feature_names = [
        "f0",
        "english_count",
        "same_language",
        "language_reliability_count",
        "f4",
    ]
    indices = module._language_feature_indices(feature_names)
    assert indices == [1, 2, 3]


def test_feature_parity_allows_small_language_mismatch():
    module = _load_compare_module()
    feature_names = [
        "first_names_equal",
        "english_count",
        "same_language",
        "language_reliability_count",
        "year_diff",
    ]

    python_features = np.array(
        [
            [1.0, 2.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0, 1.0, 4.0],
        ],
        dtype=np.float64,
    )
    rust_features = python_features.copy()
    rust_features[0, 2] = 0.0

    parity = module._compute_feature_parity(
        python_features,
        rust_features,
        feature_names,
        non_language_rtol=1e-6,
        non_language_atol=1e-6,
        language_max_mismatch_fraction=0.20,
    )

    assert parity["non_language"]["pass"] is True
    assert parity["language"]["pass"] is True
    assert parity["pass"] is True


def test_feature_parity_fails_on_non_language_mismatch():
    module = _load_compare_module()
    feature_names = ["first_names_equal", "english_count", "year_diff"]
    python_features = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    rust_features = np.array([[1.0, 2.0, 3.5]], dtype=np.float64)

    parity = module._compute_feature_parity(
        python_features,
        rust_features,
        feature_names,
        non_language_rtol=1e-6,
        non_language_atol=1e-6,
        language_max_mismatch_fraction=1.0,
    )

    assert parity["non_language"]["pass"] is False
    assert parity["pass"] is False


def test_load_dataset_inputs_force_paths_writes_limited_json(tmp_path):
    module = _load_compare_module()
    dataset = "mini"
    dataset_dir = tmp_path / "data" / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    signatures = {
        "s1": {"signature_id": "s1", "paper_id": 1},
        "s2": {"signature_id": "s2", "paper_id": 2},
        "s3": {"signature_id": "s3", "paper_id": 3},
    }
    papers = {
        "1": {"paper_id": 1, "title": "A"},
        "2": {"paper_id": 2, "title": "B"},
        "3": {"paper_id": 3, "title": "C"},
    }

    with (dataset_dir / f"{dataset}_signatures.json").open("w", encoding="utf-8") as f:
        json.dump(signatures, f)
    with (dataset_dir / f"{dataset}_papers.json").open("w", encoding="utf-8") as f:
        json.dump(papers, f)

    signatures_input, papers_input, tmpdir = module._load_dataset_inputs(
        dataset,
        limit=2,
        project_root=str(tmp_path),
        force_paths=True,
    )

    assert isinstance(signatures_input, str)
    assert isinstance(papers_input, str)
    assert tmpdir is not None
    signatures_path = Path(signatures_input)
    papers_path = Path(papers_input)
    assert signatures_path.exists()
    assert papers_path.exists()

    with signatures_path.open("r", encoding="utf-8") as f:
        signatures_limited = json.load(f)
    with papers_path.open("r", encoding="utf-8") as f:
        papers_limited = json.load(f)

    assert len(signatures_limited) == 2
    assert set(papers_limited.keys()) == {"1", "2"}


def test_build_run_metadata_handles_missing_git(monkeypatch):
    module = _load_compare_module()

    def _raise_file_not_found(*_args, **_kwargs):
        raise FileNotFoundError("git missing")

    monkeypatch.setattr(module.subprocess, "run", _raise_file_not_found)
    metadata = module._build_run_metadata()

    assert metadata["git_commit"] is None
    assert metadata["git_branch"] is None
    assert metadata["git_dirty"] is None
    assert metadata["script"].endswith("compare_python_vs_rust.py")
    assert isinstance(metadata["env"], dict)

