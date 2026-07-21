from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts._rust_suite import compare_cmd


def test_feature_parity_rejects_any_language_mismatch():
    feature_names = [
        "first_names_equal",
        "english_count",
        "same_language",
        "language_reliability_min",
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

    parity = compare_cmd._compute_feature_parity(
        python_features,
        rust_features,
        feature_names,
        non_language_rtol=0.0,
        non_language_atol=1e-6,
    )

    assert parity["non_language"]["pass"] is True
    assert parity["language"]["pass"] is False
    assert parity["pass"] is False


def test_feature_parity_fails_on_non_language_or_discrete_mismatch():
    feature_names = ["first_names_equal", "english_count", "year_diff"]
    python_features = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    rust_features = np.array([[1.0, 2.0, 3.5]], dtype=np.float64)

    parity = compare_cmd._compute_feature_parity(
        python_features,
        rust_features,
        feature_names,
        non_language_rtol=0.0,
        non_language_atol=1e-6,
    )

    assert parity["non_language"]["pass"] is False
    assert parity["pass"] is False

    discrete_parity = compare_cmd._compute_feature_parity(
        np.asarray([[1.0]], dtype=np.float64),
        np.asarray([[1.0 + 1e-7]], dtype=np.float64),
        ["english_count"],
        non_language_rtol=0.0,
        non_language_atol=1e-6,
    )

    assert discrete_parity["pass"] is False


def test_load_dataset_inputs_force_paths_writes_limited_json(tmp_path):
    dataset = "mini"
    data_root = tmp_path / "data"
    dataset_dir = data_root / dataset
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

    signatures_input, papers_input, tmpdir = compare_cmd._load_dataset_inputs(
        dataset,
        limit=2,
        data_root=data_root,
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
