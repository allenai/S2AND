from __future__ import annotations

import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

import scripts._rust_suite.compare_cmd as compare_cmd


def test_run_compare_rejects_different_sampled_pairs(monkeypatch) -> None:
    subprocess_results = iter(
        [
            {
                "records_sha256": "same-records",
                "name_counts_sha256": "same-counts",
                "signature_ids_sha256": "same-signatures",
                "pairs_sha256": "python-pairs",
            },
            {
                "records_sha256": "same-records",
                "name_counts_sha256": "same-counts",
                "signature_ids_sha256": "same-signatures",
                "pairs_sha256": "rust-pairs",
            },
        ]
    )
    monkeypatch.setattr(compare_cmd, "_run_subprocess_once", lambda **_kwargs: next(subprocess_results))

    with pytest.raises(RuntimeError, match="same sampled pairs"):
        compare_cmd._run_compare(
            Namespace(
                dataset="dummy",
                limit=2,
                pair_count=1,
                n_jobs=1,
                seed=7,
            )
        )


def test_run_subprocess_surfaces_child_stderr(monkeypatch, tmp_path: Path) -> None:
    def fail(cmd, **_kwargs):
        raise subprocess.CalledProcessError(3, cmd, output="child stdout", stderr="child failure details")

    monkeypatch.setattr(compare_cmd.subprocess, "run", fail)

    with pytest.raises(RuntimeError, match="child failure details"):
        compare_cmd._run_subprocess_once(
            script_path=tmp_path / "compare.py",
            backend="rust",
            features_npy_path=tmp_path / "features.npy",
            args=Namespace(
                dataset="dummy",
                data_root=str(tmp_path),
                limit=2,
                pair_count=1,
                n_jobs=1,
                chunk_size=10,
                seed=7,
                require_non_dev_rust=0,
                require_rust_release=0,
            ),
        )


@pytest.mark.parametrize(
    ("limit", "pair_count", "message"),
    [
        (0, 1, "--limit must be a positive integer"),
        (-1, 1, "--limit must be a positive integer"),
        (2, -1, "--pair-count must be non-negative"),
    ],
)
def test_validate_bounded_args_rejects_unsafe_limits(limit: int, pair_count: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compare_cmd._validate_bounded_args(Namespace(limit=limit, pair_count=pair_count))


def test_bounded_name_count_mappings_use_canonical_keys() -> None:
    mappings = compare_cmd._bounded_name_count_mappings(
        {
            "s1": {"author_info": {"first": "Abdul", "middle": None, "last": "Sattar"}},
            "s2": {"author_info": {"first": "ABDUL", "middle": "X", "last": "Sattar"}},
        }
    )

    assert mappings == (
        {"abdul": 2},
        {"sattar": 2},
        {"abdul sattar": 2},
        {"sattar a": 2},
    )
