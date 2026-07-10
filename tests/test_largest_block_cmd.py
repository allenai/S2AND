from __future__ import annotations

from pathlib import Path

import pytest

from scripts._rust_suite import largest_block_cmd


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


def test_sample_unique_pair_indices_never_duplicates_pairs() -> None:
    pairs = largest_block_cmd._sample_unique_pair_indices(6, 12, largest_block_cmd.random.Random(7))  # noqa: SLF001

    assert len(pairs) == 12
    assert len(set(pairs)) == 12
    assert all(0 <= i < j < 6 for i, j in pairs)


def test_sample_unique_pair_indices_returns_all_pairs_when_sample_exceeds_population() -> None:
    pairs = largest_block_cmd._sample_unique_pair_indices(4, 99, largest_block_cmd.random.Random(7))  # noqa: SLF001

    assert pairs == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
