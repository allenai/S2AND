"""Embedding refresh saves progress at a bounded row interval for every batch size."""

from __future__ import annotations

import pytest

from scripts.production.data._pairwise_embeddings import CHECKPOINT_ROWS, checkpoint_due


def _saved_rows(batch_size: int, rows: int, completed: int = 0) -> list[int]:
    saved, checkpoints = completed, []
    for start in range(completed, rows, batch_size):
        stop = min(start + batch_size, rows)
        if checkpoint_due(stop, saved, rows):
            saved = stop
            checkpoints.append(stop)
    return checkpoints


@pytest.mark.parametrize("batch_size", [3, 16, 24, 1000, 1024, 5000])
def test_progress_saves_within_one_interval_and_at_completion(batch_size: int) -> None:
    rows = 10_000
    checkpoints = _saved_rows(batch_size, rows)
    assert checkpoints[-1] == rows
    gaps = [later - earlier for earlier, later in zip([0, *checkpoints], checkpoints, strict=False)]
    assert max(gaps) < CHECKPOINT_ROWS + batch_size


def test_resumed_progress_counts_from_the_saved_row() -> None:
    assert _saved_rows(24, 5_000, completed=3_000)[0] == 3_000 + 43 * 24
