"""Embedding refresh saves progress at a bounded row interval for every batch size."""

from __future__ import annotations

from scripts.production.data._pairwise_embeddings import CHECKPOINT_ROWS, checkpoint_due


def _saved_rows(batch_size: int, rows: int, completed: int = 0) -> list[int]:
    saved, checkpoints = completed, []
    for start in range(completed, rows, batch_size):
        stop = min(start + batch_size, rows)
        if checkpoint_due(stop, saved, rows):
            saved = stop
            checkpoints.append(stop)
    return checkpoints


def test_progress_saves_within_one_interval_and_at_completion() -> None:
    batch_size = 24
    rows = 5_000
    checkpoints = _saved_rows(batch_size, rows, completed=3_000)
    assert checkpoints[0] == 3_000 + 43 * batch_size
    assert checkpoints[-1] == rows
    gaps = [later - earlier for earlier, later in zip([3_000, *checkpoints], checkpoints, strict=False)]
    assert max(gaps) < CHECKPOINT_ROWS + batch_size
