import sys

import pytest

from scripts import tutorial_for_predicting_with_the_prod_model as tutorial


def test_tutorial_desired_memory_requires_batched_json_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tutorial_for_predicting_with_the_prod_model.py",
            "--model-path",
            "unused-complete-bundle",
            "--desired-memory-use",
            "1000",
        ],
    )

    with pytest.raises(ValueError, match="--desired-memory-use requires --batching-threshold"):
        tutorial.main()
