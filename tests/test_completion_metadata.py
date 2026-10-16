"""First-name preparation at the incremental completion boundary."""

import json
from pathlib import Path

import pytest

from s2and.data import ANDData
from s2and.incremental_linking.completion_metadata import SignatureFirstNames, signature_first


@pytest.mark.parametrize(
    ("raw_first", "expected"),
    [("Dr. Jean–Paul Henry", "jean paul"), (None, "")],
)
def test_completion_first_canonicalizes_unprepared_names(raw_first: str | None, expected: str) -> None:
    signatures = json.loads(Path("tests/dummy/signatures.json").read_text(encoding="utf-8"))
    signatures["0"]["author_info"]["first"] = raw_first
    dataset = ANDData(
        signatures=signatures,
        papers="tests/dummy/papers.json",
        name="completion-unprepared-first",
        mode="inference",
        preprocess=False,
        name_counts_index=None,
        name_tuples=set(),
    )

    assert dataset.signatures["0"].author_info_first_normalized_without_apostrophe is None
    assert signature_first(dataset.signatures["0"]) == expected
    assert SignatureFirstNames(dataset.signatures)["0"] == expected


@pytest.mark.parametrize("prepared_first", ["", "prepared"])
def test_completion_first_preserves_prepared_name(prepared_first: str) -> None:
    dataset = ANDData(
        signatures="tests/dummy/signatures.json",
        papers="tests/dummy/papers.json",
        name="completion-prepared-first",
        mode="inference",
        preprocess=False,
        name_counts_index=None,
        name_tuples=set(),
    )
    signature = dataset.signatures["0"]._replace(
        author_info_first="Ada", author_info_first_normalized_without_apostrophe=prepared_first
    )

    assert signature_first(signature) == prepared_first
