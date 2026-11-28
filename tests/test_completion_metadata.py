"""First-name preparation at the incremental completion boundary."""

import json
from pathlib import Path

from s2and.data import ANDData
from s2and.incremental_linking.completion_metadata import SignatureFirstNames, signature_first


def test_completion_first_uses_prepared_names_and_canonicalizes_missing_values() -> None:
    signatures = json.loads(Path("tests/dummy/signatures.json").read_text(encoding="utf-8"))
    signatures["0"]["author_info"]["first"] = "Dr. Jean–Paul Henry"
    signatures["1"]["author_info"]["first"] = None
    dataset = ANDData(
        signatures=signatures,
        papers="tests/dummy/papers.json",
        name="completion-unprepared-first",
        mode="inference",
        preprocess=False,
        name_counts_index=None,
        name_tuples=set(),
    )
    assert dataset.signatures["0"].author_info_first_normalized is None
    first_names = SignatureFirstNames(dataset.signatures)
    assert (first_names["0"], first_names["1"]) == ("jean paul", "")
    assert signature_first(dataset.signatures["0"]._replace(author_info_first_normalized="")) == ""
    assert signature_first(dataset.signatures["0"]._replace(author_info_first_normalized="prepared")) == "prepared"
