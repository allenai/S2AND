from __future__ import annotations

import logging
from collections import Counter

from s2and.data import _ordered_coauthors_for_signature
from tests.helpers import build_dummy_dataset


def test_materialize_signature_ngrams_python_missing_paper_warns_and_uses_empty_coauthors(caplog):
    dataset = build_dummy_dataset("dummy_materialize_missing_paper")
    signature_id = next(iter(dataset.signatures.keys()))
    signature = dataset.signatures[signature_id]

    removed_paper = dataset.papers.pop(str(signature.paper_id), None)
    assert removed_paper is not None

    dataset.signatures[signature_id] = signature._replace(
        author_info_affiliations_n_grams=None,
        author_info_coauthor_n_grams=None,
    )

    with caplog.at_level(logging.WARNING, logger="s2and"):
        dataset.materialize_signature_ngrams_python(batch_size=1)

    updated_signature = dataset.signatures[signature_id]
    assert updated_signature.author_info_coauthor_n_grams == Counter()
    assert updated_signature.author_info_affiliations_n_grams is not None

    warning_messages = [record.message for record in caplog.records]
    assert any("Missing paper for signature ngram materialization" in message for message in warning_messages)
    assert any(f"signature_id={signature_id}" in message for message in warning_messages)
    assert any(f"paper_id={signature.paper_id}" in message for message in warning_messages)


def test_ordered_coauthors_missing_paper_returns_empty_and_warns(caplog):
    dataset = build_dummy_dataset("dummy_ordered_coauthors_missing_paper")
    signature_id = next(iter(dataset.signatures.keys()))
    signature = dataset.signatures[signature_id]

    removed_paper = dataset.papers.pop(str(signature.paper_id), None)
    assert removed_paper is not None

    with caplog.at_level(logging.WARNING, logger="s2and"):
        coauthors = _ordered_coauthors_for_signature(signature, dataset.papers)

    assert coauthors == []
    warning_messages = [record.message for record in caplog.records]
    assert any("Missing paper for signature ngram materialization" in message for message in warning_messages)
    assert any(f"signature_id={signature_id}" in message for message in warning_messages)
    assert any(f"paper_id={signature.paper_id}" in message for message in warning_messages)
