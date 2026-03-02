import logging
import os
import random
from contextlib import contextmanager
from itertools import combinations

import pytest

import s2and.featurizer as featurizer_mod
from s2and import feature_port
from s2and.featurizer import _single_pair_featurize
from s2and.subblocking import make_subblocks
from s2and.text import AFFILIATIONS_STOP_WORDS, get_text_ngrams, get_text_ngrams_words
from tests.helpers import build_dummy_dataset, equalish

if not feature_port.rust_signature_preprocess_available():
    pytest.skip("s2and_rust signature preprocessing API is unavailable", allow_module_level=True)


@contextmanager
def _temporary_env(name: str, value: str | None):
    original = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


def _prefilter_affiliation_text(text: str) -> str:
    tokens = [word for word in text.split() if word not in AFFILIATIONS_STOP_WORDS and len(word) > 1]
    return " ".join(tokens)


def _signature_scalar_fields(signature) -> dict[str, object]:
    return {
        "author_info_first_normalized": signature.author_info_first_normalized,
        "author_info_first_normalized_without_apostrophe": signature.author_info_first_normalized_without_apostrophe,
        "author_info_middle_normalized_without_apostrophe": signature.author_info_middle_normalized_without_apostrophe,
        "author_info_last_normalized": signature.author_info_last_normalized,
        "author_info_coauthors": signature.author_info_coauthors,
        "author_info_coauthor_blocks": signature.author_info_coauthor_blocks,
        "author_info_affiliations": signature.author_info_affiliations,
        "author_info_name_counts": signature.author_info_name_counts,
        "author_info_orcid": signature.author_info_orcid,
    }


def _sample_pairs(signature_ids: list[str], limit: int = 8) -> list[tuple[str, str]]:
    pairs = []
    for s1, s2 in combinations(signature_ids, 2):
        pairs.append((s1, s2))
        if len(pairs) >= limit:
            break
    return pairs


@pytest.mark.parametrize(
    "coauthor_text,affiliation_text",
    [
        ("", ""),
        ("Alice Smith Bob Jones", "University of Washington Seattle"),
        ("Renaud Séguier Abdul Sattar", "Department of Computer Science"),
        ("A.B. C-D", "A I lab"),
        (
            " ".join([f"Author{i}" for i in range(80)]),
            " ".join([f"Institute{i}" for i in range(30)]),
        ),
    ],
)
def test_signature_ngrams_batch_rust_parity(coauthor_text: str, affiliation_text: str):
    filtered_affiliation_text = _prefilter_affiliation_text(affiliation_text)
    rust_coauthor, rust_affiliation = feature_port.signature_ngrams_batch_rust(
        [coauthor_text],
        [filtered_affiliation_text],
        num_threads=1,
    )
    assert len(rust_coauthor) == 1
    assert len(rust_affiliation) == 1

    expected_coauthor = get_text_ngrams(coauthor_text, stopwords=None, use_bigrams=True)
    expected_affiliation = get_text_ngrams_words(filtered_affiliation_text, stopwords=set())

    assert rust_coauthor[0] == expected_coauthor
    assert rust_affiliation[0] == expected_affiliation


def test_signature_preprocess_dataset_rust_defers_signature_fields():
    with _temporary_env("S2AND_BACKEND", "python"):
        dataset_python = build_dummy_dataset("dummy_signature_preprocess_python")
    with _temporary_env("S2AND_BACKEND", "rust"):
        dataset_rust = build_dummy_dataset("dummy_signature_preprocess_rust")

    assert set(dataset_python.signatures.keys()) == set(dataset_rust.signatures.keys())
    for signature_id in dataset_python.signatures:
        signature_rust = dataset_rust.signatures[signature_id]
        assert signature_rust.author_info_first_normalized is None
        assert signature_rust.author_info_first_normalized_without_apostrophe is None
        assert signature_rust.author_info_middle_normalized_without_apostrophe is None
        assert signature_rust.author_info_last_normalized is None
        assert signature_rust.author_info_suffix_normalized is None
        assert signature_rust.author_info_coauthors is None
        assert signature_rust.author_info_coauthor_blocks is None
        assert dataset_rust.signatures[signature_id].author_info_affiliations_n_grams is None
        assert dataset_rust.signatures[signature_id].author_info_coauthor_n_grams is None


def test_signature_preprocess_pair_features_and_constraints_parity_with_deferred_signature_fields():
    with _temporary_env("S2AND_BACKEND", "python"):
        dataset_python = build_dummy_dataset("dummy_signature_preprocess_materialize_python")
    with _temporary_env("S2AND_BACKEND", "rust"):
        dataset_rust = build_dummy_dataset("dummy_signature_preprocess_materialize_rust")

    featurizer_mod.global_dataset = dataset_python  # type: ignore
    signature_ids = list(dataset_python.signatures.keys())
    pairs = _sample_pairs(signature_ids, limit=8)
    assert len(pairs) > 0

    for s1, s2 in pairs:
        python_features, _ = _single_pair_featurize((s1, s2))
        rust_features = feature_port.featurize_pair_rust(dataset_rust, s1, s2)
        assert len(python_features) == len(rust_features)
        for idx, (python_value, rust_value) in enumerate(zip(python_features, rust_features, strict=True)):
            assert equalish(python_value, rust_value), (
                f"Feature mismatch for pair ({s1}, {s2}) at idx={idx}: " f"python={python_value} rust={rust_value}"
            )

        python_constraint = dataset_python.get_constraint(s1, s2)
        rust_constraint = feature_port.get_constraint_rust(dataset_rust, s1, s2)
        if python_constraint is None or rust_constraint is None:
            assert python_constraint is None and rust_constraint is None
        else:
            assert python_constraint == rust_constraint


def test_signature_preprocess_lazy_materialization_ngrams_match_python():
    with _temporary_env("S2AND_BACKEND", "python"):
        dataset_python = build_dummy_dataset("dummy_signature_preprocess_materialize_python_ngrams")
    with _temporary_env("S2AND_BACKEND", "rust"):
        dataset_rust = build_dummy_dataset("dummy_signature_preprocess_materialize_rust_ngrams")
    dataset_rust.materialize_signature_ngrams_python()

    for signature_id in dataset_python.signatures:
        signature_python = dataset_python.signatures[signature_id]
        signature_rust = dataset_rust.signatures[signature_id]
        assert signature_python.author_info_affiliations_n_grams == signature_rust.author_info_affiliations_n_grams
        assert signature_python.author_info_coauthor_n_grams == signature_rust.author_info_coauthor_n_grams


def test_subblocking_handles_missing_signature_affiliation_ngrams():
    with _temporary_env("S2AND_BACKEND", "rust"):
        dataset_rust = build_dummy_dataset("dummy_signature_preprocess_subblocking_rust")
    signature_ids = list(dataset_rust.signatures.keys())
    output = make_subblocks(signature_ids, dataset_rust, maximum_size=2)
    assert sum(len(subblock) for subblock in output.values()) == len(signature_ids)


def test_subblocking_membership_parity_python_vs_rust():
    with _temporary_env("S2AND_BACKEND", "python"):
        dataset_python = build_dummy_dataset("dummy_signature_preprocess_subblocking_python")
    with _temporary_env("S2AND_BACKEND", "rust"):
        dataset_rust = build_dummy_dataset("dummy_signature_preprocess_subblocking_rust_parity")

    signature_ids = list(dataset_python.signatures.keys())
    random.seed(12345)
    output_python = make_subblocks(signature_ids, dataset_python, maximum_size=2)
    random.seed(12345)
    output_rust = make_subblocks(signature_ids, dataset_rust, maximum_size=2)

    clusters_python = {tuple(sorted(subblock)) for subblock in output_python.values()}
    clusters_rust = {tuple(sorted(subblock)) for subblock in output_rust.values()}
    assert clusters_python == clusters_rust


def test_signature_preprocess_backend_decision_logged(caplog):
    with _temporary_env("S2AND_BACKEND", "rust"):
        with caplog.at_level(logging.INFO, logger="s2and"):
            build_dummy_dataset("dummy_signature_preprocess_backend_log")

    assert any("Signature preprocessing backend decision:" in record.message for record in caplog.records)


def test_rust_json_ingest_uses_minimal_python_paper_preprocess():
    with _temporary_env("S2AND_BACKEND", "rust"):
        dataset_train = build_dummy_dataset(
            "dummy_signature_preprocess_full_papers",
            mode="train",
            compute_reference_features=True,
        )
    with _temporary_env("S2AND_BACKEND", "rust"):
        dataset_inference = build_dummy_dataset("dummy_signature_preprocess_minimal_papers", mode="inference")

    paper_id = next(iter(dataset_train.papers.keys()))
    train_paper = dataset_train.papers[paper_id]
    inference_paper = dataset_inference.papers[paper_id]

    assert train_paper.title_ngrams_chars is not None
    assert inference_paper.title_ngrams_chars is None
    assert train_paper.title_ngrams_words is not None
    assert inference_paper.title_ngrams_words is None


