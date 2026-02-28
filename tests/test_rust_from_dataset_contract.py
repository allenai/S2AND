from __future__ import annotations

import copy
import math
import os
from collections import Counter, namedtuple

import pytest

import s2and.featurizer as featurizer_mod
from s2and.data import ANDData, Author, NameCounts
from s2and.featurizer import _single_pair_featurize


def _import_s2and_rust():
    try:
        import s2and_rust

        rust_featurizer = getattr(s2and_rust, "RustFeaturizer", None)
        if rust_featurizer is None or not hasattr(rust_featurizer, "from_dataset"):
            return False, None
        return True, s2and_rust
    except Exception:
        return False, None


HAS_RUST, s2and_rust = _import_s2and_rust()
if not HAS_RUST:
    pytest.skip("s2and_rust RustFeaturizer.from_dataset is unavailable", allow_module_level=True)


def _equalish(a: float, b: float, rel_tol: float = 1e-6, abs_tol: float = 1e-3) -> bool:
    if math.isnan(float(a)) and math.isnan(float(b)):
        return True
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


def _build_minimal_dataset(name: str) -> ANDData:
    os.environ.setdefault("S2AND_SKIP_FASTTEXT", "1")
    signatures = {
        "s1": {
            "signature_id": "s1",
            "paper_id": 1,
            "author_info": {
                "position": 0,
                "block": "a_smith",
                "first": "Alice",
                "middle": "Q",
                "last": "Smith",
                "suffix": None,
                "email": "alice@uni.edu",
                "affiliations": ["Alpha Institute"],
                "given_block": "a_smith",
            },
        },
        "s2": {
            "signature_id": "s2",
            "paper_id": 2,
            "author_info": {
                "position": 0,
                "block": "a_smith",
                "first": "Alice",
                "middle": "",
                "last": "Smith",
                "suffix": None,
                "email": "alice@uni.edu",
                "affiliations": ["Beta Lab"],
                "given_block": "a_smith",
            },
        },
    }
    papers = {
        "1": {
            "paper_id": 1,
            "title": "This paper presents a method for author disambiguation in digital libraries.",
            "abstract": "A",
            "authors": [
                {"author_name": "Alice Smith", "position": 0},
                {"author_name": "Bob Jones", "position": 1},
            ],
            "venue": "Conf A",
            "journal_name": "Journal X",
            "year": 2020,
            "references": [],
        },
        "2": {
            "paper_id": 2,
            "title": (
                "Cet article presente une methode pour la desambiguation des auteurs dans les "
                "bibliotheques numeriques."
            ),
            "abstract": "B",
            "authors": [
                {"author_name": "Alice Smith", "position": 0},
                {"author_name": "Carol Lee", "position": 1},
            ],
            "venue": "Conf A",
            "journal_name": "Journal Y",
            "year": 2021,
            "references": [],
        },
    }
    clusters = {"c1": {"cluster_id": "c1", "signature_ids": ["s1", "s2"], "model_version": -1}}

    prior_backend = os.environ.get("S2AND_BACKEND")
    os.environ["S2AND_BACKEND"] = "python"
    try:
        return ANDData(
            signatures=signatures,
            papers=papers,
            name=name,
            mode="train",
            specter_embeddings=None,
            clusters=clusters,
            cluster_seeds=None,
            block_type="s2",
            train_pairs=None,
            val_pairs=None,
            test_pairs=None,
            train_pairs_size=10,
            val_pairs_size=10,
            test_pairs_size=10,
            n_jobs=1,
            load_name_counts=False,
            preprocess=True,
            random_seed=42,
            name_tuples="filtered",
            use_orcid_id=True,
            use_sinonym_overwrite=False,
            compute_reference_features=False,
        )
    finally:
        if prior_backend is None:
            os.environ.pop("S2AND_BACKEND", None)
        else:
            os.environ["S2AND_BACKEND"] = prior_backend


@pytest.fixture(autouse=True)
def _cleanup_global_dataset():
    """Restore featurizer_mod.global_dataset after each test."""
    original = featurizer_mod.global_dataset
    yield
    featurizer_mod.global_dataset = original


def test_from_dataset_fastpath_parity_for_field_sensitive_values():
    dataset = _build_minimal_dataset("rust_contract_fastpath_parity")

    dataset.papers["1"] = dataset.papers["1"]._replace(
        venue_ngrams=Counter({"shared_venue": 2}),
        journal_ngrams=Counter({"journal_only_left": 1}),
        reference_details=(Counter(), Counter(), Counter(), Counter()),
    )
    dataset.papers["2"] = dataset.papers["2"]._replace(
        venue_ngrams=Counter({"shared_venue": 4}),
        journal_ngrams=Counter({"journal_only_right": 1}),
        reference_details=(Counter(), Counter(), Counter(), Counter()),
    )

    dataset.signatures["s1"] = dataset.signatures["s1"]._replace(
        author_info_coauthors={"bob jones"},
        author_info_coauthor_blocks={"b_jones"},
        author_info_affiliations_n_grams=Counter({"alpha": 1}),
        author_info_coauthor_n_grams=Counter({"bob": 1}),
        author_info_name_counts=NameCounts(11.0, 21.0, 31.0, 41.0),
    )
    dataset.signatures["s2"] = dataset.signatures["s2"]._replace(
        author_info_coauthors={"carol lee"},
        author_info_coauthor_blocks={"c_lee"},
        author_info_affiliations_n_grams=Counter({"beta": 1}),
        author_info_coauthor_n_grams=Counter({"carol": 1}),
        author_info_name_counts=NameCounts(13.0, 23.0, 33.0, 43.0),
    )

    featurizer_mod.global_dataset = dataset  # type: ignore
    python_features, _ = _single_pair_featurize(("s1", "s2"))
    rust_featurizer = s2and_rust.RustFeaturizer.from_dataset(dataset, 0.0, 10000.0, 1)
    rust_features = rust_featurizer.featurize_pair("s1", "s2")
    feature_names = featurizer_mod.FeaturizationInfo().get_feature_names()
    venue_idx = feature_names.index("venue_overlap")
    journal_idx = feature_names.index("journal_overlap")

    assert python_features[venue_idx] == pytest.approx(0.5)
    assert python_features[journal_idx] == pytest.approx(0.0)
    assert len(python_features) == len(rust_features)
    for idx, (ref_val, got_val) in enumerate(zip(python_features, rust_features, strict=False)):
        assert _equalish(ref_val, got_val), f"Mismatch idx={idx}: ref={ref_val} got={got_val}"


def test_from_dataset_raw_papers_match_preprocessed_for_language_and_coauthors():
    dataset_preprocessed = _build_minimal_dataset("rust_contract_raw_vs_preprocessed")
    for signature_id in ("s1", "s2"):
        dataset_preprocessed.signatures[signature_id] = dataset_preprocessed.signatures[signature_id]._replace(
            author_info_coauthors=None,
            author_info_coauthor_blocks=None,
            author_info_coauthor_n_grams=None,
        )

    rust_preprocessed = s2and_rust.RustFeaturizer.from_dataset(dataset_preprocessed, 0.0, 10000.0, 1)
    expected_features = rust_preprocessed.featurize_pair("s1", "s2")
    expected_constraint = rust_preprocessed.get_constraint("s1", "s2")

    dataset_raw = copy.deepcopy(dataset_preprocessed)
    dataset_raw.preprocess = True
    dataset_raw.papers["1"] = dataset_raw.papers["1"]._replace(
        title="This paper presents a method for author disambiguation in digital libraries.",
        authors=[Author(author_name="Alice Smith", position=0), Author(author_name="Bob Jones", position=1)],
        predicted_language=None,
        is_reliable=None,
        title_ngrams_words=None,
        title_ngrams_chars=None,
        venue_ngrams=None,
        journal_ngrams=None,
    )
    dataset_raw.papers["2"] = dataset_raw.papers["2"]._replace(
        title="Cet article presente une methode pour la desambiguation des auteurs dans les bibliotheques numeriques.",
        authors=[Author(author_name="Alice Smith", position=0), Author(author_name="Carol Lee", position=1)],
        predicted_language=None,
        is_reliable=None,
        title_ngrams_words=None,
        title_ngrams_chars=None,
        venue_ngrams=None,
        journal_ngrams=None,
    )

    rust_raw = s2and_rust.RustFeaturizer.from_dataset(dataset_raw, 0.0, 10000.0, 1)
    observed_features = rust_raw.featurize_pair("s1", "s2")
    observed_constraint = rust_raw.get_constraint("s1", "s2")

    assert len(expected_features) == len(observed_features)
    for idx, (expected, observed) in enumerate(zip(expected_features, observed_features, strict=False)):
        assert _equalish(expected, observed), f"Mismatch idx={idx}: expected={expected} observed={observed}"
    assert _equalish(expected_constraint, observed_constraint)
    assert not math.isnan(float(expected_constraint))


def test_from_dataset_rejects_namedtuple_field_order_mismatch():
    dataset = _build_minimal_dataset("rust_contract_field_order_mismatch")
    paper_fields = list(dataset.papers["1"]._fields)
    venue_index = paper_fields.index("venue_ngrams")
    journal_index = paper_fields.index("journal_ngrams")
    paper_fields[venue_index], paper_fields[journal_index] = paper_fields[journal_index], paper_fields[venue_index]
    SwappedPaper = namedtuple("SwappedPaper", paper_fields)

    for paper_id, paper in list(dataset.papers.items()):
        swapped_values = [getattr(paper, field_name) for field_name in paper_fields]
        dataset.papers[paper_id] = SwappedPaper(*swapped_values)

    with pytest.raises(ValueError, match="Paper fast-path contract mismatch"):
        s2and_rust.RustFeaturizer.from_dataset(dataset, 0.0, 10000.0, 1)
