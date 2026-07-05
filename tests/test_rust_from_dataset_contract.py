from __future__ import annotations

import copy
import math
import os
from collections import Counter
from typing import Any, cast

import numpy as np
import pytest

import s2and.featurizer as featurizer_mod
import s2and.text as text_mod
from s2and.data import ANDData, Author, NameCounts
from s2and.featurizer import _single_pair_featurize
from tests.helpers import equalish, import_s2and_rust

HAS_RUST, s2and_rust = import_s2and_rust(required_method="from_dataset")
if not HAS_RUST:
    raise pytest.skip.Exception("s2and_rust RustFeaturizer.from_dataset is unavailable", allow_module_level=True)
assert s2and_rust is not None and not isinstance(s2and_rust, Exception)
_S2AND_RUST = cast(Any, s2and_rust)


def _build_minimal_dataset(name: str) -> ANDData:
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
            compute_reference_features=False,
        )
    finally:
        if prior_backend is None:
            os.environ.pop("S2AND_BACKEND", None)
        else:
            os.environ["S2AND_BACKEND"] = prior_backend


def _rust_minimal_featurizer(name: str) -> tuple[Any, int, int]:
    dataset = _build_minimal_dataset(name)
    rust_featurizer = _S2AND_RUST.RustFeaturizer.from_dataset(dataset, 0.0, 10000.0, 1)
    signature_id_to_index = _signature_id_to_index(rust_featurizer)
    return rust_featurizer, signature_id_to_index["s1"], signature_id_to_index["s2"]


def _signature_id_to_index(rust_featurizer: Any) -> dict[str, int]:
    return {str(signature_id): index for index, signature_id in enumerate(rust_featurizer.signature_ids())}


def _indexed_pair_matrix(
    rust_featurizer: Any,
    left_signature_id: str,
    right_signature_id: str,
    selected_indices: list[int] | None = None,
) -> np.ndarray:
    signature_id_to_index = _signature_id_to_index(rust_featurizer)
    return np.asarray(
        rust_featurizer.featurize_pairs_matrix_indexed(
            [(signature_id_to_index[left_signature_id], signature_id_to_index[right_signature_id])],
            selected_indices,
            1,
            np.nan,
        )
    )


def _indexed_constraint(rust_featurizer: Any, left_signature_id: str, right_signature_id: str) -> float | None:
    signature_id_to_index = _signature_id_to_index(rust_featurizer)
    return rust_featurizer.get_constraints_matrix_indexed(
        [(signature_id_to_index[left_signature_id], signature_id_to_index[right_signature_id])]
    )[0]


def test_from_dataset_requires_boundary_option_attributes() -> None:
    dataset = _build_minimal_dataset("rust_contract_requires_boundary_options")
    delattr(dataset, "compute_reference_features")

    with pytest.raises(AttributeError, match="compute_reference_features"):
        _S2AND_RUST.RustFeaturizer.from_dataset(dataset, 0.0, 10000.0, 1)


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

    python_features, _ = _single_pair_featurize(("s1", "s2"), dataset=dataset)
    rust_featurizer = _S2AND_RUST.RustFeaturizer.from_dataset(dataset, 0.0, 10000.0, 1)
    rust_features = _indexed_pair_matrix(rust_featurizer, "s1", "s2")[0]
    feature_names = featurizer_mod.FeaturizationInfo().get_feature_names()
    venue_idx = feature_names.index("venue_overlap")
    journal_idx = feature_names.index("journal_overlap")

    assert python_features[venue_idx] == pytest.approx(0.5)
    assert python_features[journal_idx] == pytest.approx(0.0)
    assert len(python_features) == len(rust_features)
    for idx, (ref_val, got_val) in enumerate(zip(python_features, rust_features, strict=True)):
        assert equalish(ref_val, got_val), f"Mismatch idx={idx}: ref={ref_val} got={got_val}"


def test_from_dataset_raw_papers_match_preprocessed_for_language_names_and_ngrams():
    dataset_preprocessed = _build_minimal_dataset("rust_contract_raw_vs_preprocessed")

    rust_preprocessed = _S2AND_RUST.RustFeaturizer.from_dataset(dataset_preprocessed, 0.0, 10000.0, 1)
    expected_features = _indexed_pair_matrix(rust_preprocessed, "s1", "s2")[0]
    expected_constraint = _indexed_constraint(rust_preprocessed, "s1", "s2")

    dataset_raw = copy.deepcopy(dataset_preprocessed)
    dataset_raw.preprocess = True
    dataset_raw.papers["1"] = dataset_raw.papers["1"]._replace(
        title="This paper presents a method for author disambiguation in digital libraries.",
        authors=[Author(author_name="Alice Smith", position=0), Author(author_name="Bob Jones", position=1)],
        venue="Conf A",
        journal_name="Journal X",
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
        venue="Conf A",
        journal_name="Journal Y",
        predicted_language=None,
        is_reliable=None,
        title_ngrams_words=None,
        title_ngrams_chars=None,
        venue_ngrams=None,
        journal_ngrams=None,
    )
    raw_affiliations = {"s1": ["Alpha Institute"], "s2": ["Beta Lab"]}
    for signature_id in ("s1", "s2"):
        dataset_raw.signatures[signature_id] = dataset_raw.signatures[signature_id]._replace(
            author_info_first_normalized_without_apostrophe=None,
            author_info_first_normalized=None,
            author_info_middle_normalized_without_apostrophe=None,
            author_info_last_normalized=None,
            author_info_suffix_normalized=None,
            author_info_affiliations=raw_affiliations[signature_id],
            author_info_affiliations_n_grams=None,
            author_info_coauthors=None,
            author_info_coauthor_blocks=None,
            author_info_coauthor_n_grams=None,
        )

    for paper in dataset_raw.papers.values():
        assert paper.predicted_language is None
        assert paper.title_ngrams_words is None
        assert paper.title_ngrams_chars is None
        assert paper.venue_ngrams is None
        assert paper.journal_ngrams is None
    for signature in dataset_raw.signatures.values():
        assert signature.author_info_first_normalized_without_apostrophe is None
        assert signature.author_info_last_normalized is None
        assert signature.author_info_affiliations_n_grams is None
        assert signature.author_info_coauthor_n_grams is None

    rust_raw = _S2AND_RUST.RustFeaturizer.from_dataset(dataset_raw, 0.0, 10000.0, 1)
    observed_features = _indexed_pair_matrix(rust_raw, "s1", "s2")[0]
    observed_constraint = _indexed_constraint(rust_raw, "s1", "s2")

    assert len(expected_features) == len(observed_features)
    for idx, (expected, observed) in enumerate(zip(expected_features, observed_features, strict=True)):
        assert equalish(expected, observed), f"Mismatch idx={idx}: expected={expected} observed={observed}"
    if expected_constraint is None or observed_constraint is None:
        assert expected_constraint is None
        assert observed_constraint is None
    else:
        assert equalish(expected_constraint, observed_constraint)
        assert not math.isnan(float(expected_constraint))


def test_from_dataset_raw_language_detection_uses_fasttext_sensitive_outputs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("S2AND_SKIP_FASTTEXT", raising=False)
    text_mod.set_fasttext_loading_enabled(True)
    # fastText is mandatory in production, so an absent model now raises rather
    # than returning None; skip this parity check when the model is unavailable.
    try:
        fasttext_model = text_mod._get_fasttext_model()
    except RuntimeError:
        pytest.skip("fastText language model is unavailable")
    if fasttext_model is None:
        pytest.skip("fastText language model is unavailable")

    dataset_raw = _build_minimal_dataset("rust_contract_raw_fasttext_language")
    # Titles long/clear enough that fastText AND cld2 agree, so detection is
    # reliable under the "both detectors must agree" rule (short titles collapse
    # to 'un' because cld2 abstains). This keeps the English-vs-German
    # distinction the test relies on while exercising fastText sensitivity.
    raw_titles = {
        "1": "A neural network approach to automatic speech recognition",
        "2": "Ein neuronaler Netzwerkansatz zur automatischen Spracherkennung",
    }
    expected_language = {}
    for paper_id, title in raw_titles.items():
        is_reliable, _is_english, predicted_language = text_mod.detect_language(title)
        expected_language[paper_id] = (predicted_language, is_reliable)
        dataset_raw.papers[paper_id] = dataset_raw.papers[paper_id]._replace(
            title=title,
            predicted_language=None,
            is_reliable=None,
            title_ngrams_words=None,
            title_ngrams_chars=None,
        )

    assert expected_language == {"1": ("en", True), "2": ("de", True)}

    rust_raw = _S2AND_RUST.RustFeaturizer.from_dataset(dataset_raw, 0.0, 10000.0, 1)
    feature_names = featurizer_mod.FeaturizationInfo().get_feature_names()
    english_count_idx = feature_names.index("english_count")
    same_language_idx = feature_names.index("same_language")
    language_reliability_idx = feature_names.index("language_reliability_count")

    sports_features = _indexed_pair_matrix(rust_raw, "s1", "s1")[0]
    german_features = _indexed_pair_matrix(rust_raw, "s2", "s2")[0]

    assert sports_features[english_count_idx] == 2.0
    assert sports_features[same_language_idx] == 1.0
    assert sports_features[language_reliability_idx] == 2.0
    assert german_features[english_count_idx] == 0.0
    assert german_features[same_language_idx] == 1.0
    assert german_features[language_reliability_idx] == 2.0


def test_from_dataset_specter_cosine_matches_python_float64_path():
    dataset = _build_minimal_dataset("rust_contract_specter_float64_parity")
    dataset.specter_embeddings = {
        "1": np.array([0.1234567, 0.7654321, 0.3333333, 0.9999999], dtype=np.float32),
        "2": np.array([0.2222222, 0.4444444, 0.8888888, 0.1111111], dtype=np.float32),
    }
    for paper_id in ("1", "2"):
        dataset.papers[paper_id] = dataset.papers[paper_id]._replace(predicted_language="en", is_reliable=True)

    python_features, _ = _single_pair_featurize(("s1", "s2"), dataset=dataset)
    rust_featurizer = _S2AND_RUST.RustFeaturizer.from_dataset(dataset, 0.0, 10000.0, 1)
    rust_features = _indexed_pair_matrix(rust_featurizer, "s1", "s2")[0]
    feature_names = featurizer_mod.FeaturizationInfo().get_feature_names()
    specter_idx = feature_names.index("specter_cosine_sim")

    assert python_features[specter_idx] == pytest.approx(1.5781916063205044, abs=1e-15)
    assert rust_features[specter_idx] == pytest.approx(python_features[specter_idx], abs=1e-15)


def test_matrix_entrypoints_preserve_duplicate_selected_indices_and_empty_early_return():
    rust_featurizer, left_index, right_index = _rust_minimal_featurizer("rust_matrix_duplicate_indices")
    full_matrix = np.asarray(
        rust_featurizer.featurize_pairs_matrix_indexed([(left_index, right_index)], None, 1, np.nan)
    )
    full_cols = full_matrix.shape[1]
    selected_indices = [2, 2, 3]

    selected_matrix = np.asarray(
        rust_featurizer.featurize_pairs_matrix_indexed([(left_index, right_index)], selected_indices, 1, np.nan)
    )

    assert selected_matrix.shape == (1, 3)
    np.testing.assert_allclose(selected_matrix, full_matrix[:, selected_indices], equal_nan=True)

    with pytest.raises(ValueError, match=f"selected_indices contains out-of-range index {full_cols}"):
        rust_featurizer.featurize_pairs_matrix_indexed([(left_index, right_index)], [full_cols], 1, np.nan)

    assert np.asarray(rust_featurizer.featurize_pairs_matrix_indexed([], [full_cols], 1, np.nan)).shape == (0, 0)
    assert np.asarray(
        rust_featurizer.featurize_block_upper_triangle_matrix_indexed(
            [left_index, right_index], 0, 0, [full_cols], 1, np.nan
        )
    ).shape == (0, 0)


def test_aggregate_entrypoints_validate_indices_and_preserve_tuple_shapes():
    rust_featurizer, left_index, right_index = _rust_minimal_featurizer("rust_aggregate_index_contracts")
    full_cols = np.asarray(
        rust_featurizer.featurize_pairs_matrix_indexed([(left_index, right_index)], None, 1, np.nan)
    ).shape[1]
    selected_indices = [2, 2, 3]
    aggregate_indices = [2, 3, 2]
    left_indices = np.asarray([left_index], dtype=np.uint32)
    right_indices = np.asarray([right_index], dtype=np.uint32)
    owner_rows = np.asarray([0], dtype=np.uint32)

    matrix, counts, valid_counts, sums, mins, maxs = rust_featurizer.linker_pair_index_arrays_and_aggregate_stats(
        left_indices,
        right_indices,
        owner_rows,
        1,
        selected_indices,
        aggregate_indices,
        1,
        0.0,
        None,
    )
    matrix = np.asarray(matrix)
    assert matrix.shape == (1, 3)
    np.testing.assert_array_equal(np.asarray(counts), np.asarray([1], dtype=np.uint32))
    np.testing.assert_array_equal(np.asarray(valid_counts), np.ones((1, 3), dtype=np.uint64))
    np.testing.assert_allclose(np.asarray(sums), matrix[:, [0, 2, 0]], equal_nan=True)
    np.testing.assert_allclose(np.asarray(mins), matrix[:, [0, 2, 0]], equal_nan=True)
    np.testing.assert_allclose(np.asarray(maxs), matrix[:, [0, 2, 0]], equal_nan=True)

    empty = np.asarray([], dtype=np.uint32)
    empty_matrix, empty_counts, empty_valid_counts, empty_sums, empty_mins, empty_maxs = (
        rust_featurizer.linker_pair_index_arrays_and_aggregate_stats(
            empty,
            empty,
            empty,
            0,
            selected_indices,
            aggregate_indices,
            1,
            0.0,
            None,
        )
    )
    assert np.asarray(empty_matrix).shape == (0, 3)
    assert np.asarray(empty_counts).shape == (0,)
    assert np.asarray(empty_valid_counts).shape == (0, 3)
    assert np.asarray(empty_sums).shape == (0, 3)
    assert np.asarray(empty_mins).shape == (0, 3)
    assert np.asarray(empty_maxs).shape == (0, 3)

    with pytest.raises(ValueError, match=f"aggregate_indices contains out-of-range index {full_cols}"):
        rust_featurizer.linker_pair_index_arrays_and_aggregate_stats(
            empty,
            empty,
            empty,
            0,
            None,
            [full_cols],
            1,
            np.nan,
            None,
            False,
        )
    with pytest.raises(ValueError, match="aggregate index 3 is not present in matrix_indices"):
        rust_featurizer.linker_pair_index_arrays_and_aggregate_stats(
            left_indices,
            right_indices,
            owner_rows,
            1,
            [2],
            [3],
            1,
            np.nan,
            None,
        )
