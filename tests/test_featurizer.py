from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import s2and.feature_port as feature_port
import s2and.memory_budget as memory_budget
from s2and.consts import LARGE_INTEGER
from s2and.data import ANDData
from s2and.featurizer import (
    NUM_FEATURES,
    FeaturizationInfo,
    _ensure_python_pair_signature_ngrams,
    _signature_id_to_index_or_raise,
    many_pairs_featurize,
)
from s2and.runtime import RuntimeContext
from tests.helpers import tiny_name_counts

_FULL_FEATURES = [
    "name_similarity",
    "affiliation_similarity",
    "email_similarity",
    "coauthor_similarity",
    "venue_similarity",
    "year_diff",
    "title_similarity",
    "misc_features",
    "name_counts",
    "journal_similarity",
    "advanced_name_similarity",
]


def _dummy_dataset(
    name: str,
    *,
    load_name_counts: bool = True,
) -> ANDData:
    return ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        name=name,
        load_name_counts=tiny_name_counts() if load_name_counts else False,
    )


def _assert_feature_arrays_equal(left: np.ndarray, right: np.ndarray) -> None:
    assert left.shape == right.shape
    np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-10, equal_nan=True)


def test_default_features_are_instance_isolated() -> None:
    first = FeaturizationInfo()
    first.features_to_use.remove("name_similarity")

    second = FeaturizationInfo()
    assert "name_similarity" in second.features_to_use
    assert first.features_to_use is not second.features_to_use


def test_featurization_info_rejects_unknown_feature_groups() -> None:
    with pytest.raises(ValueError, match="Unknown feature group"):
        FeaturizationInfo(features_to_use=["year_diff", "reference_features"])


def test_featurizer_computes_requested_pairs() -> None:
    dataset = _dummy_dataset("dummy_featurizer")
    featurizer = FeaturizationInfo(features_to_use=_FULL_FEATURES)
    test_pairs = [
        ("3", "0", 0),
        ("3", "1", 0),
        ("3", "2", 0),
        ("3", "2", -1),
    ]

    features, labels, _ = many_pairs_featurize(test_pairs, dataset, featurizer, 2, False, 1, nan_value=-1)

    expected_width = sum(len(featurizer.feature_group_to_index[name]) for name in _FULL_FEATURES)
    assert features.shape == (len(test_pairs), expected_width)
    np.testing.assert_array_equal(labels, np.asarray([0, 0, 0, -1]))
    assert np.any(features != -LARGE_INTEGER)


def test_empty_python_pair_featurization_does_not_mark_missing_ngrams_ready() -> None:
    runtime_context = RuntimeContext(
        operation="featurization_run",
        requested_backend="python",
        resolved_backend="python",
        use_rust=False,
        run_id="run-python-ngrams",
        source="argument",
    )
    signature = SimpleNamespace(author_info_affiliations_n_grams=None, author_info_coauthor_n_grams=None)
    state = {"materialized": 0}

    def materialize_signature_ngrams_python() -> None:
        state["materialized"] += 1
        signature.author_info_affiliations_n_grams = {}
        signature.author_info_coauthor_n_grams = {}

    dataset = cast(
        ANDData,
        SimpleNamespace(
            signatures={"a": signature},
            materialize_signature_ngrams_python=materialize_signature_ngrams_python,
        ),
    )

    _ensure_python_pair_signature_ngrams(dataset, [], runtime_context)
    assert not getattr(dataset, "_s2and_python_pair_ngrams_ready", False)

    _ensure_python_pair_signature_ngrams(dataset, [("a", "a", 1)], runtime_context)
    assert state["materialized"] == 1
    assert dataset._s2and_python_pair_ngrams_ready is True


def test_python_pair_featurization_rejects_rust_deferred_signature_fields() -> None:
    runtime_context = RuntimeContext(
        operation="featurization_run",
        requested_backend="python",
        resolved_backend="python",
        use_rust=False,
        run_id="run-python-deferred-fields",
        source="argument",
    )
    dataset = cast(
        ANDData,
        SimpleNamespace(
            signatures={},
            rust_lifecycle_policy=SimpleNamespace(defer_signature_fields_to_rust=True),
        ),
    )

    with pytest.raises(RuntimeError, match="normalized signature fields deferred to Rust"):
        _ensure_python_pair_signature_ngrams(dataset, [("a", "b", 1)], runtime_context)


def test_delete_training_data_uses_global_coauthor_similarity_index(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = cast(ANDData, SimpleNamespace(name="delete_training_data", mode="train", signatures={}))
    featurizer_info = FeaturizationInfo(features_to_use=["coauthor_similarity"])
    runtime_context = RuntimeContext(
        operation="featurization_run",
        requested_backend="python",
        resolved_backend="python",
        use_rust=False,
        run_id="run-delete-training-data",
        source="argument",
    )

    def fake_single_pair_featurize(_pair: tuple[str, str], index: int) -> tuple[np.ndarray, int]:
        row = np.zeros(NUM_FEATURES, dtype=np.float64)
        row[featurizer_info.feature_group_to_index["coauthor_similarity"][1]] = 1.0
        return row, index

    monkeypatch.setattr("s2and.featurizer._single_pair_featurize", fake_single_pair_featurize)

    features, labels, _nameless = many_pairs_featurize(
        [("a", "b", 0), ("c", "d", 1)],
        dataset,
        featurizer_info,
        n_jobs=1,
        use_cache=False,
        chunk_size=1,
        delete_training_data=True,
        runtime_context=runtime_context,
    )

    assert features.shape == (1, 3)
    np.testing.assert_array_equal(labels, np.asarray([1.0]))


def test_rust_prewarm_happens_before_rss_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    # Placeholder Arrow paths mark the dataset as Rust-eligible; the actual Rust
    # featurizer build is mocked out below via feature_port._get_rust_featurizer.
    dataset = cast(
        ANDData,
        SimpleNamespace(
            name="dummy",
            mode="train",
            rust_featurizer_arrow_paths={"signatures": "signatures.arrow"},
        ),
    )
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    runtime_context = RuntimeContext(
        operation="featurization_run",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="run-1",
        source="default",
    )

    state = {"prewarm_called": False, "rss_called": False}

    class FakeRustFeaturizer:
        def signature_ids(self) -> list[str]:
            return ["a", "b"]

        def featurize_pairs_matrix_indexed(
            self,
            pairs: object,
            selected_indices: object,
            num_threads: object,
            nan_value: object,
        ) -> np.ndarray:
            del selected_indices, num_threads, nan_value
            return np.zeros((len(cast(Any, pairs)), NUM_FEATURES), dtype=np.float64)

    def fake_get_rust_featurizer(*_args: object, **_kwargs: object) -> object:
        state["prewarm_called"] = True
        return FakeRustFeaturizer()

    def fake_resolve_total_ram_bytes(_total_ram_bytes: object) -> tuple[int, str]:
        return 1024, "test"

    def fake_current_rss(_total_ram_bytes: object) -> tuple[int, str]:
        state["rss_called"] = True
        assert state["prewarm_called"] is True
        return 128, "test"

    monkeypatch.setattr(feature_port, "s2and_rust", object())
    monkeypatch.setattr(feature_port, "_get_rust_featurizer", fake_get_rust_featurizer)
    monkeypatch.setattr(memory_budget, "resolve_total_ram_bytes", fake_resolve_total_ram_bytes)
    monkeypatch.setattr(memory_budget, "current_rss_bytes_best_effort", fake_current_rss)

    many_pairs_featurize(
        [("a", "b", -1)],
        dataset,
        featurizer_info,
        n_jobs=1,
        use_cache=False,
        chunk_size=1,
        runtime_context=runtime_context,
    )

    assert state["prewarm_called"] is True
    assert state["rss_called"] is True


def test_many_pairs_featurize_uses_lazy_rust_loader_before_unavailable_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Placeholder Arrow paths mark the dataset as Rust-eligible; the actual Rust
    # featurizer build is mocked out below via feature_port._get_rust_featurizer.
    dataset = cast(
        ANDData,
        SimpleNamespace(
            name="dummy",
            mode="train",
            rust_featurizer_arrow_paths={"signatures": "signatures.arrow"},
        ),
    )
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    runtime_context = RuntimeContext(
        operation="featurization_run",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="run-lazy-load",
        source="default",
    )
    state = {"prewarm_called": False}

    class FakeRustFeaturizer:
        def signature_ids(self) -> list[str]:
            return ["a", "b"]

        def featurize_pairs_matrix_indexed(
            self,
            pairs: object,
            selected_indices: object,
            num_threads: object,
            nan_value: object,
        ) -> np.ndarray:
            del selected_indices, num_threads, nan_value
            return np.zeros((len(cast(Any, pairs)), NUM_FEATURES), dtype=np.float64)

    def fake_get_rust_featurizer(*_args: object, **_kwargs: object) -> object:
        state["prewarm_called"] = True
        return FakeRustFeaturizer()

    monkeypatch.setattr(feature_port, "s2and_rust", None)
    monkeypatch.setattr(feature_port, "_get_rust_featurizer", fake_get_rust_featurizer)

    many_pairs_featurize(
        [("a", "b", -1)],
        dataset,
        featurizer_info,
        n_jobs=1,
        use_cache=False,
        chunk_size=1,
        runtime_context=runtime_context,
    )

    assert state["prewarm_called"] is True


def test_featurizer_with_feature_subset_ok() -> None:
    dataset_no_ref = _dummy_dataset("dummy_no_ref_ok")
    features_to_use = [
        "name_similarity",
        "affiliation_similarity",
        "email_similarity",
        "coauthor_similarity",
        "venue_similarity",
        "year_diff",
        "title_similarity",
        "misc_features",
        "name_counts",
        "journal_similarity",
        "advanced_name_similarity",
    ]
    featurizer = FeaturizationInfo(features_to_use=features_to_use)
    test_pairs = [("3", "0", 0), ("3", "1", 0)]

    features, labels, _ = many_pairs_featurize(
        test_pairs,
        dataset_no_ref,
        featurizer,
        n_jobs=1,
        use_cache=False,
        chunk_size=1,
        nan_value=-1,
    )

    assert features.shape[0] == len(test_pairs)
    assert features.shape[1] == sum(len(featurizer.feature_group_to_index[name]) for name in features_to_use)
    assert labels.tolist() == [0, 0]
    assert np.any(features != -LARGE_INTEGER)


def test_get_constraint() -> None:
    dataset = _dummy_dataset("dummy_constraints")

    assert dataset.get_constraint("0", "8", high_value=100) == 100
    assert dataset.get_constraint("6", "8", high_value=100) == 100
    assert dataset.get_constraint("0", "1") is None


def test_multiprocessing_featurization_consistency() -> None:
    dataset = _dummy_dataset("dummy_mp_consistency")
    featurizer = FeaturizationInfo(features_to_use=_FULL_FEATURES)
    test_pairs = [
        ("3", "0", 0),
        ("3", "1", 0),
        ("3", "2", 0),
        ("0", "1", 1),
    ]

    features_single, labels_single, _ = many_pairs_featurize(
        test_pairs,
        dataset,
        featurizer,
        n_jobs=1,
        use_cache=False,
        chunk_size=1,
        nan_value=-1,
    )
    features_multi, labels_multi, _ = many_pairs_featurize(
        test_pairs,
        dataset,
        featurizer,
        n_jobs=2,
        use_cache=False,
        chunk_size=1,
        nan_value=-1,
    )

    _assert_feature_arrays_equal(features_single, features_multi)
    np.testing.assert_array_equal(labels_single, labels_multi)


def test_bound_dataset_is_available_in_workers() -> None:
    dataset = _dummy_dataset("dummy_bound_workers")
    featurizer = FeaturizationInfo(features_to_use=_FULL_FEATURES)
    test_pairs = [
        ("3", "0", 0),
        ("3", "1", 0),
    ]

    try:
        features, _labels, _ = many_pairs_featurize(
            test_pairs,
            dataset,
            featurizer,
            n_jobs=2,
            use_cache=False,
            chunk_size=1,
            nan_value=-1,
        )
    except (AttributeError, NameError) as exc:
        raise AssertionError(f"Dataset not available in worker processes: {exc}") from exc

    assert features.shape[0] == len(test_pairs)


def test_multiprocessing_with_different_chunk_sizes() -> None:
    dataset = _dummy_dataset("dummy_mp_chunk_sizes")
    featurizer = FeaturizationInfo(features_to_use=_FULL_FEATURES)
    test_pairs = [
        ("3", "0", 0),
        ("3", "1", 0),
        ("3", "2", 0),
        ("0", "1", 1),
        ("0", "2", 0),
        ("1", "2", 1),
    ]

    features_chunk1, labels_chunk1, _ = many_pairs_featurize(
        test_pairs,
        dataset,
        featurizer,
        n_jobs=2,
        use_cache=False,
        chunk_size=1,
        nan_value=-1,
    )
    features_chunk3, labels_chunk3, _ = many_pairs_featurize(
        test_pairs,
        dataset,
        featurizer,
        n_jobs=2,
        use_cache=False,
        chunk_size=3,
        nan_value=-1,
    )

    _assert_feature_arrays_equal(features_chunk1, features_chunk3)
    np.testing.assert_array_equal(labels_chunk1, labels_chunk3)


def test_multiprocessing_fallback_to_single_thread() -> None:
    dataset = _dummy_dataset("dummy_mp_small_work")
    featurizer = FeaturizationInfo(features_to_use=_FULL_FEATURES)

    features, labels, _ = many_pairs_featurize(
        [("3", "0", 0)],
        dataset,
        featurizer,
        n_jobs=4,
        use_cache=False,
        chunk_size=1,
        nan_value=-1,
    )

    assert features.shape[0] == 1
    assert labels.shape[0] == 1


def test_spawn_context_compatibility() -> None:
    dataset = _dummy_dataset("dummy_spawn_context")
    featurizer = FeaturizationInfo(features_to_use=_FULL_FEATURES)
    test_pairs = [
        ("3", "0", 0),
        ("3", "1", 0),
        ("0", "1", 1),
    ]

    features, _labels, _ = many_pairs_featurize(
        test_pairs,
        dataset,
        featurizer,
        n_jobs=2,
        use_cache=False,
        chunk_size=1,
        nan_value=-1,
    )

    assert features.shape[0] == len(test_pairs)
    assert not np.all(features == -LARGE_INTEGER)
    assert len(features[features != -LARGE_INTEGER]) > 0


def test_signature_id_to_index_or_raise_accepts_non_string_pair_ids() -> None:
    signature_id_to_index = {"1": 10, "2": 20}

    assert _signature_id_to_index_or_raise(signature_id_to_index, 1) == 10
    assert _signature_id_to_index_or_raise(signature_id_to_index, 2) == 20


def test_signature_id_to_index_or_raise_reports_missing_signature_id() -> None:
    signature_id_to_index = {"1": 10}

    with pytest.raises(ValueError, match="999"):
        _signature_id_to_index_or_raise(signature_id_to_index, 999)


def test_many_pairs_featurize_surfaces_rust_initialization_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # Placeholder Arrow paths mark the dataset as Rust-eligible; the Rust build
    # itself is replaced with fail_prewarm below, so no real build happens.
    dataset = cast(
        ANDData,
        SimpleNamespace(
            name="dummy",
            rust_featurizer_arrow_paths={"signatures": "signatures.arrow"},
        ),
    )
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    runtime_context = RuntimeContext(
        operation="featurization_run",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="run-raises",
        source="default",
    )

    monkeypatch.setattr(feature_port, "s2and_rust", object())

    def fail_prewarm(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("native init failed")

    monkeypatch.setattr(feature_port, "_get_rust_featurizer", fail_prewarm)

    with pytest.raises(RuntimeError, match="Rust featurizer init failed"):
        many_pairs_featurize(
            [],
            dataset,
            featurizer_info,
            n_jobs=1,
            use_cache=False,
            chunk_size=1,
            runtime_context=runtime_context,
        )
