from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import s2and.feature_port as feature_port
import s2and.featurizer as featurizer_module
import s2and.memory_budget as memory_budget
from s2and.consts import LARGE_INTEGER
from s2and.data import ANDData
from s2and.featurizer import (
    DEFAULT_FEATURE_GROUPS,
    NUM_FEATURES,
    FeaturizationInfo,
    _ensure_python_pair_signature_ngrams,
    _signature_id_to_index_or_raise,
    many_pairs_featurize,
)
from s2and.runtime import RuntimeContext
from tests.helpers import tiny_name_counts_index

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
        name_counts_index=tiny_name_counts_index() if load_name_counts else None,
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


def test_default_feature_group_policy_is_canonical() -> None:
    assert tuple(FeaturizationInfo().features_to_use) == DEFAULT_FEATURE_GROUPS


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

    features, labels, _ = many_pairs_featurize(test_pairs, dataset, featurizer, n_jobs=2, chunk_size=1, nan_value=-1)

    expected_width = sum(len(featurizer.feature_group_to_index[name]) for name in _FULL_FEATURES)
    assert features.shape == (len(test_pairs), expected_width)
    np.testing.assert_array_equal(labels, np.asarray([0, 0, 0, -1]))
    assert np.any(features != -LARGE_INTEGER)


def test_single_pair_featurize_surfaces_missing_preprocessed_fields() -> None:
    dataset = _dummy_dataset("dummy_missing_preprocessed_field")
    dataset.signatures["0"] = dataset.signatures["0"]._replace(author_info_first_normalized_without_apostrophe=None)

    with pytest.raises(
        RuntimeError,
        match=r"requires preprocessed field signature_1\.author_info_first_normalized_without_apostrophe",
    ):
        featurizer_module._single_pair_featurize(("0", "1"), dataset=dataset)


def test_malformed_emails_produce_only_missing_features() -> None:
    dataset = _dummy_dataset("dummy_malformed_emails", load_name_counts=False)
    dataset.signatures["0"] = dataset.signatures["0"]._replace(author_info_email="a@b@c")
    dataset.signatures["1"] = dataset.signatures["1"]._replace(author_info_email="ab@c")
    featurizer = FeaturizationInfo(features_to_use=["email_similarity"])

    features, _labels, _ = many_pairs_featurize(
        [("0", "1", 0)],
        dataset,
        featurizer,
        n_jobs=1,
        chunk_size=1,
        nan_value=np.nan,
    )

    assert features.shape == (1, 2)
    assert np.isnan(features).all()


def test_concurrent_python_featurization_uses_each_request_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    first_dataset = _dummy_dataset("concurrent_first", load_name_counts=False)
    second_dataset = _dummy_dataset("concurrent_second", load_name_counts=False)
    for dataset, year_difference in ((first_dataset, 5), (second_dataset, 20)):
        first_paper_id = str(dataset.signatures["0"].paper_id)
        second_paper_id = str(dataset.signatures["1"].paper_id)
        dataset.papers[first_paper_id] = dataset.papers[first_paper_id]._replace(year=2000)
        dataset.papers[second_paper_id] = dataset.papers[second_paper_id]._replace(year=2000 + year_difference)

    original_execute = featurizer_module._execute_python_featurization_phase  # noqa: SLF001
    execute_barrier = threading.Barrier(2)

    def synchronized_execute(**kwargs: Any) -> str:
        execute_barrier.wait(timeout=5)
        return original_execute(**kwargs)

    monkeypatch.setattr(featurizer_module, "_execute_python_featurization_phase", synchronized_execute)
    featurizer = FeaturizationInfo(features_to_use=["year_diff"])

    def featurize(dataset: ANDData) -> float:
        features, _labels, _nameless = many_pairs_featurize(
            [("0", "1", 0)],
            dataset,
            featurizer,
            n_jobs=1,
            chunk_size=1,
            nan_value=np.nan,
        )
        return float(features[0, 0])

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(featurize, first_dataset)
        second_future = executor.submit(featurize, second_dataset)

    assert first_future.result() == 5.0
    assert second_future.result() == 20.0


@pytest.mark.parametrize("zero_embedding", [[0.0, 0.0], np.asarray([0.0, 0.0])])
def test_zero_specter_embedding_is_missing_for_lists_and_arrays(zero_embedding) -> None:
    dataset = _dummy_dataset("dummy_zero_specter")
    paper_id_0 = str(dataset.signatures["0"].paper_id)
    paper_id_1 = str(dataset.signatures["1"].paper_id)
    dataset.specter_embeddings = {
        paper_id_0: zero_embedding,
        paper_id_1: [1.0, 0.0],
    }
    featurizer = FeaturizationInfo(features_to_use=["embedding_similarity"])

    features, _labels, _ = many_pairs_featurize(
        [("0", "1", 0)],
        dataset,
        featurizer,
        n_jobs=1,
        chunk_size=1,
        nan_value=np.nan,
    )

    assert features.shape == (1, 1)
    assert np.isnan(features[0, 0])


def test_specter_embedding_must_be_one_dimensional() -> None:
    dataset = _dummy_dataset("dummy_invalid_specter")
    paper_id_0 = str(dataset.signatures["0"].paper_id)
    paper_id_1 = str(dataset.signatures["1"].paper_id)
    dataset.specter_embeddings = {
        paper_id_0: [[1.0, 0.0]],
        paper_id_1: [1.0, 0.0],
    }
    featurizer = FeaturizationInfo(features_to_use=["embedding_similarity"])

    with pytest.raises(ValueError, match="one-dimensional"):
        many_pairs_featurize(
            [("0", "1", 0)],
            dataset,
            featurizer,
            n_jobs=1,
            chunk_size=1,
            nan_value=np.nan,
        )


def test_numeric_specter_tuple_keys_are_available_to_pair_featurization() -> None:
    dataset = ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        specter_embeddings=(
            np.asarray([[1.0, 0.0], [1.0, 0.0]]),
            [53235312, 27077319],
        ),
        name="dummy_numeric_specter_keys",
        name_counts_index=None,
    )
    featurizer = FeaturizationInfo(features_to_use=["embedding_similarity"])

    features, _labels, _ = many_pairs_featurize(
        [("0", "1", 0)],
        dataset,
        featurizer,
        n_jobs=1,
        chunk_size=1,
        nan_value=np.nan,
    )

    np.testing.assert_array_equal(features, np.asarray([[2.0]]))


def test_empty_python_pair_featurization_does_not_mark_missing_ngrams_ready() -> None:
    runtime_context = RuntimeContext(
        operation="featurization_run",
        backend="python",
        run_id="run-python-ngrams",
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
        backend="python",
        run_id="run-python-deferred-fields",
    )
    dataset = cast(
        ANDData,
        SimpleNamespace(
            signatures={},
            arrow_paths={"signatures": "signatures.arrow"},
        ),
    )

    with pytest.raises(RuntimeError, match="normalized signature fields deferred to Rust"):
        _ensure_python_pair_signature_ngrams(dataset, [("a", "b", 1)], runtime_context)


def test_delete_training_data_uses_global_coauthor_similarity_index(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = cast(ANDData, SimpleNamespace(name="delete_training_data", mode="train", signatures={}))
    featurizer_info = FeaturizationInfo(features_to_use=["coauthor_similarity"])
    runtime_context = RuntimeContext(
        operation="featurization_run",
        backend="python",
        run_id="run-delete-training-data",
    )

    def fake_single_pair_featurize(
        _pair: tuple[str, str],
        index: int,
        *,
        dataset: ANDData | None = None,
    ) -> tuple[np.ndarray, int]:
        assert dataset is not None
        row = np.zeros(NUM_FEATURES, dtype=np.float64)
        row[featurizer_info.feature_group_to_index["coauthor_similarity"][1]] = 1.0
        return row, index

    monkeypatch.setattr("s2and.featurizer._single_pair_featurize", fake_single_pair_featurize)

    features, labels, _nameless = many_pairs_featurize(
        [("a", "b", 0), ("c", "d", 1)],
        dataset,
        featurizer_info,
        n_jobs=1,
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
            arrow_paths={"signatures": "signatures.arrow"},
        ),
    )
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    runtime_context = RuntimeContext(
        operation="featurization_run",
        backend="rust",
        run_id="run-1",
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
        chunk_size=1,
        runtime_context=runtime_context,
    )

    assert state["prewarm_called"] is True
    assert state["rss_called"] is True


def test_ineligible_rust_calibration_does_not_consume_process_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(featurizer_module, "_RUST_BATCH_CALIBRATED_FIXED_OVERHEAD_BYTES", None)
    monkeypatch.setattr(featurizer_module, "_RUST_BATCH_CALIBRATION_ATTEMPTED", False)
    monkeypatch.setattr(
        featurizer_module,
        "_rust_batch_probe_row_counts",
        lambda total_pairs, **_kwargs: [] if total_pairs < 3 else [1, 2, 3],
    )
    monkeypatch.setattr(memory_budget, "current_rss_bytes_best_effort", lambda _total_ram: (0, "test"))

    class FakeRustFeaturizer:
        def __init__(self) -> None:
            self.calls = 0

        def featurize_pairs_matrix_indexed(
            self,
            pairs: list[tuple[int, int]],
            selected_indices: list[int] | None,
            num_threads: int,
            nan_value: float,
        ) -> np.ndarray:
            del num_threads, nan_value
            self.calls += 1
            width = NUM_FEATURES if selected_indices is None else len(selected_indices)
            return np.zeros((len(pairs), width), dtype=np.float64)

    rust_featurizer = FakeRustFeaturizer()
    kwargs = {
        "rust_featurizer": rust_featurizer,
        "signature_id_to_index": {"a": 0, "b": 1},
        "rust_selected_indices": [0],
        "selected_feature_count": 1,
        "nameless_feature_count": 0,
        "row_overhead_bytes": 0,
        "persistent_row_overhead_bytes": 0,
        "configured_fixed_overhead_bytes": 0,
        "num_threads": 1,
        "total_ram_for_stage": 1024,
        "run_id": "test-calibration",
    }

    result = featurizer_module._maybe_calibrate_rust_batch_fixed_overhead_bytes(
        pieces_of_work=[(("a", "b"), 0)],
        **kwargs,
    )

    assert result is None
    assert featurizer_module._RUST_BATCH_CALIBRATION_ATTEMPTED is False

    result = featurizer_module._maybe_calibrate_rust_batch_fixed_overhead_bytes(
        pieces_of_work=[(("a", "b"), index) for index in range(3)],
        **kwargs,
    )

    assert result is None
    assert rust_featurizer.calls == 3
    assert featurizer_module._RUST_BATCH_CALIBRATION_ATTEMPTED is True


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
            arrow_paths={"signatures": "signatures.arrow"},
        ),
    )
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    runtime_context = RuntimeContext(
        operation="featurization_run",
        backend="rust",
        run_id="run-lazy-load",
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
        chunk_size=1,
        runtime_context=runtime_context,
    )

    assert state["prewarm_called"] is True


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
        chunk_size=1,
        nan_value=-1,
    )
    features_multi, labels_multi, _ = many_pairs_featurize(
        test_pairs,
        dataset,
        featurizer,
        n_jobs=2,
        chunk_size=3,
        nan_value=-1,
    )

    _assert_feature_arrays_equal(features_single, features_multi)
    np.testing.assert_array_equal(labels_single, labels_multi)


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
            arrow_paths={"signatures": "signatures.arrow"},
        ),
    )
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    runtime_context = RuntimeContext(
        operation="featurization_run",
        backend="rust",
        run_id="run-raises",
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
            chunk_size=1,
            runtime_context=runtime_context,
        )
