from __future__ import annotations

from dataclasses import fields

from s2and.rust_lifecycle import (
    PYTHON_ONLY_POLICY,
    RUST_ARROW_TRAINING_POLICY,
    RustLifecyclePolicy,
)


def test_python_policy_keeps_dataset_preprocessing_in_python() -> None:
    assert PYTHON_ONLY_POLICY.mode == "python_only"
    assert PYTHON_ONLY_POLICY.skip_python_paper_preprocess is False
    assert PYTHON_ONLY_POLICY.defer_signature_ngrams_to_rust is False
    assert PYTHON_ONLY_POLICY.defer_signature_fields_to_rust is False


def test_arrow_training_policy_defers_dataset_build_work_to_rust() -> None:
    assert RUST_ARROW_TRAINING_POLICY.mode == "rust_arrow_training"
    assert RUST_ARROW_TRAINING_POLICY.skip_python_paper_preprocess is True
    assert RUST_ARROW_TRAINING_POLICY.defer_signature_ngrams_to_rust is True
    assert RUST_ARROW_TRAINING_POLICY.defer_signature_fields_to_rust is True


def test_lifecycle_policy_stores_only_route() -> None:
    assert [field.name for field in fields(RustLifecyclePolicy)] == ["mode"]
