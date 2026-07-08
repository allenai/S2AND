from __future__ import annotations

from dataclasses import fields

import pytest

from s2and.rust_lifecycle import (
    PYTHON_ONLY_POLICY,
    RustLifecyclePolicy,
    build_rust_lifecycle_policy,
)


def test_python_backend_always_returns_python_only_policy():
    for mode in ("train", "inference"):
        for arrow_featurization in (False, True):
            policy = build_rust_lifecycle_policy(
                backend="python",
                mode=mode,
                preprocess=True,
                arrow_featurization=arrow_featurization,
            )
            assert policy == PYTHON_ONLY_POLICY


def test_rust_inference_does_not_skip_python_paper_preprocess():
    policy = build_rust_lifecycle_policy(
        backend="rust",
        mode="inference",
        preprocess=True,
    )
    assert policy.mode == "rust_inference"
    assert policy.skip_python_paper_preprocess is False
    assert policy.defer_signature_ngrams_to_rust is True
    assert policy.defer_signature_fields_to_rust is False


def test_rust_inference_without_preprocess_is_python_only():
    policy = build_rust_lifecycle_policy(
        backend="rust",
        mode="inference",
        preprocess=False,
    )
    assert policy == PYTHON_ONLY_POLICY


def test_rust_training_without_arrow_featurization_is_python_only():
    """JSON-ingested training datasets featurize in Python (no Rust ingestion door)."""

    policy = build_rust_lifecycle_policy(
        backend="rust",
        mode="train",
        preprocess=True,
    )
    assert policy == PYTHON_ONLY_POLICY


@pytest.mark.parametrize("mode", ["train", "inference"])
def test_arrow_featurization_defers_dataset_build_work_to_rust(mode: str):
    policy = build_rust_lifecycle_policy(
        backend="rust",
        mode=mode,
        preprocess=True,
        arrow_featurization=True,
    )
    assert policy.mode == "rust_arrow_training"
    assert policy.skip_python_paper_preprocess is True
    assert policy.defer_signature_ngrams_to_rust is True
    assert policy.defer_signature_fields_to_rust is True


def test_lifecycle_policy_stores_only_canonical_mode():
    assert [field.name for field in fields(RustLifecyclePolicy)] == ["mode"]
