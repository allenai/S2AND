from __future__ import annotations

import pytest

from s2and.rust_lifecycle import (
    FORCE_PYTHON_PAPER_PREPROCESS_ENV,
    PYTHON_ONLY_POLICY,
    RustLifecyclePolicy,
    build_rust_lifecycle_policy,
)


def test_python_backend_always_returns_python_only_policy():
    for mode in ("train", "inference"):
        policy = build_rust_lifecycle_policy(
            backend="python",
            mode=mode,
            has_signatures_path=True,
            has_papers_path=True,
            preprocess=True,
            use_rust=False,
        )
        assert policy == PYTHON_ONLY_POLICY


@pytest.mark.parametrize(
    ("has_signatures_path", "has_papers_path", "expected_build_path"),
    [
        (False, False, "from_dataset"),
        (False, True, "from_dataset"),
        (True, False, "from_dataset"),
        (True, True, "from_json_paths"),
    ],
)
def test_rust_inference_build_path_requires_both_json_paths(
    has_signatures_path: bool,
    has_papers_path: bool,
    expected_build_path: str,
):
    policy = build_rust_lifecycle_policy(
        backend="rust",
        mode="inference",
        has_signatures_path=has_signatures_path,
        has_papers_path=has_papers_path,
        preprocess=True,
        use_rust=True,
    )
    assert policy.rust_build_path == expected_build_path


def test_rust_inference_without_paths_does_not_skip_python_paper_preprocess():
    policy = build_rust_lifecycle_policy(
        backend="rust",
        mode="inference",
        has_signatures_path=False,
        has_papers_path=False,
        preprocess=True,
        use_rust=True,
    )
    assert policy.rust_build_path == "from_dataset"
    assert policy.skip_python_paper_preprocess is False


def test_rust_training_from_dataset_skips_python_paper_preprocess_when_capability_present():
    policy = build_rust_lifecycle_policy(
        backend="rust",
        mode="train",
        has_signatures_path=False,
        has_papers_path=False,
        preprocess=True,
        compute_reference_features=False,
        use_rust=True,
        from_dataset_paper_preprocess_available=True,
    )
    assert policy.rust_build_path == "from_dataset"
    assert policy.skip_python_paper_preprocess is True


def test_rust_training_from_dataset_does_not_skip_with_reference_features():
    policy = build_rust_lifecycle_policy(
        backend="rust",
        mode="train",
        has_signatures_path=False,
        has_papers_path=False,
        preprocess=True,
        compute_reference_features=True,
        use_rust=True,
        from_dataset_paper_preprocess_available=True,
    )
    assert policy.rust_build_path == "from_dataset"
    assert policy.skip_python_paper_preprocess is False


def test_rust_training_from_dataset_does_not_skip_without_capability():
    policy = build_rust_lifecycle_policy(
        backend="rust",
        mode="train",
        has_signatures_path=False,
        has_papers_path=False,
        preprocess=True,
        compute_reference_features=False,
        use_rust=True,
        from_dataset_paper_preprocess_available=False,
    )
    assert policy.rust_build_path == "from_dataset"
    assert policy.skip_python_paper_preprocess is False


def test_force_python_paper_preprocess_env_disables_skip(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(FORCE_PYTHON_PAPER_PREPROCESS_ENV, "1")
    policy = build_rust_lifecycle_policy(
        backend="rust",
        mode="train",
        has_signatures_path=False,
        has_papers_path=False,
        preprocess=True,
        compute_reference_features=False,
        use_rust=True,
        from_dataset_paper_preprocess_available=True,
    )
    assert policy.rust_build_path == "from_dataset"
    assert policy.skip_python_paper_preprocess is False


def test_force_python_paper_preprocess_env_invalid_value_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(FORCE_PYTHON_PAPER_PREPROCESS_ENV, "maybe")
    with pytest.raises(ValueError, match="Invalid"):
        build_rust_lifecycle_policy(
            backend="rust",
            mode="train",
            has_signatures_path=False,
            has_papers_path=False,
            preprocess=True,
            compute_reference_features=False,
            use_rust=True,
            from_dataset_paper_preprocess_available=True,
        )


def test_rust_inference_with_sinonym_overwrite_keeps_from_json_paths():
    policy = build_rust_lifecycle_policy(
        backend="rust",
        mode="inference",
        has_signatures_path=True,
        has_papers_path=True,
        preprocess=True,
        use_rust=True,
        use_sinonym_overwrite=True,
    )
    assert policy.rust_build_path == "from_json_paths"
    assert policy.skip_python_paper_preprocess is True


@pytest.mark.parametrize("preprocess", [False, True])
@pytest.mark.parametrize("use_rust", [False, True])
def test_defer_signature_ngrams_requires_preprocess_and_rust(preprocess: bool, use_rust: bool):
    policy = build_rust_lifecycle_policy(
        backend="rust",
        mode="train",
        has_signatures_path=True,
        has_papers_path=True,
        preprocess=preprocess,
        use_rust=use_rust,
    )
    assert policy.defer_signature_ngrams_to_rust is (preprocess and use_rust)


@pytest.mark.parametrize("mode", ["train", "inference"])
@pytest.mark.parametrize("use_rust", [False, True])
def test_defer_signature_fields_requires_rust_and_non_inference(
    mode: str,
    use_rust: bool,
):
    policy = build_rust_lifecycle_policy(
        backend="rust",
        mode=mode,
        has_signatures_path=True,
        has_papers_path=True,
        preprocess=True,
        use_rust=use_rust,
    )
    expected = bool(mode == "train" and use_rust)
    assert policy.defer_signature_fields_to_rust is expected


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("mode", ["train", "inference"])
@pytest.mark.parametrize("has_signatures_path", [False, True])
@pytest.mark.parametrize("has_papers_path", [False, True])
@pytest.mark.parametrize("preprocess", [False, True])
@pytest.mark.parametrize("use_rust", [False, True])
def test_policy_covers_all_combinations(
    backend: str,
    mode: str,
    has_signatures_path: bool,
    has_papers_path: bool,
    preprocess: bool,
    use_rust: bool,
):
    """Smoke test: build_rust_lifecycle_policy doesn't raise for any valid combination."""
    policy = build_rust_lifecycle_policy(
        backend=backend,  # type: ignore[arg-type]
        mode=mode,
        has_signatures_path=has_signatures_path,
        has_papers_path=has_papers_path,
        preprocess=preprocess,
        use_rust=use_rust,
    )
    assert isinstance(policy, RustLifecyclePolicy)
    if backend == "python":
        assert policy == PYTHON_ONLY_POLICY
