from __future__ import annotations

from itertools import product

import pytest

from s2and.runtime import RuntimeStage
from s2and.rust_lifecycle import PYTHON_ONLY_POLICY, RustLifecyclePolicy, build_rust_lifecycle_policy


def _stages(
    *,
    ingest_preprocess: bool,
    pair_featurization: bool,
    constraints: bool,
) -> dict[RuntimeStage, bool]:
    return {
        "ingest_preprocess": ingest_preprocess,
        "pair_featurization": pair_featurization,
        "constraints": constraints,
    }


def _legacy_expected(
    *,
    backend: str,
    mode: str,
    has_signatures_path: bool,
    has_papers_path: bool,
    preprocess: bool,
    stage_enablement: dict[RuntimeStage, bool],
) -> RustLifecyclePolicy:
    if backend == "python":
        return PYTHON_ONLY_POLICY

    is_inference = mode.strip().lower() == "inference"
    ingest_enabled = bool(stage_enablement.get("ingest_preprocess", False))
    pair_enabled = bool(stage_enablement.get("pair_featurization", False))
    constraints_enabled = bool(stage_enablement.get("constraints", False))
    use_json_paths = is_inference and has_signatures_path and has_papers_path
    use_rust_json_ingest = is_inference and ingest_enabled

    return RustLifecyclePolicy(
        rust_build_path="from_json_paths" if use_json_paths else "from_dataset",
        skip_python_paper_preprocess=bool(preprocess and use_rust_json_ingest),
        defer_signature_ngrams_to_rust=bool(preprocess and ingest_enabled),
        defer_signature_fields_to_rust=bool(
            preprocess and ingest_enabled and pair_enabled and constraints_enabled and not use_rust_json_ingest
        ),
    )


def test_python_backend_always_returns_python_only_policy():
    stage_enablement = _stages(ingest_preprocess=True, pair_featurization=True, constraints=True)
    for mode in ("train", "inference"):
        policy = build_rust_lifecycle_policy(
            backend="python",
            mode=mode,
            has_signatures_path=True,
            has_papers_path=True,
            preprocess=True,
            stage_enablement=stage_enablement,
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
        stage_enablement=_stages(ingest_preprocess=True, pair_featurization=True, constraints=True),
    )
    assert policy.rust_build_path == expected_build_path


def test_rust_inference_without_paths_still_skips_python_paper_preprocess_when_ingest_is_enabled():
    policy = build_rust_lifecycle_policy(
        backend="rust",
        mode="inference",
        has_signatures_path=False,
        has_papers_path=False,
        preprocess=True,
        stage_enablement=_stages(ingest_preprocess=True, pair_featurization=True, constraints=True),
    )
    assert policy.rust_build_path == "from_dataset"
    assert policy.skip_python_paper_preprocess is True



@pytest.mark.parametrize("preprocess", [False, True])
@pytest.mark.parametrize("ingest_enabled", [False, True])
def test_defer_signature_ngrams_requires_preprocess_and_ingest(preprocess: bool, ingest_enabled: bool):
    policy = build_rust_lifecycle_policy(
        backend="rust",
        mode="train",
        has_signatures_path=True,
        has_papers_path=True,
        preprocess=preprocess,
        stage_enablement=_stages(
            ingest_preprocess=ingest_enabled,
            pair_featurization=True,
            constraints=True,
        ),
    )
    assert policy.defer_signature_ngrams_to_rust is (preprocess and ingest_enabled)


@pytest.mark.parametrize("mode", ["train", "inference"])
@pytest.mark.parametrize("ingest_enabled", [False, True])
@pytest.mark.parametrize("pair_enabled", [False, True])
@pytest.mark.parametrize("constraints_enabled", [False, True])
def test_defer_signature_fields_requires_all_rust_stages_and_no_inference_json_ingest(
    mode: str,
    ingest_enabled: bool,
    pair_enabled: bool,
    constraints_enabled: bool,
):
    policy = build_rust_lifecycle_policy(
        backend="rust",
        mode=mode,
        has_signatures_path=True,
        has_papers_path=True,
        preprocess=True,
        stage_enablement=_stages(
            ingest_preprocess=ingest_enabled,
            pair_featurization=pair_enabled,
            constraints=constraints_enabled,
        ),
    )
    expected = bool(mode == "train" and ingest_enabled and pair_enabled and constraints_enabled)
    assert policy.defer_signature_fields_to_rust is expected


_STAGE_CASES: list[dict[RuntimeStage, bool]] = [
    _stages(
        ingest_preprocess=ingest_preprocess,
        pair_featurization=pair_featurization,
        constraints=constraints,
    )
    for ingest_preprocess, pair_featurization, constraints in product((False, True), repeat=3)
]


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("mode", ["train", "inference"])
@pytest.mark.parametrize("has_signatures_path", [False, True])
@pytest.mark.parametrize("has_papers_path", [False, True])
@pytest.mark.parametrize("preprocess", [False, True])
@pytest.mark.parametrize("stage_enablement", _STAGE_CASES)
def test_policy_matches_legacy_decision_rules(
    backend: str,
    mode: str,
    has_signatures_path: bool,
    has_papers_path: bool,
    preprocess: bool,
    stage_enablement: dict[RuntimeStage, bool],
):
    actual = build_rust_lifecycle_policy(
        backend=backend,  # type: ignore[arg-type]
        mode=mode,
        has_signatures_path=has_signatures_path,
        has_papers_path=has_papers_path,
        preprocess=preprocess,
        stage_enablement=stage_enablement,
    )
    expected = _legacy_expected(
        backend=backend,
        mode=mode,
        has_signatures_path=has_signatures_path,
        has_papers_path=has_papers_path,
        preprocess=preprocess,
        stage_enablement=stage_enablement,
    )
    assert actual == expected
