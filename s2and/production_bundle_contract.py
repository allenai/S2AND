"""Single manifest and lifecycle authority for production model bundles."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, cast

PRODUCTION_MODEL_KIND = "s2and_model"
PAIRWISE_PREDICTION_FIXTURE_TOLERANCE = 1e-10

BundleKind = Literal["pairwise_only", "complete"]
EpsCalibration = Literal["pending", "calibrated"]

PAIRWISE_ONLY_BUNDLE_KIND: BundleKind = "pairwise_only"
COMPLETE_BUNDLE_KIND: BundleKind = "complete"
PENDING_EPS_CALIBRATION: EpsCalibration = "pending"
CALIBRATED_EPS_CALIBRATION: EpsCalibration = "calibrated"
PENDING_PAIRWISE_EPS = 0.5

VALID_BUNDLE_STATES: frozenset[tuple[BundleKind, EpsCalibration]] = frozenset(
    {
        (PAIRWISE_ONLY_BUNDLE_KIND, PENDING_EPS_CALIBRATION),
        (PAIRWISE_ONLY_BUNDLE_KIND, CALIBRATED_EPS_CALIBRATION),
        (COMPLETE_BUNDLE_KIND, CALIBRATED_EPS_CALIBRATION),
    }
)

PAIRWISE_ONLY_MANIFEST_FILES = {
    "clusterer_config": "clusterer.json",
    "pairwise_main_fixture": "pairwise/main_prediction_fixture.json",
    "pairwise_main_model": "pairwise/main.lgb",
    "pairwise_nameless_fixture": "pairwise/nameless_prediction_fixture.json",
    "pairwise_nameless_model": "pairwise/nameless.lgb",
}
PAIRWISE_REPRODUCIBILITY_MANIFEST_FILES = {
    "pairwise_training_config": "reproducibility/pairwise_training_config.json",
    "pairwise_training_summary": "reproducibility/pairwise_training_summary.json",
}
COMPLETE_MANIFEST_FILES = {
    **PAIRWISE_ONLY_MANIFEST_FILES,
    "incremental_linker_booster": "incremental_linker/booster.lgb",
    "incremental_linker_metadata": "incremental_linker/metadata.json",
    "incremental_linker_training_target": "reproducibility/incremental_linker_training_target.json",
}


def production_manifest_files(
    *,
    bundle_kind: BundleKind,
    include_pairwise_reproducibility: bool,
) -> dict[str, str]:
    """Return the exact supported manifest mapping for one bundle state."""

    files = dict(COMPLETE_MANIFEST_FILES if bundle_kind == COMPLETE_BUNDLE_KIND else PAIRWISE_ONLY_MANIFEST_FILES)
    if include_pairwise_reproducibility:
        files.update(PAIRWISE_REPRODUCIBILITY_MANIFEST_FILES)
    return files


def infer_bundle_kind(checksum_paths: Iterable[str]) -> BundleKind:
    """Infer the bundle role from its exact checksum inventory."""

    observed = set(checksum_paths)
    for bundle_kind in (PAIRWISE_ONLY_BUNDLE_KIND, COMPLETE_BUNDLE_KIND):
        for include_reproducibility in (False, True):
            expected = set(
                production_manifest_files(
                    bundle_kind=bundle_kind,
                    include_pairwise_reproducibility=include_reproducibility,
                ).values()
            )
            if observed == expected:
                return bundle_kind
    raise ValueError("Production model bundle checksum inventory is neither pairwise-only nor complete")


def require_bundle_state(bundle_kind: object, eps_calibration: object) -> tuple[BundleKind, EpsCalibration]:
    """Validate and return one supported bundle lifecycle state."""

    state = (bundle_kind, eps_calibration)
    if state not in VALID_BUNDLE_STATES:
        raise ValueError(
            "Unsupported production bundle lifecycle state: "
            f"bundle_kind={bundle_kind!r} eps_calibration={eps_calibration!r}"
        )
    return cast(BundleKind, bundle_kind), cast(EpsCalibration, eps_calibration)
