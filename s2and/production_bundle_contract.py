"""Single manifest-file authority shared by production bundle writers/loaders."""

from __future__ import annotations

PRODUCTION_MODEL_BUNDLE_SCHEMA_VERSION = "s2and_production_model_bundle_v5"
PAIRWISE_PREDICTION_FIXTURE_SCHEMA_VERSION = "pairwise_prediction_fixture_v1"
PAIRWISE_PREDICTION_FIXTURE_TOLERANCE = 1e-10
CLUSTERER_CONFIG_SCHEMA_VERSION = "s2and_clusterer_config_v5"

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
    incremental_linker_version: str | None,
    include_pairwise_reproducibility: bool,
) -> dict[str, str]:
    """Return the exact supported manifest mapping for one bundle state."""

    files = dict(COMPLETE_MANIFEST_FILES if incremental_linker_version is not None else PAIRWISE_ONLY_MANIFEST_FILES)
    if include_pairwise_reproducibility:
        files.update(PAIRWISE_REPRODUCIBILITY_MANIFEST_FILES)
    return files


def production_bundle_status(incremental_linker_version: str | None) -> str:
    """Derive the user-facing bundle status from its sole state discriminator."""

    return "complete" if incremental_linker_version is not None else "pairwise_only"
