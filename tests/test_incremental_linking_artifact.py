from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from s2and.incremental_linking.artifact import (
    BOOSTER_FILENAME,
    METADATA_FILENAME,
    load_incremental_linking_artifact,
    save_incremental_linking_artifact,
)
from s2and.incremental_linking.contracts import (
    retrieval_constraint_decision_policy_payload,
    retrieval_stack_contract_payload,
    validate_required_rust_capabilities,
)
from s2and.incremental_linking.features import promoted_linker_feature_columns
from s2and.incremental_linking.logistic_gate import logistic_gate_config
from tests.helpers import import_s2and_rust
from tests.promoted_linking_helpers import build_tiny_promoted_booster

_HAS_RUST_LIGHTGBM, _RUST_LIGHTGBM_PAYLOAD = import_s2and_rust(
    required_module_attrs=("RustLightGBMBooster",),
    prefer_site_packages=True,
)
requires_rust_lightgbm = pytest.mark.skipif(
    not _HAS_RUST_LIGHTGBM,
    reason=f"RustLightGBMBooster unavailable: {_RUST_LIGHTGBM_PAYLOAD!r}",
)


def _logistic_gate_config(link: bool = True) -> dict[str, object]:
    return logistic_gate_config(
        feature_names=("chosen_probability",),
        weights=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        bias=np.asarray([0.0, 0.0, 10.0 if link else -10.0], dtype=np.float64),
        missing_values=np.asarray([0.0], dtype=np.float64),
        calibration_mode="test",
    )


@requires_rust_lightgbm
def test_save_and_load_incremental_linking_artifact_round_trip(tmp_path: Path) -> None:
    booster, fixture = build_tiny_promoted_booster()
    metadata = save_incremental_linking_artifact(
        booster,
        tmp_path,
        prediction_fixture_matrix=fixture,
        gate_config=_logistic_gate_config(),
        audit_metadata={"artifact_version": "v1.2", "pairwise_model": {"version": "1.2"}},
    )

    assert (tmp_path / BOOSTER_FILENAME).exists()
    assert (tmp_path / METADATA_FILENAME).exists()
    loaded = load_incremental_linking_artifact(tmp_path)

    assert loaded.metadata.feature_columns == promoted_linker_feature_columns()
    assert loaded.metadata.feature_schema_digest == metadata.feature_schema_digest
    assert loaded.metadata.audit_metadata["artifact_version"] == "v1.2"
    assert loaded.metadata.audit_metadata["pairwise_model"]["version"] == "1.2"
    assert loaded.metadata.audit_metadata["runtime_decision_policy"] == retrieval_constraint_decision_policy_payload()
    np.testing.assert_allclose(
        loaded.predict_probabilities(fixture),
        np.asarray(metadata.prediction_fixture_expected_probabilities),
        rtol=1e-10,
        atol=1e-10,
    )


def test_save_incremental_linking_artifact_requires_lightgbm_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    booster, fixture = build_tiny_promoted_booster()
    monkeypatch.delattr(lgb, "__version__", raising=False)

    with pytest.raises(RuntimeError, match="lightgbm.__version__ is required"):
        save_incremental_linking_artifact(
            booster,
            tmp_path,
            prediction_fixture_matrix=fixture,
            gate_config=_logistic_gate_config(),
        )

    assert not (tmp_path / METADATA_FILENAME).exists()


@pytest.mark.parametrize(
    ("field_name", "message"),
    (
        ("feature_schema_digest", "feature_schema_digest mismatch"),
        ("production_contract_digest", "production_contract_digest mismatch"),
        ("retrieval_stack_digest", "retrieval_stack_digest mismatch"),
    ),
)
def test_load_incremental_linking_artifact_rejects_digest_drift(
    tmp_path: Path,
    field_name: str,
    message: str,
) -> None:
    booster, fixture = build_tiny_promoted_booster()
    save_incremental_linking_artifact(
        booster,
        tmp_path,
        prediction_fixture_matrix=fixture,
        gate_config=_logistic_gate_config(),
    )
    metadata_path = tmp_path / METADATA_FILENAME
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload[field_name] = "bad"
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_incremental_linking_artifact(tmp_path)


def test_retrieval_stack_contract_records_constraint_decision_policy() -> None:
    payload = retrieval_stack_contract_payload(retrieval_top_k=25)

    assert payload["candidate_filter_policy"] == "post_retrieval_constraint_row_policy"
    assert payload["orcid_policy"] == "return_all_matches_force_link_exempt_from_disallow_veto"
    assert payload["constraint_decision_policy"] == retrieval_constraint_decision_policy_payload()


def test_validate_required_rust_capabilities_rejects_missing_names() -> None:
    with pytest.raises(RuntimeError, match="Missing required Rust capabilities"):
        validate_required_rust_capabilities(
            ("incremental_linking_pair_plan_v1",),
            available=(),
        )
