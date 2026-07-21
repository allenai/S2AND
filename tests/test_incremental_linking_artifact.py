from __future__ import annotations

import copy
import hashlib
import json
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import pytest

import s2and.incremental_linking.artifact as artifact_module
from s2and.consts import FEATURIZER_VERSION
from s2and.incremental_linking.artifact import (
    BOOSTER_FILENAME,
    METADATA_FILENAME,
    IncrementalLinkingArtifactMetadata,
    load_incremental_linking_artifact,
    save_incremental_linking_artifact,
)
from s2and.incremental_linking.contracts import (
    canonical_json_digest,
    retrieval_constraint_decision_policy_payload,
)
from s2and.incremental_linking.features import promoted_linker_feature_columns
from s2and.incremental_linking.logistic_gate import load_logistic_gate_config, logistic_gate_config
from tests.helpers import import_s2and_rust
from tests.promoted_linking_helpers import build_tiny_promoted_booster, synthetic_pairwise_bundle_binding

_HAS_RUST_LIGHTGBM, _RUST_LIGHTGBM_PAYLOAD = import_s2and_rust()
requires_rust_lightgbm = pytest.mark.skipif(
    not _HAS_RUST_LIGHTGBM,
    reason=f"s2and_rust unavailable: {_RUST_LIGHTGBM_PAYLOAD!r}",
)
_TEST_TARGET_SPEC = {"variant": "test"}


def _logistic_gate_config(link: bool = True) -> dict[str, object]:
    return logistic_gate_config(
        feature_names=("chosen_probability",),
        weights=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        bias=np.asarray([0.0, 0.0, 10.0 if link else -10.0], dtype=np.float64),
        missing_values=np.asarray([0.0], dtype=np.float64),
        calibration_mode="test",
    )


def _valid_metadata_payload() -> dict[str, Any]:
    columns = promoted_linker_feature_columns()
    return IncrementalLinkingArtifactMetadata.build(
        feature_columns=columns,
        gate_config=_logistic_gate_config(),
        prediction_fixture_matrix=((0.0,) * len(columns),),
        prediction_fixture_expected_probabilities=(0.5,),
        booster_sha256="a" * 64,
        lightgbm_version="test-version",
        target_spec_digest=canonical_json_digest(_TEST_TARGET_SPEC),
        pairwise_bundle_binding=synthetic_pairwise_bundle_binding(),
        audit_metadata={"nested": {"values": [1, 2]}},
    ).to_json_dict()


class _ConstantRustBooster:
    def predict_proba_positive_f32(
        self,
        matrix: np.ndarray,
        *,
        num_threads: int | None = None,
    ) -> np.ndarray:
        del num_threads
        return np.full(len(matrix), 0.5, dtype=np.float64)


def _write_fake_artifact(artifact_dir: Path) -> Path:
    artifact_dir.mkdir()
    booster_bytes = b"fake-lightgbm-booster"
    (artifact_dir / BOOSTER_FILENAME).write_bytes(booster_bytes)
    payload = _valid_metadata_payload()
    payload["booster_sha256"] = hashlib.sha256(booster_bytes).hexdigest()
    (artifact_dir / METADATA_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return artifact_dir


@requires_rust_lightgbm
def test_save_and_load_incremental_linking_artifact_round_trip(tmp_path: Path) -> None:
    booster, fixture = build_tiny_promoted_booster()
    artifact_dir = tmp_path / "artifact"
    metadata = save_incremental_linking_artifact(
        booster,
        artifact_dir,
        prediction_fixture_matrix=fixture,
        gate_config=_logistic_gate_config(),
        target_spec=_TEST_TARGET_SPEC,
        pairwise_bundle_binding=synthetic_pairwise_bundle_binding(),
        audit_metadata={
            "artifact_version": "v1.2",
            "pairwise_model": {"version": "1.2"},
        },
    )

    assert (artifact_dir / BOOSTER_FILENAME).exists()
    assert (artifact_dir / METADATA_FILENAME).exists()
    loaded = load_incremental_linking_artifact(artifact_dir)

    assert loaded.metadata.feature_columns == promoted_linker_feature_columns()
    assert loaded.metadata.feature_schema_digest == metadata.feature_schema_digest
    assert loaded.metadata.target_spec_digest == canonical_json_digest(_TEST_TARGET_SPEC)
    assert loaded.metadata.audit_metadata["artifact_version"] == "v1.2"
    assert loaded.metadata.audit_metadata["pairwise_model"]["version"] == "1.2"
    assert (
        loaded.metadata.to_json_dict()["audit_metadata"]["runtime_decision_policy"]
        == retrieval_constraint_decision_policy_payload()
    )
    np.testing.assert_allclose(
        loaded.predict_probabilities(fixture),
        np.asarray(metadata.prediction_fixture_expected_probabilities),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        loaded.predict_probabilities(fixture, max_rows_per_chunk=1),
        np.asarray(metadata.prediction_fixture_expected_probabilities),
        rtol=1e-10,
        atol=1e-10,
    )


def test_save_rejects_reserved_audit_metadata_binding_key(tmp_path: Path) -> None:
    booster, fixture = build_tiny_promoted_booster()
    with pytest.raises(ValueError, match="'pairwise_bundle_binding' is reserved"):
        save_incremental_linking_artifact(
            booster,
            tmp_path,
            prediction_fixture_matrix=fixture,
            gate_config=_logistic_gate_config(),
            target_spec=_TEST_TARGET_SPEC,
            pairwise_bundle_binding=synthetic_pairwise_bundle_binding(),
            audit_metadata={"pairwise_bundle_binding": synthetic_pairwise_bundle_binding()},
        )
    assert not (tmp_path / METADATA_FILENAME).exists()


def test_save_rejects_empty_pairwise_bundle_binding(tmp_path: Path) -> None:
    booster, fixture = build_tiny_promoted_booster()
    with pytest.raises(ValueError, match="pairwise_bundle_binding is required"):
        save_incremental_linking_artifact(
            booster,
            tmp_path,
            prediction_fixture_matrix=fixture,
            gate_config=_logistic_gate_config(),
            target_spec=_TEST_TARGET_SPEC,
            pairwise_bundle_binding={},
        )
    assert not (tmp_path / METADATA_FILENAME).exists()


@requires_rust_lightgbm
def test_load_rejects_nested_audit_binding(tmp_path: Path) -> None:
    booster, fixture = build_tiny_promoted_booster()
    artifact_dir = tmp_path / "artifact"
    save_incremental_linking_artifact(
        booster,
        artifact_dir,
        prediction_fixture_matrix=fixture,
        gate_config=_logistic_gate_config(),
        target_spec=_TEST_TARGET_SPEC,
        pairwise_bundle_binding=synthetic_pairwise_bundle_binding(),
    )
    metadata_path = artifact_dir / METADATA_FILENAME
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["audit_metadata"]["pairwise_bundle_binding"] = {"legacy": "historical-copy"}
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="audit_metadata key 'pairwise_bundle_binding' is reserved"):
        load_incremental_linking_artifact(artifact_dir)


@requires_rust_lightgbm
def test_artifact_publication_failure_leaves_target_absent_and_is_retry_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    booster, fixture = build_tiny_promoted_booster()
    artifact_dir = tmp_path / "artifact"
    real_replace = artifact_module.os.replace
    failed = False

    def fail_publish_once(source: str | Path, destination: str | Path) -> None:
        nonlocal failed
        if not failed and Path(destination) == artifact_dir:
            failed = True
            raise OSError("injected artifact publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(artifact_module.os, "replace", fail_publish_once)
    with pytest.raises(OSError, match="injected artifact publication failure"):
        save_incremental_linking_artifact(
            booster,
            artifact_dir,
            prediction_fixture_matrix=fixture,
            gate_config=_logistic_gate_config(),
            target_spec=_TEST_TARGET_SPEC,
            pairwise_bundle_binding=synthetic_pairwise_bundle_binding(),
        )
    assert not artifact_dir.exists()

    monkeypatch.setattr(artifact_module.os, "replace", real_replace)
    save_incremental_linking_artifact(
        booster,
        artifact_dir,
        prediction_fixture_matrix=fixture,
        gate_config=_logistic_gate_config(),
        target_spec=_TEST_TARGET_SPEC,
        pairwise_bundle_binding=synthetic_pairwise_bundle_binding(),
    )
    load_incremental_linking_artifact(artifact_dir)


@requires_rust_lightgbm
def test_artifact_publication_requires_a_new_directory(tmp_path: Path) -> None:
    booster, fixture = build_tiny_promoted_booster()
    artifact_dir = tmp_path / "artifact"
    kwargs: dict[str, Any] = {
        "prediction_fixture_matrix": fixture,
        "gate_config": _logistic_gate_config(),
        "target_spec": _TEST_TARGET_SPEC,
        "pairwise_bundle_binding": synthetic_pairwise_bundle_binding(),
    }
    save_incremental_linking_artifact(booster, artifact_dir, **kwargs)
    original_metadata = (artifact_dir / METADATA_FILENAME).read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        save_incremental_linking_artifact(booster, artifact_dir, **kwargs)
    with pytest.raises(FileExistsError, match="already exists"):
        save_incremental_linking_artifact(
            booster,
            artifact_dir,
            prediction_fixture_matrix=fixture,
            gate_config=_logistic_gate_config(link=False),
            target_spec=_TEST_TARGET_SPEC,
            pairwise_bundle_binding=synthetic_pairwise_bundle_binding(),
        )
    assert (artifact_dir / METADATA_FILENAME).read_bytes() == original_metadata


def test_concurrent_conflicting_artifact_publication_has_one_immutable_winner(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    staging_dirs = (tmp_path / "staging-a", tmp_path / "staging-b")
    payloads = (b"first", b"second")
    for staging_dir, payload in zip(staging_dirs, payloads, strict=True):
        staging_dir.mkdir()
        (staging_dir / "payload").write_bytes(payload)
    start = threading.Barrier(len(staging_dirs))

    def publish(staging_dir: Path) -> str:
        start.wait(timeout=5.0)
        try:
            artifact_module._publish_immutable_artifact(staging_dir, artifact_dir)
        except FileExistsError:
            return "conflict"
        return "published"

    with ThreadPoolExecutor(max_workers=len(staging_dirs)) as executor:
        outcomes = list(executor.map(publish, staging_dirs))

    assert sorted(outcomes) == ["conflict", "published"]
    assert (artifact_dir / "payload").read_bytes() in payloads


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
            target_spec=_TEST_TARGET_SPEC,
            pairwise_bundle_binding=synthetic_pairwise_bundle_binding(),
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
    artifact_dir = tmp_path / "artifact"
    save_incremental_linking_artifact(
        booster,
        artifact_dir,
        prediction_fixture_matrix=fixture,
        gate_config=_logistic_gate_config(),
        target_spec=_TEST_TARGET_SPEC,
        pairwise_bundle_binding=synthetic_pairwise_bundle_binding(),
    )
    metadata_path = artifact_dir / METADATA_FILENAME
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload[field_name] = "bad"
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_incremental_linking_artifact(artifact_dir)


def test_metadata_v4_rejects_missing_top_level_field() -> None:
    payload = _valid_metadata_payload()
    del payload["booster_sha256"]

    with pytest.raises(ValueError, match="fields do not match the v4 schema"):
        IncrementalLinkingArtifactMetadata.from_mapping(payload)


def test_metadata_v4_rejects_unknown_top_level_fields() -> None:
    payload = _valid_metadata_payload()
    payload["future_implicit_default"] = True

    with pytest.raises(ValueError, match="unknown=.*future_implicit_default"):
        IncrementalLinkingArtifactMetadata.from_mapping(payload)


def test_metadata_v4_rejects_v3_schema() -> None:
    payload = _valid_metadata_payload()
    payload["schema_version"] = "incremental_linking_artifact_v3"

    with pytest.raises(ValueError, match="Unsupported incremental linker artifact schema_version"):
        IncrementalLinkingArtifactMetadata.from_mapping(payload)


@pytest.mark.parametrize("invalid_value", ("A" * 64, "a" * 63, "not-a-digest"))
def test_metadata_rejects_invalid_target_spec_digest(invalid_value: str) -> None:
    payload = _valid_metadata_payload()
    payload["target_spec_digest"] = invalid_value

    with pytest.raises(ValueError, match="target_spec_digest is not a SHA-256"):
        IncrementalLinkingArtifactMetadata.from_mapping(payload)


@pytest.mark.parametrize(
    "invalid_value",
    (str(FEATURIZER_VERSION), float(FEATURIZER_VERSION), True, False),
)
def test_metadata_rejects_non_integer_pairwise_featurizer_version(invalid_value: object) -> None:
    payload = _valid_metadata_payload()
    payload["pairwise_bundle_binding"]["featurizer_version"] = invalid_value

    with pytest.raises(ValueError, match="featurizer_version must be an integer"):
        IncrementalLinkingArtifactMetadata.from_mapping(payload)


@pytest.mark.parametrize(
    ("field_name", "empty_value"),
    (
        ("schema_version", ""),
        ("booster_sha256", ""),
        ("lightgbm_version", ""),
        ("target_spec_digest", ""),
        ("feature_columns", []),
        ("prediction_fixture_matrix", []),
        ("prediction_fixture_expected_probabilities", []),
        ("gate_config", {}),
        ("pairwise_bundle_binding", {}),
    ),
)
def test_metadata_v4_rejects_empty_required_values(field_name: str, empty_value: object) -> None:
    payload = _valid_metadata_payload()
    payload[field_name] = empty_value

    with pytest.raises(ValueError):
        IncrementalLinkingArtifactMetadata.from_mapping(payload)


@pytest.mark.parametrize(
    "case",
    (
        "wrong_width",
        "probability_count",
        "non_numeric_matrix",
        "nonfinite_matrix",
        "nonfinite_probability",
        "boolean_matrix",
        "out_of_range_probability",
    ),
)
def test_metadata_v4_rejects_invalid_prediction_fixtures(case: str) -> None:
    payload = _valid_metadata_payload()
    matrix = payload["prediction_fixture_matrix"]
    probabilities = payload["prediction_fixture_expected_probabilities"]
    assert isinstance(matrix, list)
    assert isinstance(probabilities, list)
    assert isinstance(matrix[0], list)

    if case == "wrong_width":
        matrix[0].pop()
    elif case == "probability_count":
        matrix.append(list(matrix[0]))
    elif case == "non_numeric_matrix":
        matrix[0][0] = "0"
    elif case == "nonfinite_matrix":
        matrix[0][0] = float("nan")
    elif case == "nonfinite_probability":
        probabilities[0] = float("inf")
    elif case == "boolean_matrix":
        matrix[0][0] = True
    elif case == "out_of_range_probability":
        probabilities[0] = 1.01
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(case)

    with pytest.raises(ValueError):
        IncrementalLinkingArtifactMetadata.from_mapping(payload)


def test_load_always_verifies_booster_hash_before_loading_scorer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = _write_fake_artifact(tmp_path / "artifact")
    load_count = 0

    def load_booster(booster_path: Path) -> _ConstantRustBooster:
        nonlocal load_count
        assert booster_path == artifact_dir.resolve() / BOOSTER_FILENAME
        load_count += 1
        return _ConstantRustBooster()

    monkeypatch.setattr(artifact_module, "_load_rust_lightgbm_booster", load_booster)
    loaded = load_incremental_linking_artifact(artifact_dir)
    assert loaded.artifact_dir == artifact_dir.resolve()
    assert load_count == 1

    (artifact_dir / BOOSTER_FILENAME).write_bytes(b"mutated-booster")
    with pytest.raises(ValueError, match="booster_sha256 mismatch"):
        load_incremental_linking_artifact(artifact_dir)
    assert load_count == 1


def test_load_rejects_deleted_booster_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = _write_fake_artifact(tmp_path / "artifact")
    (artifact_dir / BOOSTER_FILENAME).unlink()
    monkeypatch.setattr(
        artifact_module,
        "_load_rust_lightgbm_booster",
        lambda booster_path: _ConstantRustBooster(),
    )

    with pytest.raises(FileNotFoundError):
        load_incremental_linking_artifact(artifact_dir)


def test_metadata_and_gate_state_are_transitively_immutable() -> None:
    payload = _valid_metadata_payload()
    metadata = IncrementalLinkingArtifactMetadata.from_mapping(payload)
    gate = load_logistic_gate_config(metadata.gate_config)

    payload["audit_metadata"]["nested"]["values"].append(3)
    assert metadata.audit_metadata["nested"]["values"] == (1, 2)

    with pytest.raises(TypeError):
        cast(Any, metadata.audit_metadata)["new"] = "value"
    with pytest.raises(TypeError):
        metadata.audit_metadata["nested"]["new"] = "value"
    with pytest.raises(TypeError):
        cast(Any, metadata.gate_config)["model_type"] = "changed"
    with pytest.raises(TypeError):
        cast(Any, metadata.pairwise_bundle_binding)["normalization_version"] = 0
    assert metadata.audit_metadata["nested"]["values"] == (1, 2)

    exported = metadata.to_json_dict()
    exported["audit_metadata"]["nested"]["values"].append(4)
    assert metadata.audit_metadata["nested"]["values"] == (1, 2)

    for array in (gate.weights, gate.bias, gate.missing_values):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 1.0
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        cast(Any, gate.error_weights)["false_link"] = 0.0


def test_deepcopy_shares_only_immutable_artifact_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = _write_fake_artifact(tmp_path / "artifact")
    monkeypatch.setattr(
        artifact_module,
        "_load_rust_lightgbm_booster",
        lambda booster_path: _ConstantRustBooster(),
    )
    artifact = load_incremental_linking_artifact(artifact_dir)
    fixture = np.asarray(artifact.metadata.prediction_fixture_matrix, dtype=np.float32)
    before = artifact.predict_probabilities(fixture)

    copied = copy.deepcopy(artifact)

    assert copied is artifact
    assert copy.deepcopy(artifact.metadata) is artifact.metadata
    np.testing.assert_array_equal(copied.predict_probabilities(fixture), before)


def test_pickle_reload_is_independent_of_current_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = _write_fake_artifact(tmp_path / "artifact")
    monkeypatch.setattr(
        artifact_module,
        "_load_rust_lightgbm_booster",
        lambda booster_path: _ConstantRustBooster(),
    )
    monkeypatch.chdir(tmp_path)
    artifact = load_incremental_linking_artifact(Path("artifact"))
    serialized = pickle.dumps(artifact)

    other_dir = tmp_path / "other-cwd"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)
    restored = pickle.loads(serialized)

    assert restored.artifact_dir == artifact_dir.resolve()
    fixture = np.asarray(restored.metadata.prediction_fixture_matrix, dtype=np.float32)
    np.testing.assert_array_equal(restored.predict_probabilities(fixture), np.asarray([0.5]))
