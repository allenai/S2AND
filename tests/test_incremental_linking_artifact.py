from __future__ import annotations

import copy
import hashlib
import json
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import s2and.incremental_linking.artifact as artifact_module
from s2and import __version__
from s2and.incremental_linking.artifact import (
    BOOSTER_FILENAME,
    INCREMENTAL_LINKER_KIND,
    METADATA_FILENAME,
    load_incremental_linking_artifact,
    save_incremental_linking_artifact,
)
from s2and.incremental_linking.contracts import canonical_json_digest
from s2and.incremental_linking.features import promoted_linker_feature_columns
from tests.promoted_linking_helpers import (
    build_tiny_promoted_booster,
    synthetic_pairwise_bundle_binding,
    tiny_logistic_gate_config,
)

_TEST_TARGET_SPEC = {"variant": "test"}


def _valid_metadata_payload() -> dict[str, Any]:
    return {
        "booster_sha256": "a" * 64,
        "gate_config": tiny_logistic_gate_config(),
        "generated_by_runtime": __version__,
        "kind": INCREMENTAL_LINKER_KIND,
        "pairwise_bundle_binding_digest": canonical_json_digest(synthetic_pairwise_bundle_binding()),
        "retrieval_top_k": 25,
        "target_spec_digest": canonical_json_digest(_TEST_TARGET_SPEC),
    }


class _ConstantRustBooster:
    def __init__(self, feature_count: int | None = None) -> None:
        self.feature_count = feature_count or len(promoted_linker_feature_columns())

    def num_features(self) -> int:
        return self.feature_count

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


def test_artifact_lifecycle_preserves_predictions_and_immutable_identity(tmp_path: Path, monkeypatch) -> None:
    booster, fixture = build_tiny_promoted_booster()
    artifact_dir = tmp_path / "artifact"
    binding = synthetic_pairwise_bundle_binding()
    with pytest.raises(ValueError, match="pairwise_bundle_binding is required"):
        save_incremental_linking_artifact(
            booster,
            artifact_dir,
            gate_config=tiny_logistic_gate_config(),
            target_spec=_TEST_TARGET_SPEC,
            pairwise_bundle_binding={},
        )
    assert not artifact_dir.exists()
    metadata = save_incremental_linking_artifact(
        booster,
        artifact_dir,
        gate_config=tiny_logistic_gate_config(),
        target_spec=_TEST_TARGET_SPEC,
        pairwise_bundle_binding=binding,
    )

    assert set(metadata) == {
        "booster_sha256",
        "gate_config",
        "generated_by_runtime",
        "kind",
        "pairwise_bundle_binding_digest",
        "retrieval_top_k",
        "target_spec_digest",
    }
    assert metadata["kind"] == INCREMENTAL_LINKER_KIND
    assert metadata["generated_by_runtime"] == __version__
    assert metadata["pairwise_bundle_binding_digest"] == canonical_json_digest(binding)
    assert metadata["target_spec_digest"] == canonical_json_digest(_TEST_TARGET_SPEC)

    loaded = load_incremental_linking_artifact(artifact_dir)
    assert loaded.feature_columns == promoted_linker_feature_columns()
    assert loaded.retrieval_top_k == 25
    assert loaded.pairwise_bundle_binding_digest == canonical_json_digest(binding)
    assert loaded.target_spec_digest == canonical_json_digest(_TEST_TARGET_SPEC)
    expected = np.asarray(booster.predict(fixture), dtype=np.float64)
    np.testing.assert_allclose(loaded.predict_probabilities(fixture), expected, rtol=1e-10, atol=1e-10)
    assert copy.deepcopy(loaded) is loaded
    with pytest.raises(FileExistsError, match="already exists"):
        save_incremental_linking_artifact(
            booster,
            artifact_dir,
            gate_config=tiny_logistic_gate_config(),
            target_spec=_TEST_TARGET_SPEC,
            pairwise_bundle_binding=binding,
        )
    assert json.loads((artifact_dir / METADATA_FILENAME).read_text()) == metadata
    monkeypatch.chdir(tmp_path)
    serialized = pickle.dumps(load_incremental_linking_artifact(Path("artifact")))
    other_dir = tmp_path / "other-cwd"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)
    restored = pickle.loads(serialized)
    assert restored.artifact_dir == artifact_dir.resolve()
    np.testing.assert_allclose(restored.predict_probabilities(fixture), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        loaded.predict_probabilities(fixture, max_rows_per_chunk=1),
        expected,
        rtol=1e-10,
        atol=1e-10,
    )


def test_artifact_publication_failure_leaves_target_absent_and_is_retry_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    booster, _fixture = build_tiny_promoted_booster()
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

    def save() -> dict[str, Any]:
        return save_incremental_linking_artifact(
            booster,
            artifact_dir,
            gate_config=tiny_logistic_gate_config(),
            target_spec=_TEST_TARGET_SPEC,
            pairwise_bundle_binding=synthetic_pairwise_bundle_binding(),
        )

    with pytest.raises(OSError, match="injected artifact publication failure"):
        save()
    assert not artifact_dir.exists()

    monkeypatch.setattr(artifact_module.os, "replace", real_replace)
    save()
    load_incremental_linking_artifact(artifact_dir)


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


def test_metadata_rejects_invalid_fields() -> None:
    for field_name in ("booster_sha256", "pairwise_bundle_binding_digest", "target_spec_digest"):
        payload = _valid_metadata_payload()
        payload[field_name] = "invalid"
        with pytest.raises(ValueError, match=f"{field_name} is not a SHA-256"):
            artifact_module._validated_metadata(payload)

    for invalid_value in (0, 1.5, True):
        payload = _valid_metadata_payload()
        payload["retrieval_top_k"] = invalid_value
        with pytest.raises(ValueError, match="retrieval_top_k must be a positive integer"):
            artifact_module._validated_metadata(payload)

    missing = _valid_metadata_payload()
    del missing["booster_sha256"]
    with pytest.raises(ValueError, match="fields do not match the current contract.*booster_sha256"):
        artifact_module._validated_metadata(missing)

    unknown = _valid_metadata_payload()
    unknown["future_implicit_default"] = True
    with pytest.raises(ValueError, match="unknown=.*future_implicit_default"):
        artifact_module._validated_metadata(unknown)

    payload = _valid_metadata_payload()
    payload["gate_config"] = {}
    with pytest.raises(ValueError, match="gate_config must be a nonempty object"):
        artifact_module._validated_metadata(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("kind", "not-a-linker", "kind must be"),
        ("generated_by_runtime", "0.0.0", "runtime mismatch"),
    ),
    ids=("wrong-kind", "runtime-mismatch"),
)
def test_load_rejects_wrong_identity_before_booster_io(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    artifact_dir = _write_fake_artifact(tmp_path / "artifact")
    metadata_path = artifact_dir / METADATA_FILENAME
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload[field] = value
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    (artifact_dir / BOOSTER_FILENAME).unlink()

    with pytest.raises(ValueError, match=message):
        load_incremental_linking_artifact(artifact_dir)


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


def test_load_rejects_wrong_booster_feature_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = _write_fake_artifact(tmp_path / "artifact")
    monkeypatch.setattr(
        artifact_module,
        "_load_rust_lightgbm_booster",
        lambda booster_path: _ConstantRustBooster(len(promoted_linker_feature_columns()) - 1),
    )

    with pytest.raises(ValueError, match="booster feature count mismatch"):
        load_incremental_linking_artifact(artifact_dir)
