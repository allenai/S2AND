from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pytest

import s2and.production_model as production_model_module
from s2and.production_bundle import finalize_pairwise_eps
from s2and.production_model import _load_pairwise_staging_model
from scripts.production.model import release_pairwise
from tests.promoted_linking_helpers import write_synthetic_pairwise_bundle

ARTIFACT_HASHES = {
    "name_tuples_data_sha256": "a" * 64,
    "orcid_prefix_counts_data_sha256": "b" * 64,
}
REPO_ROOT = Path(__file__).resolve().parents[1]


class _PythonLightGBMScorer:
    def __init__(self, model_path: str) -> None:
        self.booster = lgb.Booster(model_file=model_path)

    def num_features(self) -> int:
        return int(self.booster.num_feature())

    def predict_proba_positive(self, features: np.ndarray, *, num_threads: int | None = None) -> np.ndarray:
        return np.asarray(self.booster.predict(features, num_threads=num_threads), dtype=np.float64)

    def predict_proba_positive_f32(self, features: np.ndarray, *, num_threads: int | None = None) -> np.ndarray:
        return self.predict_proba_positive(features, num_threads=num_threads)


def _bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(production_model_module, "_load_rust_lightgbm_booster", _PythonLightGBMScorer)
    monkeypatch.setattr(production_model_module, "canonical_artifact_hashes", lambda: dict(ARTIFACT_HASHES))
    bundle = tmp_path / "production_model_v9.9"
    write_synthetic_pairwise_bundle(
        bundle,
        artifact_hashes=ARTIFACT_HASHES,
        bundle_version="9.9",
        source_model_version="9.9",
    )
    return bundle


def test_module_entrypoint_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.production.model.release_pairwise", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "calibrate-eps" in completed.stdout
    assert "evaluate-clusters" in completed.stdout


def test_finalize_eps_changes_only_clusterer_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _bundle(tmp_path, monkeypatch)
    source_bytes = {
        path.relative_to(source).as_posix(): path.read_bytes() for path in source.rglob("*") if path.is_file()
    }
    source_manifest_sha256 = hashlib.sha256(source_bytes["manifest.json"]).hexdigest()
    output = tmp_path / "calibrated" / "production_model_v9.9"

    finalize_pairwise_eps(
        source_bundle_dir=source,
        output_bundle_dir=output,
        expected_manifest_sha256=source_manifest_sha256,
        expected_old_eps=0.5,
        new_eps=0.65,
    )

    assert {
        path.relative_to(source).as_posix(): path.read_bytes() for path in source.rglob("*") if path.is_file()
    } == source_bytes
    output_bytes = {
        path.relative_to(output).as_posix(): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }
    assert set(output_bytes) == set(source_bytes)
    for relpath in set(source_bytes) - {"clusterer.json", "manifest.json"}:
        assert output_bytes[relpath] == source_bytes[relpath]
    assert output_bytes["clusterer.json"] != source_bytes["clusterer.json"]
    assert output_bytes["manifest.json"] != source_bytes["manifest.json"]
    assert json.loads(output_bytes["clusterer.json"])["cluster_model"]["eps"] == 0.65
    assert _load_pairwise_staging_model(output).cluster_model.eps == 0.65


def test_finalize_eps_rejects_source_digest_and_old_eps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _bundle(tmp_path, monkeypatch)
    manifest_sha256 = hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        finalize_pairwise_eps(
            source_bundle_dir=source,
            output_bundle_dir=tmp_path / "bad-digest",
            expected_manifest_sha256="0" * 64,
            expected_old_eps=0.4,
            new_eps=0.5,
        )
    with pytest.raises(ValueError, match="source EPS mismatch"):
        finalize_pairwise_eps(
            source_bundle_dir=source,
            output_bundle_dir=tmp_path / "bad-eps",
            expected_manifest_sha256=manifest_sha256,
            expected_old_eps=0.3,
            new_eps=0.5,
        )


def test_pairwise_metric_contract_averages_once_and_uses_strict_threshold() -> None:
    metrics, probabilities = release_pairwise.pairwise_metrics(
        np.asarray([0, 1]),
        np.asarray([0.0, 1.0]),
        np.asarray([1.0, 0.0]),
    )

    np.testing.assert_array_equal(probabilities, np.asarray([0.5, 0.5]))
    assert metrics["auroc"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx(1 / 3)


def test_calibration_parser_has_no_test_manifest_route() -> None:
    parser = release_pairwise.build_parser()
    args = parser.parse_args(
        [
            "calibrate-eps",
            "--pairwise-model",
            "model",
            "--eps",
            "0.4",
            "0.5",
            "--output-json",
            "report.json",
        ]
    )
    assert args.command == "calibrate-eps"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "calibrate-eps",
                "--pairwise-model",
                "model",
                "--eps",
                "0.5",
                "--output-json",
                "report.json",
                "--manifest",
                "test.json",
            ]
        )


def test_cluster_evaluator_has_no_unused_ram_option() -> None:
    parser = release_pairwise.build_parser()
    common = [
        "evaluate-clusters",
        "--pairwise-model",
        "model",
        "--manifest",
        "manifest.json",
        "--expected-manifest-sha256",
        "a" * 64,
        "--unblind-record",
        "unblind.json",
        "--output-json",
        "report.json",
    ]
    assert parser.parse_args(common).command == "evaluate-clusters"
    with pytest.raises(SystemExit):
        parser.parse_args([*common, "--total-ram-bytes", "1024"])


def test_existing_output_fails_before_unblind(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_text("{}\n", encoding="utf-8")
    unblind = tmp_path / "unblind.json"

    with pytest.raises(FileExistsError, match="output already exists"):
        release_pairwise.evaluate_pairs(argparse.Namespace(output_json=output, unblind_record=unblind))
    assert not unblind.exists()


def test_unblind_record_is_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "unblind.json"
    kwargs: dict[str, Any] = {
        "manifest_path": tmp_path / "manifest.json",
        "manifest_sha256": "a" * 64,
        "model_path": tmp_path / "model",
    }
    release_pairwise._record_unblind(path, **kwargs)  # noqa: SLF001

    with pytest.raises(FileExistsError):
        release_pairwise._record_unblind(path, **kwargs)  # noqa: SLF001


def test_dataset_files_resolve_relative_to_manifest(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    pairs = inputs / "pairs.json"
    pairs.write_text("[]\n", encoding="utf-8")
    spec = {
        "name": "toy",
        "files": {
            "pairs": {
                "path": "../inputs/pairs.json",
                "sha256": hashlib.sha256(pairs.read_bytes()).hexdigest(),
            }
        },
    }

    resolved = release_pairwise._resolved_dataset_files(  # noqa: SLF001
        tmp_path / "manifests" / "manifest.json",
        spec,
        ("pairs",),
    )

    assert resolved == {"pairs": pairs.resolve()}


@pytest.mark.parametrize(
    "files",
    [
        {},
        {
            "pairs": {"path": "pairs.json", "sha256": "a" * 64},
            "extra": {"path": "extra.json", "sha256": "b" * 64},
        },
    ],
)
def test_dataset_files_require_exact_roles(tmp_path: Path, files: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="exact file roles"):
        release_pairwise._resolved_dataset_files(  # noqa: SLF001
            tmp_path / "manifest.json",
            {"name": "toy", "files": files},
            ("pairs",),
        )


def test_dataset_file_role_must_be_exact_object(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must contain exactly path and sha256"):
        release_pairwise._resolved_dataset_files(  # noqa: SLF001
            tmp_path / "manifest.json",
            {"name": "toy", "files": {"pairs": {"path": "pairs.json"}}},
            ("pairs",),
        )


def test_manifest_input_drift_fails_before_model_load_and_unblind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files: dict[str, dict[str, str]] = {}
    for role in ("signatures", "papers", "specter_embeddings", "pairs"):
        path = tmp_path / f"{role}.json"
        path.write_text("[]\n", encoding="utf-8")
        files[role] = {
            "path": path.name,
            "sha256": "0" * 64 if role == "pairs" else hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": release_pairwise.PAIR_MANIFEST_SCHEMA,
                "datasets": [{"name": "toy", "files": files}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        release_pairwise,
        "_load_pairwise_staging_model",
        lambda _: pytest.fail("model loaded before input verification"),
    )
    unblind = tmp_path / "unblind.json"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        release_pairwise.evaluate_pairs(
            argparse.Namespace(
                output_json=tmp_path / "report.json",
                manifest=manifest,
                expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                pairwise_model=tmp_path / "model",
                unblind_record=unblind,
            )
        )
    assert not unblind.exists()


def test_b3_aggregation_reports_macro_and_signature_weighted() -> None:
    aggregate = release_pairwise._aggregate_b3(  # noqa: SLF001
        {
            "small": {"precision": 1.0, "recall": 0.5, "f1": 0.5, "signature_count": 1},
            "large": {"precision": 0.0, "recall": 1.0, "f1": 1.0, "signature_count": 3},
        }
    )

    assert aggregate["dataset_macro"]["f1"] == pytest.approx(0.75)
    assert aggregate["signature_weighted"]["f1"] == pytest.approx(0.875)
    assert aggregate["signature_count"] == 4
