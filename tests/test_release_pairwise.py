from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pytest

import s2and.production_model as production_model_module
from s2and.production_bundle import finalize_pairwise_eps, write_production_manifest
from s2and.production_model import _load_pairwise_staging_model
from scripts.production.model import release_pairwise
from tests.promoted_linking_helpers import write_synthetic_pairwise_bundle

ARTIFACT_HASHES = {
    "name_tuples_data_sha256": "a" * 64,
    "orcid_prefix_counts_data_sha256": "b" * 64,
}
REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_evaluation_bindings(tmp_path: Path) -> dict[str, object]:
    quality_policy = tmp_path / "quality_policy.json"
    quality_policy.write_text("{}\n", encoding="utf-8")
    model = tmp_path / "complete_model"
    model.mkdir()
    model_manifest = model / "manifest.json"
    model_manifest.write_text("{}\n", encoding="utf-8")
    return {
        "model": model,
        "expected_model_manifest_sha256": _sha256(model_manifest),
        "quality_policy": quality_policy,
        "expected_quality_policy_sha256": _sha256(quality_policy),
    }


def _write_evaluation_manifest(
    tmp_path: Path,
    *,
    schema_version: str,
    contents: dict[str, object],
) -> tuple[Path, str]:
    files: dict[str, dict[str, str]] = {}
    for role, content in contents.items():
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps(content), encoding="utf-8")
        files[role] = {"path": path.name, "sha256": _sha256(path)}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "datasets": [{"name": "toy", "files": files}],
            }
        ),
        encoding="utf-8",
    )
    return manifest, _sha256(manifest)


def _write_manifest_dataset(root: Path, name: str, contents: dict[str, object]) -> dict[str, object]:
    files: dict[str, dict[str, str]] = {}
    for role, content in contents.items():
        suffix = ".csv" if isinstance(content, str) else ".json"
        path = root / f"{name}_{role}{suffix}"
        if isinstance(content, str):
            path.write_text(content, encoding="utf-8")
        else:
            path.write_text(json.dumps(content), encoding="utf-8")
        files[role] = {"path": path.name, "sha256": _sha256(path)}
    return {"name": name, "files": files}


def _write_training_input_fixture(
    tmp_path: Path,
    *,
    fixed_train_rows: str = "f1,f2,YES\n",
    fixed_test_pair: tuple[str, str] = ("f5", "f6"),
    random_test_pair: tuple[str, str] = ("rt1", "rt2"),
) -> tuple[Path, str, Path]:
    test_root = tmp_path / "sealed"
    pair_root = test_root / "pair"
    cluster_root = test_root / "cluster"
    pair_root.mkdir(parents=True)
    cluster_root.mkdir()
    pair_datasets = []
    for name, pair in (("random", random_test_pair), ("fixed", fixed_test_pair)):
        signature_ids = sorted(set(pair))
        pair_datasets.append(
            _write_manifest_dataset(
                pair_root,
                name,
                {
                    "signatures": {value: {} for value in signature_ids},
                    "papers": {},
                    "specter_embeddings": {},
                    "pairs": [
                        {"signature_id_1": pair[0], "signature_id_2": pair[1], "label": 1},
                    ],
                },
            )
        )
    pair_manifest = pair_root / "manifest.json"
    pair_manifest.write_text(
        json.dumps({"schema_version": release_pairwise.PAIR_MANIFEST_SCHEMA, "datasets": pair_datasets}),
        encoding="utf-8",
    )

    cluster_dataset = _write_manifest_dataset(
        cluster_root,
        "random",
        {
            "signatures": {"ct1": {}, "ct2": {}},
            "papers": {},
            "specter_embeddings": {},
            "clusters": {"c": {"signature_ids": ["ct1", "ct2"]}},
            "blocks": {"b": ["ct1", "ct2"]},
        },
    )
    cluster_manifest = cluster_root / "manifest.json"
    cluster_manifest.write_text(
        json.dumps(
            {
                "schema_version": release_pairwise.CLUSTER_MANIFEST_SCHEMA,
                "datasets": [cluster_dataset],
            }
        ),
        encoding="utf-8",
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    random_dataset = _write_manifest_dataset(
        source_root,
        "random",
        {
            "signatures": {"r1": {}, "r2": {}},
            "papers": {},
            "specter_embeddings": {},
            "clusters": {"c": {"signature_ids": ["r1", "r2"]}},
        },
    )
    random_dataset["split_mode"] = "random_blocks"
    fixed_dataset = _write_manifest_dataset(
        source_root,
        "fixed",
        {
            "signatures": {"f1": {}, "f2": {}, "f3": {}, "f4": {}},
            "papers": {},
            "specter_embeddings": {},
            "train_pairs": "signature_id_1,signature_id_2,label\n" + fixed_train_rows,
            "val_pairs": "signature_id_1,signature_id_2,label\nf3,f4,0\n",
        },
    )
    fixed_dataset["split_mode"] = "fixed_pairs"
    source_manifest = source_root / "pairwise_inputs_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": release_pairwise.TRAINING_INPUTS_MANIFEST_SCHEMA,
                "datasets": [random_dataset, fixed_dataset],
                "sealed_test_manifests": {
                    "pairwise": {"path": str(pair_manifest), "sha256": _sha256(pair_manifest)},
                    "cluster": {"path": str(cluster_manifest), "sha256": _sha256(cluster_manifest)},
                },
            }
        ),
        encoding="utf-8",
    )
    return source_manifest, _sha256(source_manifest), tmp_path / "training_plan.json"


def test_training_input_preflight_writes_test_path_free_plan(tmp_path: Path) -> None:
    manifest, manifest_sha256, output_plan = _write_training_input_fixture(tmp_path)

    result = release_pairwise.preflight_training_inputs(
        argparse.Namespace(
            manifest=manifest,
            expected_manifest_sha256=manifest_sha256,
            output_plan=output_plan,
        )
    )

    persisted = json.loads(output_plan.read_text(encoding="utf-8"))
    assert result["plan_sha256"] == _sha256(output_plan)
    assert persisted["source_manifest_sha256"] == manifest_sha256
    assert [dataset["name"] for dataset in persisted["datasets"]] == ["random", "fixed"]
    assert '"path"' not in json.dumps(persisted["sealed_test_manifests"])
    assert "test_pairs" not in json.dumps(persisted["datasets"])
    assert persisted["sealed_test_manifests"]["pairwise"]["members"]["fixed"]["pairs"]
    assert persisted["sealed_test_manifests"]["cluster"]["members"]["random"]["blocks"]


@pytest.mark.parametrize(
    ("train_rows", "message"),
    [
        ("f1,f2,MAYBE\n", "label must be 0, 1, NO, or YES"),
        ("f1,f2,YES\nf2,f1,NO\n", "duplicate unordered pair"),
        ("f1,f2, YES\n", "label must not have surrounding whitespace"),
        (" f1,f2,YES\n", "signature id must not have surrounding whitespace"),
        ("f1,f2,YES,extra\n", "must contain exactly"),
    ],
)
def test_training_input_preflight_rejects_invalid_fixed_pairs(
    tmp_path: Path,
    train_rows: str,
    message: str,
) -> None:
    manifest, manifest_sha256, output_plan = _write_training_input_fixture(
        tmp_path,
        fixed_train_rows=train_rows,
    )

    with pytest.raises(ValueError, match=message):
        release_pairwise.preflight_training_inputs(
            argparse.Namespace(
                manifest=manifest,
                expected_manifest_sha256=manifest_sha256,
                output_plan=output_plan,
            )
        )
    assert not output_plan.exists()


@pytest.mark.parametrize(
    ("validator", "payload"),
    [
        (release_pairwise._signature_ids, {" s1": {}}),  # noqa: SLF001
        (release_pairwise._cluster_signature_ids, {"c": {"signature_ids": ["s1 "]}}),  # noqa: SLF001
        (release_pairwise._block_signature_ids, {"b": [" s1"]}),  # noqa: SLF001
    ],
)
def test_training_identity_validators_reject_surrounding_whitespace(
    tmp_path: Path,
    validator: Any,
    payload: object,
) -> None:
    path = tmp_path / "identities.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="signature id must not have surrounding whitespace"):
        validator(path, context="fixture")


@pytest.mark.parametrize(
    ("fixed_test_pair", "random_test_pair", "message"),
    [
        (("f2", "f1"), ("rt1", "rt2"), "train/test unordered pairs overlap"),
        (("f5", "f6"), ("r1", "rt2"), "contains sealed test signatures"),
    ],
)
def test_training_input_preflight_rejects_test_overlap(
    tmp_path: Path,
    fixed_test_pair: tuple[str, str],
    random_test_pair: tuple[str, str],
    message: str,
) -> None:
    manifest, manifest_sha256, output_plan = _write_training_input_fixture(
        tmp_path,
        fixed_test_pair=fixed_test_pair,
        random_test_pair=random_test_pair,
    )

    with pytest.raises(ValueError, match=message):
        release_pairwise.preflight_training_inputs(
            argparse.Namespace(
                manifest=manifest,
                expected_manifest_sha256=manifest_sha256,
                output_plan=output_plan,
            )
        )
    assert not output_plan.exists()


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


class _StubCalibrationClusterer:
    """Records the calibration call sequence so nesting and resources are provable."""

    def __init__(self) -> None:
        self.cluster_model = argparse.Namespace(eps=0.42)
        self.n_jobs = 1
        # A noncanonical bundle could still carry a finite cap; calibration must
        # not read it, so make any use of it fail loudly.
        self.val_blocks_size = 1
        self.events: list[tuple[Any, ...]] = []
        self.distance_ram: list[int | None] = []
        self.predict_ram: list[int | None] = []

    @staticmethod
    def filter_blocks(block_dict: dict[str, list[str]], num_to_keep: int | None = None) -> dict[str, list[str]]:
        assert num_to_keep is None, "calibration must not cap validation blocks"
        return {key: values for key, values in block_dict.items() if len(values) > 1}

    def make_distance_matrices(
        self,
        block_dict: dict[str, list[str]],
        dataset: Any,
        total_ram_bytes: int | None = None,
    ) -> dict[str, object]:
        self.events.append(("build", dataset.name))
        self.distance_ram.append(total_ram_bytes)
        return {key: object() for key in block_dict}

    def predict(
        self,
        block_dict: dict[str, list[str]],
        dataset: Any,
        dists: dict[str, object] | None = None,
        total_ram_bytes: int | None = None,
    ) -> tuple[dict[str, list[str]], None]:
        assert dists is not None, "calibration must reuse prebuilt distance matrices"
        self.events.append(("predict", dataset.name, float(self.cluster_model.eps)))
        self.predict_ram.append(total_ram_bytes)
        return {"cluster": ["s1", "s2"]}, None


def _calibration_dataset_inputs(tmp_path: Path, dataset_names: tuple[str, ...]) -> dict[str, Any]:
    """Write clustered validation inputs and return their frozen declarations."""

    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    dataset_inputs: dict[str, Any] = {}
    for name in dataset_names:
        files: dict[str, Any] = {}
        for role in ("signatures", "papers", "specter_embeddings", "clusters"):
            path = inputs_dir / f"{name}_{role}.json"
            path.write_text(json.dumps({"dataset": name, "role": role}), encoding="utf-8")
            files[role] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        dataset_inputs[name] = {"split_mode": "random_blocks", "files": files}
    dataset_inputs["augmented"] = {"split_mode": "fixed_pairs", "files": {}}
    return dataset_inputs


def _write_calibration_bundle(tmp_path: Path, dataset_names: tuple[str, ...]) -> Path:
    """Write a minimal source used by isolated calibration-flow tests."""

    bundle = tmp_path / "calibration_bundle"
    (bundle / "reproducibility").mkdir(parents=True)
    (bundle / "reproducibility" / "pairwise_training_config.json").write_text(
        json.dumps(
            {
                "training_scope": "production_full",
                "data_random_seed": 1111,
                "dataset_inputs": _calibration_dataset_inputs(tmp_path, dataset_names),
            }
        ),
        encoding="utf-8",
    )
    (bundle / "manifest.json").write_text("{}\n", encoding="utf-8")
    return bundle


def _write_real_calibration_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_names: tuple[str, ...],
) -> Path:
    """Add manifest-bound calibration provenance to a reloadable source bundle."""

    bundle = _bundle(tmp_path, monkeypatch)
    reproducibility = bundle / "reproducibility"
    reproducibility.mkdir()
    (reproducibility / "pairwise_training_config.json").write_text(
        json.dumps(
            {
                "training_scope": "production_full",
                "data_random_seed": 1111,
                "dataset_inputs": _calibration_dataset_inputs(tmp_path, dataset_names),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (reproducibility / "pairwise_training_summary.json").write_text("{}\n", encoding="utf-8")
    write_production_manifest(
        bundle,
        bundle_version="9.9",
        pairwise_model_version="9.9",
    )
    return bundle


def _write_calibration_spec(
    tmp_path: Path,
    bundle: Path,
    *,
    eps_grid: list[float],
    minimum_dataset_f1: float = 0.0,
    minimum_signature_weighted_f1: float = 0.0,
) -> tuple[Path, str]:
    """Write the fixed B12 calibration spec and return its digest."""

    spec = tmp_path / "eps_calibration_spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": release_pairwise.EPS_CALIBRATION_SPEC_SCHEMA,
                "source_manifest_sha256": _sha256(bundle / "manifest.json"),
                "eps_grid": eps_grid,
                "objective": "signature_weighted_b3_f1",
                "floors": {
                    "minimum_dataset_f1": minimum_dataset_f1,
                    "minimum_signature_weighted_f1": minimum_signature_weighted_f1,
                },
                "aggregation": "dataset_macro_and_signature_weighted",
                "tie_break": "smallest_eps",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return spec, _sha256(spec)


def _calibration_args(
    tmp_path: Path,
    bundle: Path,
    spec: Path,
    spec_sha256: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        source_bundle=bundle,
        spec=spec,
        expected_spec_sha256=spec_sha256,
        output_bundle=tmp_path / "calibrated_bundle",
        output_report=tmp_path / "calibration_report.json",
        n_jobs=1,
        total_ram_bytes=None,
    )


def _patch_calibration(
    monkeypatch: pytest.MonkeyPatch,
    clusterer: _StubCalibrationClusterer,
    *,
    f1_by_eps: dict[float, float | dict[str, float]],
) -> tuple[list[str], list[dict[str, Any]]]:
    constructed: list[str] = []
    finalizations: list[dict[str, Any]] = []

    def fake_anddata(name: str, files: Any, *, mode: str, random_seed: int = 1111) -> Any:
        constructed.append(name)
        return argparse.Namespace(
            name=name,
            split_cluster_signatures=lambda: ({}, {"b1": ["s1", "s2"], "b2": ["s3"]}, {}),
            construct_cluster_to_signatures=lambda blocks: {"c1": ["s1", "s2"]},
        )

    monkeypatch.setattr(release_pairwise, "_load_pairwise_staging_model", lambda path: clusterer)
    monkeypatch.setattr(release_pairwise, "_anddata", fake_anddata)

    def fake_b3(true_clusters: object, predicted: object) -> dict[str, Any]:
        score = f1_by_eps[float(clusterer.cluster_model.eps)]
        f1 = score[clusterer.events[-1][1]] if isinstance(score, dict) else score
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": f1,
            "signature_count": 2,
        }

    monkeypatch.setattr(release_pairwise, "_b3_report", fake_b3)

    def fake_finalize(**kwargs: Any) -> None:
        finalizations.append(kwargs)
        shutil.copytree(kwargs["source_bundle_dir"], kwargs["output_bundle_dir"])

    monkeypatch.setattr(release_pairwise, "finalize_pairwise_eps", fake_finalize)
    return constructed, finalizations


def test_calibrate_eps_builds_each_matrix_once_and_holds_one_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _write_calibration_bundle(tmp_path, ("qian", "pubmed"))
    spec, spec_sha256 = _write_calibration_spec(tmp_path, bundle, eps_grid=[0.6, 0.3])
    clusterer = _StubCalibrationClusterer()
    _, finalizations = _patch_calibration(monkeypatch, clusterer, f1_by_eps={0.3: 0.5, 0.6: 0.9})
    args = _calibration_args(tmp_path, bundle, spec, spec_sha256)
    args.n_jobs = 5
    args.total_ram_bytes = 2048

    report = release_pairwise.calibrate_eps(args)

    assert [event for event in clusterer.events if event[0] == "build"] == [("build", "pubmed"), ("build", "qian")]
    assert clusterer.events == [
        ("build", "pubmed"),
        ("predict", "pubmed", 0.3),
        ("predict", "pubmed", 0.6),
        ("build", "qian"),
        ("predict", "qian", 0.3),
        ("predict", "qian", 0.6),
    ]
    assert clusterer.n_jobs == 5
    assert clusterer.distance_ram == [2048, 2048]
    assert set(clusterer.predict_ram) == {2048}
    assert clusterer.cluster_model.eps == pytest.approx(0.42)
    assert report["selected_eps"] == pytest.approx(0.6)
    assert [trial["eps"] for trial in report["trials"]] == [0.3, 0.6]
    assert set(report["validation_identities"]) == {"qian", "pubmed"}
    assert all("blocks" not in identity for identity in report["validation_identities"].values())
    assert finalizations[0]["new_eps"] == pytest.approx(0.6)
    assert args.output_bundle.is_dir()
    assert args.output_report.is_file()


def test_calibrate_eps_tie_breaks_to_smallest_eps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _write_calibration_bundle(tmp_path, ("qian",))
    spec, spec_sha256 = _write_calibration_spec(tmp_path, bundle, eps_grid=[0.7, 0.2])
    clusterer = _StubCalibrationClusterer()
    _patch_calibration(monkeypatch, clusterer, f1_by_eps={0.2: 0.8, 0.7: 0.8})

    report = release_pairwise.calibrate_eps(_calibration_args(tmp_path, bundle, spec, spec_sha256))

    assert report["selected_eps"] == pytest.approx(0.2)


def test_calibrate_eps_applies_floors_before_objective_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _write_calibration_bundle(tmp_path, ("qian", "pubmed"))
    spec, spec_sha256 = _write_calibration_spec(
        tmp_path,
        bundle,
        eps_grid=[0.3, 0.6],
        minimum_dataset_f1=0.65,
        minimum_signature_weighted_f1=0.65,
    )
    clusterer = _StubCalibrationClusterer()
    _patch_calibration(
        monkeypatch,
        clusterer,
        f1_by_eps={
            0.3: {"qian": 1.0, "pubmed": 0.6},
            0.6: {"qian": 0.7, "pubmed": 0.7},
        },
    )

    report = release_pairwise.calibrate_eps(_calibration_args(tmp_path, bundle, spec, spec_sha256))

    assert report["selected_eps"] == pytest.approx(0.6)
    assert report["trials"][0]["signature_weighted"]["f1"] == pytest.approx(0.8)


def test_calibrate_eps_rejects_when_no_trial_meets_weighted_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _write_calibration_bundle(tmp_path, ("qian",))
    spec, spec_sha256 = _write_calibration_spec(
        tmp_path,
        bundle,
        eps_grid=[0.5],
        minimum_signature_weighted_f1=0.9,
    )
    clusterer = _StubCalibrationClusterer()
    _, finalizations = _patch_calibration(monkeypatch, clusterer, f1_by_eps={0.5: 0.8})
    args = _calibration_args(tmp_path, bundle, spec, spec_sha256)

    with pytest.raises(RuntimeError, match="No EPS calibration trial met"):
        release_pairwise.calibrate_eps(args)

    assert finalizations == []
    assert not args.output_bundle.exists()
    assert not args.output_report.exists()


def test_calibrate_eps_rejects_unknown_split_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _write_calibration_bundle(tmp_path, ("qian",))
    config_path = bundle / "reproducibility" / "pairwise_training_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["dataset_inputs"]["qian"]["split_mode"] = "typo"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    spec, spec_sha256 = _write_calibration_spec(tmp_path, bundle, eps_grid=[0.5])
    clusterer = _StubCalibrationClusterer()
    _patch_calibration(monkeypatch, clusterer, f1_by_eps={0.5: 0.8})

    with pytest.raises(ValueError, match="unknown split_mode"):
        release_pairwise.calibrate_eps(_calibration_args(tmp_path, bundle, spec, spec_sha256))


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("spec_digest", "Calibration-spec SHA-256 mismatch"),
        ("source_digest", "Calibration source manifest SHA-256 mismatch"),
        ("input_digest", "Calibration input drift for qian:papers"),
        ("existing_bundle", "Calibration output bundle already exists"),
        ("existing_report", "Calibration output report already exists"),
    ],
)
def test_calibrate_eps_rejects_wrong_input_before_matrix_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    message: str,
) -> None:
    bundle = _write_calibration_bundle(tmp_path, ("qian",))
    spec, spec_sha256 = _write_calibration_spec(tmp_path, bundle, eps_grid=[0.5])
    clusterer = _StubCalibrationClusterer()
    constructed, finalizations = _patch_calibration(monkeypatch, clusterer, f1_by_eps={0.5: 0.5})
    args = _calibration_args(tmp_path, bundle, spec, spec_sha256)
    if failure == "spec_digest":
        args.expected_spec_sha256 = "0" * 64
    elif failure == "source_digest":
        (bundle / "manifest.json").write_text('{"drift": true}\n', encoding="utf-8")
    elif failure == "input_digest":
        (tmp_path / "inputs" / "qian_papers.json").write_text("tampered", encoding="utf-8")
    elif failure == "existing_report":
        args.output_report.write_text("{}\n", encoding="utf-8")
    else:
        args.output_bundle.mkdir()

    with pytest.raises((FileExistsError, ValueError), match=message):
        release_pairwise.calibrate_eps(args)

    assert constructed == []
    assert clusterer.events == []
    assert finalizations == []
    assert failure == "existing_bundle" or not args.output_bundle.exists()


@pytest.mark.parametrize(
    ("eps_grid", "message"),
    [
        ([1.5], "Calibration EPS must be finite"),
        ([float("nan")], "Calibration EPS must be finite"),
        ([0.5, 0.5], "eps_grid values must be unique"),
    ],
)
def test_calibrate_eps_rejects_invalid_frozen_grid_before_matrix_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    eps_grid: list[float],
    message: str,
) -> None:
    bundle = _write_calibration_bundle(tmp_path, ("qian",))
    spec, spec_sha256 = _write_calibration_spec(tmp_path, bundle, eps_grid=eps_grid)
    clusterer = _StubCalibrationClusterer()
    constructed, finalizations = _patch_calibration(monkeypatch, clusterer, f1_by_eps={})

    with pytest.raises(ValueError, match=message):
        release_pairwise.calibrate_eps(_calibration_args(tmp_path, bundle, spec, spec_sha256))

    assert constructed == []
    assert clusterer.events == []
    assert finalizations == []


def test_calibrate_eps_rejects_expanded_spec_schema_before_matrix_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _write_calibration_bundle(tmp_path, ("qian",))
    spec, _ = _write_calibration_spec(tmp_path, bundle, eps_grid=[0.5])
    payload = json.loads(spec.read_text(encoding="utf-8"))
    payload["compatibility_mode"] = True
    spec.write_text(json.dumps(payload), encoding="utf-8")
    clusterer = _StubCalibrationClusterer()
    constructed, finalizations = _patch_calibration(monkeypatch, clusterer, f1_by_eps={})

    with pytest.raises(ValueError, match="must contain exactly"):
        release_pairwise.calibrate_eps(_calibration_args(tmp_path, bundle, spec, _sha256(spec)))

    assert constructed == []
    assert clusterer.events == []
    assert finalizations == []


def test_calibrate_eps_parser_exposes_final_spec_driven_interface() -> None:
    parser = release_pairwise.build_parser()
    args = release_pairwise.build_parser().parse_args(
        [
            "calibrate-eps",
            "--source-bundle",
            "source",
            "--spec",
            "eps_calibration_spec.json",
            "--expected-spec-sha256",
            "a" * 64,
            "--output-bundle",
            "calibrated",
            "--output-report",
            "calibration_report.json",
            "--n-jobs",
            "8",
            "--total-ram-bytes",
            "4096",
        ]
    )

    assert args.n_jobs == 8
    assert args.total_ram_bytes == 4096
    assert "eps" not in vars(args)
    with pytest.raises(SystemExit):
        parser.parse_args(["finalize-eps"])


def test_module_entrypoint_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.production.model.release_pairwise", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "preflight-training-inputs" in completed.stdout
    assert "calibrate-eps" in completed.stdout
    assert "evaluate-clusters" in completed.stdout
    assert "finalize-eps" not in completed.stdout

    calibration_help = subprocess.run(
        [sys.executable, "-m", "scripts.production.model.release_pairwise", "calibrate-eps", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert calibration_help.returncode == 0, calibration_help.stderr
    for option in (
        "--source-bundle",
        "--spec",
        "--expected-spec-sha256",
        "--output-bundle",
        "--output-report",
    ):
        assert option in calibration_help.stdout
    assert "--eps" not in calibration_help.stdout


@pytest.mark.parametrize("selected_eps", [0.65, 0.5], ids=["changed", "unchanged"])
def test_calibrate_eps_always_finalizes_preserves_bytes_and_reloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_eps: float,
) -> None:
    source = _write_real_calibration_bundle(tmp_path, monkeypatch, ("qian",))
    spec, spec_sha256 = _write_calibration_spec(tmp_path, source, eps_grid=[selected_eps])
    source_bytes = {
        path.relative_to(source).as_posix(): path.read_bytes() for path in source.rglob("*") if path.is_file()
    }
    clusterer = _StubCalibrationClusterer()
    clusterer.cluster_model.eps = 0.5
    _patch_calibration(monkeypatch, clusterer, f1_by_eps={selected_eps: 0.8})
    finalize_calls: list[dict[str, Any]] = []

    def finalize_spy(**kwargs: Any) -> Any:
        finalize_calls.append(kwargs)
        return finalize_pairwise_eps(**kwargs)

    monkeypatch.setattr(release_pairwise, "finalize_pairwise_eps", finalize_spy)
    args = _calibration_args(tmp_path, source, spec, spec_sha256)

    report = release_pairwise.calibrate_eps(args)

    assert {
        path.relative_to(source).as_posix(): path.read_bytes() for path in source.rglob("*") if path.is_file()
    } == source_bytes
    assert len(finalize_calls) == 1
    assert finalize_calls[0]["new_eps"] == pytest.approx(selected_eps)
    assert finalize_calls[0]["expected_old_eps"] == pytest.approx(0.5)
    output = args.output_bundle
    output_bytes = {
        path.relative_to(output).as_posix(): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }
    assert set(output_bytes) == set(source_bytes)
    for relpath in set(source_bytes) - {"clusterer.json", "manifest.json"}:
        assert output_bytes[relpath] == source_bytes[relpath]
    assert json.loads(output_bytes["clusterer.json"])["cluster_model"]["eps"] == selected_eps
    assert _load_pairwise_staging_model(output).cluster_model.eps == selected_eps
    assert report["source_manifest_sha256"] == _sha256(source / "manifest.json")
    assert report["output_manifest_sha256"] == _sha256(output / "manifest.json")
    assert report["selected_eps"] == selected_eps
    assert "unchanged_members" not in report


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


@pytest.mark.parametrize("invalid_probability", [float("nan"), float("inf"), -0.1, 1.1])
def test_pairwise_metrics_rejects_nonfinite_or_unbounded_raw_probabilities(
    invalid_probability: float,
) -> None:
    with pytest.raises(ValueError, match="probabilities must all"):
        release_pairwise.pairwise_metrics(
            np.asarray([0, 1]),
            np.asarray([invalid_probability, 0.8]),
            np.asarray([0.2, 0.8]),
        )


@pytest.mark.parametrize("invalid_metric", [float("nan"), float("inf"), -0.1, 1.1])
def test_b3_report_rejects_nonfinite_or_unbounded_metrics(
    monkeypatch: pytest.MonkeyPatch,
    invalid_metric: float,
) -> None:
    monkeypatch.setattr(
        release_pairwise,
        "b3_precision_recall_fscore",
        lambda true, predicted: (invalid_metric, 0.5, 0.5, {"s1": object()}, None, None),
    )

    with pytest.raises(RuntimeError, match=r"finite and in \[0, 1\]"):
        release_pairwise._b3_report({"c1": ["s1"]}, {"c1": ["s1"]})  # noqa: SLF001


def test_fresh_json_link_failure_allows_exact_payload_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "report.json"
    real_link = release_pairwise.os.link
    attempts = 0

    def fail_link(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("simulated publication interruption")
        real_link(source, destination)

    monkeypatch.setattr(release_pairwise.os, "link", fail_link)
    with pytest.raises(OSError, match="simulated publication interruption"):
        release_pairwise._write_fresh_json(output, {"ok": True})  # noqa: SLF001

    assert not output.exists()
    assert list(tmp_path.glob(".report.json.*.tmp")) == []
    release_pairwise._write_fresh_json(output, {"ok": True})  # noqa: SLF001
    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}


def test_cluster_evaluator_has_no_unused_ram_option() -> None:
    parser = release_pairwise.build_parser()
    common = [
        "evaluate-clusters",
        "--model",
        "model",
        "--expected-model-manifest-sha256",
        "b" * 64,
        "--quality-policy",
        "quality-policy.json",
        "--expected-quality-policy-sha256",
        "c" * 64,
        "--manifest",
        "manifest.json",
        "--expected-manifest-sha256",
        "a" * 64,
        "--output-report",
        "report.json",
    ]
    args = parser.parse_args(common)
    assert args.command == "evaluate-clusters"
    assert "unblind_record" not in vars(args)
    with pytest.raises(SystemExit):
        parser.parse_args([*common, "--total-ram-bytes", "1024"])


def test_existing_output_fails_before_binding_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "report.json"
    output.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        release_pairwise,
        "_validated_evaluation_bindings",
        lambda *args, **kwargs: pytest.fail("bindings validated after output conflict"),
    )

    with pytest.raises(FileExistsError, match="output already exists"):
        release_pairwise.evaluate_pairs(argparse.Namespace(output_report=output))


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


@pytest.mark.parametrize(
    ("dataset_names", "message"),
    [
        ([""], "names must be nonempty strings"),
        (["repeated", "repeated"], "duplicate dataset name 'repeated'"),
    ],
)
@pytest.mark.parametrize(
    "schema_version",
    [release_pairwise.PAIR_MANIFEST_SCHEMA, release_pairwise.CLUSTER_MANIFEST_SCHEMA],
)
def test_manifest_requires_unique_nonempty_dataset_names(
    tmp_path: Path,
    dataset_names: list[str],
    message: str,
    schema_version: str,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "datasets": [{"name": name} for name in dataset_names],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        release_pairwise._verified_manifest(  # noqa: SLF001
            manifest,
            hashlib.sha256(manifest.read_bytes()).hexdigest(),
            schema_version,
        )


@pytest.mark.parametrize("invalid_label", [0.9, -0.1, "1", True])
def test_pair_evaluator_rejects_noninteger_json_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_label: object,
) -> None:
    manifest, manifest_sha256 = _write_evaluation_manifest(
        tmp_path,
        schema_version=release_pairwise.PAIR_MANIFEST_SCHEMA,
        contents={
            "signatures": {},
            "papers": {},
            "specter_embeddings": {},
            "pairs": [
                {
                    "signature_id_1": "s1",
                    "signature_id_2": "s2",
                    "label": invalid_label,
                }
            ],
        },
    )
    bindings = _write_evaluation_bindings(tmp_path)
    monkeypatch.setattr(
        release_pairwise,
        "load_production_model",
        lambda _: pytest.fail("model loaded before pair labels were validated"),
    )
    monkeypatch.setattr(
        release_pairwise,
        "_anddata",
        lambda *args, **kwargs: pytest.fail("dataset constructed before pair labels were validated"),
    )

    with pytest.raises(ValueError, match="JSON integer 0 or 1"):
        release_pairwise.evaluate_pairs(
            argparse.Namespace(
                output_report=tmp_path / "report.json",
                manifest=manifest,
                expected_manifest_sha256=manifest_sha256,
                n_jobs=1,
                total_ram_bytes=None,
                **bindings,
            )
        )


def test_manifest_input_drift_fails_before_model_load(
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
        "load_production_model",
        lambda _: pytest.fail("model loaded before input verification"),
    )
    bindings = _write_evaluation_bindings(tmp_path)
    report = tmp_path / "report.json"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        release_pairwise.evaluate_pairs(
            argparse.Namespace(
                output_report=report,
                manifest=manifest,
                expected_manifest_sha256=_sha256(manifest),
                **bindings,
            )
        )
    assert not report.exists()


class _StubBinaryClassifier:
    def __init__(self, positive_probabilities: list[float]) -> None:
        self.positive_probabilities = np.asarray(positive_probabilities, dtype=np.float64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        assert features.shape[0] == self.positive_probabilities.size
        return np.column_stack((1.0 - self.positive_probabilities, self.positive_probabilities))


def test_pair_evaluator_loads_complete_bundle_and_publishes_digest_only_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, manifest_sha256 = _write_evaluation_manifest(
        tmp_path,
        schema_version=release_pairwise.PAIR_MANIFEST_SCHEMA,
        contents={
            "signatures": {},
            "papers": {},
            "specter_embeddings": {},
            "pairs": [
                {"signature_id_1": "s1", "signature_id_2": "s2", "label": 0},
                {"signature_id_1": "s3", "signature_id_2": "s4", "label": 1},
            ],
        },
    )
    bindings = _write_evaluation_bindings(tmp_path)
    complete_loader_calls: list[Path] = []
    clusterer = argparse.Namespace(
        featurizer_info=object(),
        nameless_featurizer_info=object(),
        classifier=_StubBinaryClassifier([0.1, 0.9]),
        nameless_classifier=_StubBinaryClassifier([0.3, 0.7]),
    )

    def load_complete(path: Path) -> object:
        complete_loader_calls.append(path)
        return clusterer

    monkeypatch.setattr(release_pairwise, "load_production_model", load_complete)
    monkeypatch.setattr(
        release_pairwise,
        "_load_pairwise_staging_model",
        lambda _: pytest.fail("pair evaluator used the staging-only loader"),
    )
    monkeypatch.setattr(release_pairwise, "_anddata", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        release_pairwise,
        "many_pairs_featurize",
        lambda *args, **kwargs: (
            np.zeros((2, 1)),
            np.asarray([0, 1]),
            np.zeros((2, 1)),
        ),
    )
    output = tmp_path / "pair_report.json"

    report = release_pairwise.evaluate_pairs(
        argparse.Namespace(
            output_report=output,
            manifest=manifest,
            expected_manifest_sha256=manifest_sha256,
            n_jobs=1,
            total_ram_bytes=None,
            **bindings,
        )
    )

    assert complete_loader_calls == [Path(bindings["model"]).resolve()]
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["quality_policy_sha256"] == bindings["expected_quality_policy_sha256"]
    assert report["model_manifest_sha256"] == bindings["expected_model_manifest_sha256"]
    assert report["population_manifest_sha256"] == manifest_sha256
    assert set(report["datasets"]["toy"]) == {"metrics"}
    assert "pairs" not in report["datasets"]["toy"]
    assert not list(tmp_path.glob("*unblind*"))
    assert not list(tmp_path.glob("*evaluation_start*"))
    assert not list(tmp_path.glob("*promotion*"))


def test_cluster_evaluator_loads_complete_bundle_and_publishes_digest_only_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, manifest_sha256 = _write_evaluation_manifest(
        tmp_path,
        schema_version=release_pairwise.CLUSTER_MANIFEST_SCHEMA,
        contents={
            "signatures": {},
            "papers": {},
            "specter_embeddings": {},
            "clusters": {},
            "blocks": {"b1": ["s1", "s2"]},
        },
    )
    bindings = _write_evaluation_bindings(tmp_path)

    class StubClusterer:
        n_jobs = 1

        @staticmethod
        def predict(blocks: dict[str, list[str]], dataset: object) -> tuple[dict[str, list[str]], None]:
            return {"c1": ["s1"], "c2": ["s2"]}, None

    dataset = argparse.Namespace(construct_cluster_to_signatures=lambda blocks: {"c1": ["s1"], "c2": ["s2"]})
    monkeypatch.setattr(release_pairwise, "load_production_model", lambda path: StubClusterer())
    monkeypatch.setattr(
        release_pairwise,
        "_load_pairwise_staging_model",
        lambda _: pytest.fail("cluster evaluator used the staging-only loader"),
    )
    monkeypatch.setattr(release_pairwise, "_anddata", lambda *args, **kwargs: dataset)
    output = tmp_path / "cluster_report.json"

    report = release_pairwise.evaluate_clusters(
        argparse.Namespace(
            output_report=output,
            manifest=manifest,
            expected_manifest_sha256=manifest_sha256,
            n_jobs=2,
            **bindings,
        )
    )

    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["quality_policy_sha256"] == bindings["expected_quality_policy_sha256"]
    assert report["model_manifest_sha256"] == bindings["expected_model_manifest_sha256"]
    assert report["population_manifest_sha256"] == manifest_sha256
    assert set(report["datasets"]["toy"]) == {"metrics"}
    assert "predicted_clusters" not in report["datasets"]["toy"]


def test_cluster_evaluator_rejects_empty_block_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, manifest_sha256 = _write_evaluation_manifest(
        tmp_path,
        schema_version=release_pairwise.CLUSTER_MANIFEST_SCHEMA,
        contents={
            "signatures": {},
            "papers": {},
            "specter_embeddings": {},
            "clusters": {},
            "blocks": {"empty": []},
        },
    )
    bindings = _write_evaluation_bindings(tmp_path)
    monkeypatch.setattr(
        release_pairwise,
        "load_production_model",
        lambda _: pytest.fail("model loaded before empty-block validation"),
    )

    with pytest.raises(ValueError, match="nonempty signature lists"):
        release_pairwise.evaluate_clusters(
            argparse.Namespace(
                output_report=tmp_path / "cluster_report.json",
                manifest=manifest,
                expected_manifest_sha256=manifest_sha256,
                n_jobs=1,
                **bindings,
            )
        )


@pytest.mark.parametrize(
    ("binding", "message"),
    [
        ("expected_quality_policy_sha256", "Quality-policy SHA-256 mismatch"),
        ("expected_model_manifest_sha256", "Model-manifest SHA-256 mismatch"),
        ("expected_manifest_sha256", "Manifest SHA-256 mismatch"),
    ],
)
def test_wrong_binding_fails_before_model_load_or_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding: str,
    message: str,
) -> None:
    manifest, manifest_sha256 = _write_evaluation_manifest(
        tmp_path,
        schema_version=release_pairwise.PAIR_MANIFEST_SCHEMA,
        contents={
            "signatures": {},
            "papers": {},
            "specter_embeddings": {},
            "pairs": [
                {"signature_id_1": "s1", "signature_id_2": "s2", "label": 0},
                {"signature_id_1": "s3", "signature_id_2": "s4", "label": 1},
            ],
        },
    )
    arguments = {
        "output_report": tmp_path / "report.json",
        "manifest": manifest,
        "expected_manifest_sha256": manifest_sha256,
        "n_jobs": 1,
        "total_ram_bytes": None,
        **_write_evaluation_bindings(tmp_path),
    }
    arguments[binding] = "0" * 64
    monkeypatch.setattr(
        release_pairwise,
        "load_production_model",
        lambda _: pytest.fail("model loaded before all bindings were validated"),
    )
    monkeypatch.setattr(
        release_pairwise,
        "many_pairs_featurize",
        lambda *args, **kwargs: pytest.fail("scoring began before all bindings were validated"),
    )

    with pytest.raises(ValueError, match=message):
        release_pairwise.evaluate_pairs(argparse.Namespace(**arguments))

    assert not arguments["output_report"].exists()


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
