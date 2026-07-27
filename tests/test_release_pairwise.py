from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from s2and.release_evidence import (
    RELEASE_EVIDENCE_MANIFEST_SCHEMA,
    RELEASE_EVIDENCE_ROLES,
    validate_release_evidence_manifest,
)
from scripts.production.model import release_pairwise

ARTIFACT_HASHES = {
    "name_counts_manifest_sha256": "c" * 64,
    "name_tuples_data_sha256": "d" * 64,
    "orcid_prefix_counts_data_sha256": "e" * 64,
    "orcid_prefix_counts_manifest_sha256": "f" * 64,
}
REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _release_spec_payload() -> dict[str, Any]:
    return {
        "schema_version": release_pairwise.RELEASE_SPEC_SCHEMA,
        "release_identity": {
            "normalization_version": "canonical_v2",
            "python_version": "1.3.0",
            "release_commit": "a" * 40,
            "rust_version": "1.3.0",
        },
        "populations": {
            "cluster_datasets": ["toy"],
            "cluster_manifest_sha256": "2" * 64,
            "pairwise_datasets": ["toy"],
            "pairwise_manifest_sha256": "1" * 64,
        },
        "inputs": {
            "data_manifest_sha256": "7" * 64,
            "parity_fixture_manifest_sha256": "3" * 64,
            "performance_data_manifest_sha256": "4" * 64,
            "performance_workload_sha256": "5" * 64,
            "subblocking_input_manifest_sha256": "6" * 64,
        },
        "baselines": {
            "cluster_signature_weighted_b3_f1": 0.804,
            "pairwise_aggregate": {"auroc": 0.9005, "macro_f1": 0.804},
            "pairwise_datasets": {"toy": {"auroc": 0.9005, "macro_f1": 0.804}},
            "predict_seconds_p50": 10.0,
        },
        "thresholds": {
            "cluster_signature_weighted_b3_f1_max_drop": 0.005,
            "pairwise_aggregate_auroc_max_drop": 0.001,
            "pairwise_aggregate_macro_f1_max_drop": 0.005,
            "pairwise_dataset_auroc_max_drop": 0.001,
            "pairwise_dataset_macro_f1_max_drop": 0.005,
            "peak_rss_absolute_max_gb": 4.0,
            "runtime_max_ratio": 1.1,
            "subblocking_maximum_size": 100,
        },
    }


def _linker_metrics() -> dict[str, Any]:
    integer_metrics = {
        "stratified_test_errors",
        "stratified_test_false_abstain",
        "stratified_test_false_link",
        "stratified_test_queries",
        "stratified_test_wrong_candidate_link",
        "training_positive_rows",
        "training_rows",
    }
    return {
        key: (
            {
                "false_abstain_error_rate": 1.0,
                "false_link_error_rate": 2.0,
                "wrong_link_error_rate": 3.0,
            }
            if key == "weighted_average_error_weights"
            else 1
            if key in integer_metrics
            else 0.1
        )
        for key in release_pairwise.SUPPORTED_OFFICIAL_METRIC_KEYS
    }


def _write_evaluation_bindings(
    tmp_path: Path,
    population_manifest_sha256: str,
    *,
    manifest_schema: str = release_pairwise.PAIR_MANIFEST_SCHEMA,
) -> dict[str, object]:
    release_spec_payload = _release_spec_payload()
    digest_field = {
        release_pairwise.PAIR_MANIFEST_SCHEMA: "pairwise_manifest_sha256",
        release_pairwise.CLUSTER_MANIFEST_SCHEMA: "cluster_manifest_sha256",
    }[manifest_schema]
    release_spec_payload["populations"][digest_field] = population_manifest_sha256
    release_spec = tmp_path / "release_spec.json"
    release_spec.write_text(json.dumps(release_spec_payload) + "\n", encoding="utf-8")
    model = tmp_path / "complete_model"
    model.mkdir()
    model_manifest = model / "manifest.json"
    model_manifest.write_text("{}\n", encoding="utf-8")
    return {
        "model": model,
        "expected_model_manifest_sha256": _sha256(model_manifest),
        "release_spec": release_spec,
        "expected_release_spec_sha256": _sha256(release_spec),
    }


def _patch_evaluation_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[argparse.Namespace, argparse.Namespace, argparse.Namespace, dict[str, Any]]:
    name_counts_index = argparse.Namespace(manifest_sha256="c" * 64)
    name_tuples = argparse.Namespace(data_sha256="d" * 64, pairs=frozenset({("a", "b")}))
    orcid_prefix_counts = argparse.Namespace(
        data_sha256="e" * 64,
        manifest_sha256="f" * 64,
        name_tuples_sha256=name_tuples.data_sha256,
    )
    monkeypatch.setattr(
        release_pairwise,
        "_load_release_artifacts",
        lambda _args: release_pairwise.ProductionArtifactAuthority(
            name_counts_index=name_counts_index,
            name_tuples=name_tuples,
            orcid_prefix_counts=orcid_prefix_counts,
        ),
    )
    return (
        name_counts_index,
        name_tuples,
        orcid_prefix_counts,
        {
            "name_counts_index_root": Path("explicit-name-counts-index"),
        },
    )


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


def _write_release_evidence(
    tmp_path: Path,
    release_spec: dict[str, Any],
) -> tuple[Path, str, dict[str, Path]]:
    evidence_root = tmp_path / "release_evidence"
    evidence_root.mkdir()
    payloads: dict[str, dict[str, Any]] = {
        role: {"schema_version": schema}
        for role, schema in release_pairwise._RELEASE_EVIDENCE_SCHEMAS.items()  # noqa: SLF001
    }
    payloads["complete_model_manifest"] = {
        "bundle_version": "9.9",
        "incremental_linker_version": "9.9",
        "pairwise_model_version": "9.9",
        "schema_version": release_pairwise._RELEASE_EVIDENCE_SCHEMAS["complete_model_manifest"],  # noqa: SLF001
        "sha256": {},
    }
    payloads["data_manifest"] = {"schema": "inference_arrow_bundle_v1"}
    payloads["release_spec"] = release_spec
    complete_model_path = evidence_root / "complete_model_manifest.json"
    complete_model_path.write_text(json.dumps(payloads["complete_model_manifest"]) + "\n", encoding="utf-8")
    complete_model_sha256 = _sha256(complete_model_path)
    data_manifest_path = evidence_root / "data_manifest.json"
    data_manifest_path.write_text(json.dumps(payloads["data_manifest"]) + "\n", encoding="utf-8")
    release_spec["inputs"]["data_manifest_sha256"] = _sha256(data_manifest_path)
    release_spec_path = evidence_root / "release_spec.json"
    release_spec_path.write_text(json.dumps(release_spec) + "\n", encoding="utf-8")
    release_spec_sha256 = _sha256(release_spec_path)

    payloads["pairwise_evaluation_report"].update(
        {
            "release_spec_sha256": release_spec_sha256,
            "model_manifest_sha256": complete_model_sha256,
            **ARTIFACT_HASHES,
            "population_manifest_sha256": release_spec["populations"]["pairwise_manifest_sha256"],
            "aggregate": {"auroc": 0.9, "macro_f1": 0.8},
            "datasets": {"toy": {"metrics": {"auroc": 0.9, "macro_f1": 0.8}}},
        }
    )
    payloads["cluster_evaluation_report"].update(
        {
            "release_spec_sha256": release_spec_sha256,
            "model_manifest_sha256": complete_model_sha256,
            **ARTIFACT_HASHES,
            "population_manifest_sha256": release_spec["populations"]["cluster_manifest_sha256"],
            "signature_weighted": {"f1": 0.8},
            "datasets": {"toy": {"metrics": {"f1": 0.8}}},
        }
    )
    payloads["linker_evaluation_report"].update(
        {
            "model_manifest_sha256": complete_model_sha256,
            "observed_metrics": _linker_metrics(),
            "query_predictions": {
                "bytes": 42,
                "path": "classic/query_predictions.csv",
                "sha256": "a" * 64,
            },
        }
    )
    payloads["performance_evaluation_report"].update(
        {
            "model_manifest_sha256": complete_model_sha256,
            "data_manifest_sha256": release_spec["inputs"]["performance_data_manifest_sha256"],
            "workload_sha256": release_spec["inputs"]["performance_workload_sha256"],
            "summary": {
                "predict_seconds": {"p50": 10.5},
                "peak_rss_gb": {"max": 2.5},
            },
        }
    )
    payloads["subblocking_evaluation_report"].update(
        {
            "input_manifest_sha256": release_spec["inputs"]["subblocking_input_manifest_sha256"],
            "rust": {
                "partition": {"max_subblock_size": 50},
                "component_preservation": {"component_pair_recall": 1.0},
            },
        }
    )
    payloads["parity_evaluation_report"].update(
        {
            "model_manifest_sha256": complete_model_sha256,
            "fixture_manifest_sha256": release_spec["inputs"]["parity_fixture_manifest_sha256"],
            "clusters_exact_match": True,
        }
    )

    paths = {
        "complete_model_manifest": complete_model_path,
        "data_manifest": data_manifest_path,
        "release_spec": release_spec_path,
    }
    for role in sorted(RELEASE_EVIDENCE_ROLES - set(paths)):
        path = evidence_root / f"{role}.json"
        path.write_text(json.dumps(payloads[role]) + "\n", encoding="utf-8")
        paths[role] = path
    manifest_path = evidence_root / "evidence_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": RELEASE_EVIDENCE_MANIFEST_SCHEMA,
                "members": {
                    role: {
                        "path": path.relative_to(evidence_root).as_posix(),
                        "sha256": _sha256(path),
                        "size_bytes": path.stat().st_size,
                        "url": f"https://example.test/release/{path.name}",
                    }
                    for role, path in sorted(paths.items())
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path, _sha256(manifest_path), paths


def _refresh_evidence_digest(manifest_path: Path, role: str, path: Path) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["members"][role]["sha256"] = _sha256(path)
    manifest["members"][role]["size_bytes"] = path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _sha256(manifest_path)


def _evaluation_fixture(tmp_path: Path) -> argparse.Namespace:
    release_spec = _release_spec_payload()
    evidence_manifest, evidence_manifest_sha256, paths = _write_release_evidence(
        tmp_path,
        release_spec,
    )
    return argparse.Namespace(
        release_spec=release_spec,
        release_spec_path=paths["release_spec"],
        release_spec_sha256=_sha256(paths["release_spec"]),
        evidence_manifest=evidence_manifest,
        evidence_manifest_sha256=evidence_manifest_sha256,
        paths=paths,
    )


def _evaluation_report_args(
    fixture: argparse.Namespace,
    output_report: Path,
    *,
    evidence_manifest_sha256: str | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        evidence_manifest=fixture.evidence_manifest,
        expected_evidence_manifest_sha256=(
            fixture.evidence_manifest_sha256 if evidence_manifest_sha256 is None else evidence_manifest_sha256
        ),
        output_report=output_report,
    )


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
    fixed_test_pair: tuple[str, str] = ("f5", "f6"),
) -> tuple[Path, str, Path]:
    test_root = tmp_path / "sealed"
    pair_root = test_root / "pair"
    cluster_root = test_root / "cluster"
    pair_root.mkdir(parents=True)
    cluster_root.mkdir()
    pair_datasets = []
    for name, pair in (("random", ("rt1", "rt2")), ("fixed", fixed_test_pair)):
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
            "train_pairs": "signature_id_1,signature_id_2,label\nf1,f2,YES\n",
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


def test_training_input_preflight_rejects_test_overlap(
    tmp_path: Path,
) -> None:
    manifest, manifest_sha256, output_plan = _write_training_input_fixture(
        tmp_path,
        fixed_test_pair=("f2", "f1"),
    )

    with pytest.raises(ValueError, match="train/test unordered pairs overlap"):
        release_pairwise.preflight_training_inputs(
            argparse.Namespace(
                manifest=manifest,
                expected_manifest_sha256=manifest_sha256,
                output_plan=output_plan,
            )
        )
    assert not output_plan.exists()


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
        name_counts_index_root=tmp_path / "name-counts-index",
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

    name_counts_index = argparse.Namespace(manifest_sha256="c" * 64)
    name_tuples = argparse.Namespace(data_sha256="d" * 64, pairs=frozenset({("a", "b")}))
    orcid_prefix_counts = argparse.Namespace(
        data_sha256="e" * 64,
        manifest_sha256="f" * 64,
        name_tuples_sha256=name_tuples.data_sha256,
    )
    expected_hashes = {
        "name_counts_manifest_sha256": name_counts_index.manifest_sha256,
        "name_tuples_data_sha256": name_tuples.data_sha256,
        "orcid_prefix_counts_data_sha256": orcid_prefix_counts.data_sha256,
        "orcid_prefix_counts_manifest_sha256": orcid_prefix_counts.manifest_sha256,
    }

    def fake_anddata(
        name: str,
        files: Any,
        *,
        mode: str,
        name_counts_index: Any,
        name_tuples: Any,
        random_seed: int = 1111,
    ) -> Any:
        assert name_counts_index is not None
        assert name_tuples == frozenset({("a", "b")})
        constructed.append(name)
        return argparse.Namespace(
            name=name,
            split_cluster_signatures=lambda: ({}, {"b1": ["s1", "s2"], "b2": ["s3"]}, {}),
            construct_cluster_to_signatures=lambda blocks: {"c1": ["s1", "s2"]},
        )

    def load_staging(_path: Path, *, expected_artifact_hashes: dict[str, str]) -> _StubCalibrationClusterer:
        assert expected_artifact_hashes == expected_hashes
        return clusterer

    monkeypatch.setattr(release_pairwise, "_load_pairwise_staging_model", load_staging)
    monkeypatch.setattr(
        release_pairwise,
        "_load_release_artifacts",
        lambda _args: release_pairwise.ProductionArtifactAuthority(
            name_counts_index=name_counts_index,
            name_tuples=name_tuples,
            orcid_prefix_counts=orcid_prefix_counts,
        ),
    )
    monkeypatch.setattr(release_pairwise, "_load_sealed_anddata", fake_anddata)

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
    assert report["name_counts_manifest_sha256"] == "c" * 64
    assert report["name_tuples_data_sha256"] == "d" * 64
    assert [trial["eps"] for trial in report["trials"]] == [0.3, 0.6]
    assert set(report["validation_identities"]) == {"qian", "pubmed"}
    assert all("blocks" not in identity for identity in report["validation_identities"].values())
    assert finalizations[0]["new_eps"] == pytest.approx(0.6)
    assert finalizations[0]["expected_artifact_hashes"] == {
        "name_counts_manifest_sha256": "c" * 64,
        "name_tuples_data_sha256": "d" * 64,
        "orcid_prefix_counts_data_sha256": "e" * 64,
        "orcid_prefix_counts_manifest_sha256": "f" * 64,
    }
    assert args.output_bundle.is_dir()
    assert args.output_report.is_file()


def test_evaluation_report_rejects_weakened_release_spec(tmp_path: Path) -> None:
    fixture = _evaluation_fixture(tmp_path)
    fixture.release_spec["thresholds"]["pairwise_aggregate_auroc_max_drop"] = 0.0011
    fixture.release_spec_path.write_text(json.dumps(fixture.release_spec) + "\n", encoding="utf-8")
    fixture.release_spec_sha256 = _sha256(fixture.release_spec_path)
    fixture.evidence_manifest_sha256 = _refresh_evidence_digest(
        fixture.evidence_manifest,
        "release_spec",
        fixture.release_spec_path,
    )

    with pytest.raises(ValueError, match="weakens the normative maximum"):
        release_pairwise.build_evaluation_report(
            _evaluation_report_args(
                fixture,
                tmp_path / "report.json",
            )
        )


def test_evaluation_report_is_repeatable_and_all_gates_pass(tmp_path: Path) -> None:
    fixture = _evaluation_fixture(tmp_path)
    first_path = tmp_path / "quality_report_1.json"
    second_path = tmp_path / "quality_report_2.json"
    first = release_pairwise.build_evaluation_report(_evaluation_report_args(fixture, first_path))
    second = release_pairwise.build_evaluation_report(_evaluation_report_args(fixture, second_path))

    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["passed"] is True
    assert first["release_spec_sha256"] == _sha256(fixture.paths["release_spec"])
    assert first["data_manifest_sha256"] == _sha256(fixture.paths["data_manifest"])
    assert first["model_manifest_sha256"] == _sha256(fixture.paths["complete_model_manifest"])
    assert first["evidence_manifest_sha256"] == fixture.evidence_manifest_sha256
    assert first["measurements"] == {
        "cluster": {
            "datasets": {"toy": {"f1": 0.8}},
            "signature_weighted": {"f1": 0.8},
        },
        "linker": _linker_metrics(),
        "pairwise": {
            "aggregate": {"auroc": 0.9, "macro_f1": 0.8},
            "datasets": {"toy": {"auroc": 0.9, "macro_f1": 0.8}},
        },
    }
    assert first["checks"] and all(check["passed"] is True for check in first["checks"])
    members = validate_release_evidence_manifest(
        json.loads(fixture.evidence_manifest.read_text(encoding="utf-8")),
        fixture.evidence_manifest,
        require_urls=True,
        verify_local_members=False,
    )
    assert set(members) == RELEASE_EVIDENCE_ROLES


def test_evaluation_report_rejects_tampered_evidence_manifest(tmp_path: Path) -> None:
    fixture = _evaluation_fixture(tmp_path)
    fixture.evidence_manifest.write_text(
        fixture.evidence_manifest.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Release-evidence manifest SHA-256 mismatch"):
        release_pairwise.build_evaluation_report(_evaluation_report_args(fixture, tmp_path / "report.json"))


def test_evaluation_report_rejects_tampered_evidence_member(tmp_path: Path) -> None:
    fixture = _evaluation_fixture(tmp_path)
    fixture.paths["pairwise_evaluation_report"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"member 'pairwise_evaluation_report'.*size mismatch"):
        release_pairwise.build_evaluation_report(_evaluation_report_args(fixture, tmp_path / "report.json"))


def test_evaluation_report_rejects_evidence_member_path_escape(tmp_path: Path) -> None:
    fixture = _evaluation_fixture(tmp_path)
    manifest = json.loads(fixture.evidence_manifest.read_text(encoding="utf-8"))
    manifest["members"]["linker_evaluation_report"]["path"] = "../linker.json"
    fixture.evidence_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"member 'linker_evaluation_report' path must be normalized relative POSIX",
    ):
        release_pairwise.build_evaluation_report(
            _evaluation_report_args(
                fixture,
                tmp_path / "report.json",
                evidence_manifest_sha256=_sha256(fixture.evidence_manifest),
            )
        )


def test_evaluation_report_rejects_invalid_member_contract(tmp_path: Path) -> None:
    cases = (
        ("pairwise_evaluation_report", "schema_version", "wrong", r"pairwise_evaluation_report.*schema mismatch"),
        ("data_manifest", "schema", "wrong", r"data_manifest.*schema mismatch"),
        (
            "complete_model_manifest",
            "incremental_linker_version",
            None,
            "complete_model_manifest.*must describe a complete bundle",
        ),
    )
    for index, (role, field, value, message) in enumerate(cases):
        root = tmp_path / str(index)
        root.mkdir()
        fixture = _evaluation_fixture(root)
        path = fixture.paths[role]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload[field] = value
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        manifest_sha256 = _refresh_evidence_digest(fixture.evidence_manifest, role, path)

        with pytest.raises(ValueError, match=message):
            release_pairwise.build_evaluation_report(
                _evaluation_report_args(
                    fixture,
                    root / "report.json",
                    evidence_manifest_sha256=manifest_sha256,
                )
            )


def test_evaluation_report_rejects_mismatched_bindings(tmp_path: Path) -> None:
    cases = (
        ("pairwise_evaluation_report", "release_spec_sha256", "pairwise_evaluation_report.release_spec"),
        ("linker_evaluation_report", "model_manifest_sha256", "linker_evaluation_report.complete_model"),
        (
            "pairwise_evaluation_report",
            "population_manifest_sha256",
            "pairwise_evaluation_report.population_manifest_sha256",
        ),
        (
            "performance_evaluation_report",
            "data_manifest_sha256",
            "performance_evaluation_report.data_manifest_sha256",
        ),
        (
            "subblocking_evaluation_report",
            "input_manifest_sha256",
            "subblocking_evaluation_report.input_manifest_sha256",
        ),
    )
    for index, (role, field, binding) in enumerate(cases):
        root = tmp_path / str(index)
        root.mkdir()
        fixture = _evaluation_fixture(root)
        path = fixture.paths[role]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload[field] = "0" * 64
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        manifest_sha256 = _refresh_evidence_digest(fixture.evidence_manifest, role, path)
        with pytest.raises(ValueError, match=rf"binding '{binding}' does not match"):
            release_pairwise.build_evaluation_report(
                _evaluation_report_args(
                    fixture,
                    root / "report.json",
                    evidence_manifest_sha256=manifest_sha256,
                )
            )


def test_evaluation_report_rejects_invalid_linker_measurements(tmp_path: Path) -> None:
    cases = (
        (("stratified_test_accuracy",), float("nan"), "non-finite"),
        (("stratified_test_accuracy",), 1.1, r"must be in \[0, 1\]"),
        (("stratified_test_errors",), -1, "must be a nonnegative integer"),
        (("training_rows",), 0, "training counts are inconsistent"),
        (("training_positive_rows",), 2, "training counts are inconsistent"),
        (("stratified_test_queries",), 0, "test counts are inconsistent"),
        (("stratified_test_errors",), 2, "test counts are inconsistent"),
        (
            ("weighted_average_error_weights", "false_link_error_rate"),
            0.0,
            "must be positive",
        ),
        (
            ("weighted_average_error_weights",),
            {"false_link_error_rate": 1.0},
            "weights have invalid fields",
        ),
    )
    for index, (metric_path, value, message) in enumerate(cases):
        root = tmp_path / str(index)
        root.mkdir()
        fixture = _evaluation_fixture(root)
        path = fixture.paths["linker_evaluation_report"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        container = payload["observed_metrics"]
        for field in metric_path[:-1]:
            container = container[field]
        container[metric_path[-1]] = value
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        manifest_sha256 = _refresh_evidence_digest(
            fixture.evidence_manifest,
            "linker_evaluation_report",
            path,
        )

        with pytest.raises(ValueError, match=message):
            release_pairwise.build_evaluation_report(
                _evaluation_report_args(
                    fixture,
                    root / "report.json",
                    evidence_manifest_sha256=manifest_sha256,
                )
            )


def test_evaluation_report_rejects_invalid_core_measurements(tmp_path: Path) -> None:
    cases = (
        (
            "cluster_evaluation_report",
            ("datasets", "toy", "metrics", "f1"),
            1.1,
            r"must be in \[0, 1\]",
        ),
        ("performance_evaluation_report", ("summary", "peak_rss_gb", "max"), -1.0, "peak RSS must be positive"),
        (
            "subblocking_evaluation_report",
            ("rust", "partition", "max_subblock_size"),
            -1,
            "maximum size must be a positive integer",
        ),
        (
            "subblocking_evaluation_report",
            ("rust", "partition", "max_subblock_size"),
            1.5,
            "maximum size must be a positive integer",
        ),
    )
    for index, (role, path, value, message) in enumerate(cases):
        root = tmp_path / str(index)
        root.mkdir()
        fixture = _evaluation_fixture(root)
        evidence_path = fixture.paths[role]
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        container = payload
        for field in path[:-1]:
            container = container[field]
        container[path[-1]] = value
        evidence_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        manifest_sha256 = _refresh_evidence_digest(fixture.evidence_manifest, role, evidence_path)

        with pytest.raises(ValueError, match=message):
            release_pairwise.build_evaluation_report(
                _evaluation_report_args(
                    fixture,
                    root / "report.json",
                    evidence_manifest_sha256=manifest_sha256,
                )
            )


def test_evaluation_report_records_each_failed_gate(tmp_path: Path) -> None:
    cases = (
        ("cluster_evaluation_report", ("signature_weighted", "f1"), 0.7, "cluster.signature_weighted.b3_f1_drop"),
        ("pairwise_evaluation_report", ("aggregate", "auroc"), 0.7, "pairwise.aggregate.auroc_drop"),
        ("pairwise_evaluation_report", ("aggregate", "macro_f1"), 0.7, "pairwise.aggregate.macro_f1_drop"),
        ("pairwise_evaluation_report", ("datasets", "toy", "metrics", "auroc"), 0.7, "pairwise.dataset.toy.auroc_drop"),
        (
            "pairwise_evaluation_report",
            ("datasets", "toy", "metrics", "macro_f1"),
            0.7,
            "pairwise.dataset.toy.macro_f1_drop",
        ),
        ("performance_evaluation_report", ("summary", "predict_seconds", "p50"), 12.0, "performance.runtime_ratio"),
        ("performance_evaluation_report", ("summary", "peak_rss_gb", "max"), 5.0, "performance.peak_rss_absolute_gb"),
        ("subblocking_evaluation_report", ("rust", "partition", "max_subblock_size"), 101, "subblocking.maximum_size"),
        (
            "subblocking_evaluation_report",
            ("rust", "component_preservation", "component_pair_recall"),
            0.9,
            "subblocking.member_preservation",
        ),
        ("parity_evaluation_report", ("clusters_exact_match",), False, "parity.clusters_exact_match"),
    )
    for index, (role, path, value, failed_gate) in enumerate(cases):
        root = tmp_path / str(index)
        root.mkdir()
        fixture = _evaluation_fixture(root)
        evidence_path = fixture.paths[role]
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        container = payload
        for field in path[:-1]:
            container = container[field]
        container[path[-1]] = value
        evidence_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        manifest_sha256 = _refresh_evidence_digest(fixture.evidence_manifest, role, evidence_path)
        report = release_pairwise.build_evaluation_report(
            _evaluation_report_args(
                fixture,
                root / "quality_report.json",
                evidence_manifest_sha256=manifest_sha256,
            )
        )
        assert report["passed"] is False
        assert {check["id"] for check in report["checks"] if check["passed"] is False} == {failed_gate}


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


def test_release_artifacts_use_one_external_root_and_packaged_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[Path] = []
    name_counts_index = object()
    name_tuples = argparse.Namespace(data_sha256="a" * 64)
    orcid_prefix_counts = argparse.Namespace(name_tuples_sha256=name_tuples.data_sha256)

    monkeypatch.setattr(
        release_pairwise,
        "NameCountsIndex",
        argparse.Namespace(open=lambda path: opened.append(path) or name_counts_index),
    )
    monkeypatch.setattr(release_pairwise, "load_packaged_name_tuple_artifact", lambda: name_tuples)
    monkeypatch.setattr(
        release_pairwise,
        "load_canonical_orcid_prefix_counts",
        lambda path: opened.append(Path(path)) or orcid_prefix_counts,
    )

    observed = release_pairwise._load_release_artifacts(  # noqa: SLF001
        argparse.Namespace(name_counts_index_root=Path("chosen-counts"))
    )

    assert observed.name_counts_index is name_counts_index
    assert observed.name_tuples is name_tuples
    assert observed.orcid_prefix_counts is orcid_prefix_counts
    assert opened == [Path("chosen-counts"), Path(release_pairwise._PACKAGE_DATA_DIR)]  # noqa: SLF001


def test_pair_evaluator_rejects_noninteger_json_labels(
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
                {
                    "signature_id_1": "s1",
                    "signature_id_2": "s2",
                    "label": True,
                }
            ],
        },
    )
    bindings = _write_evaluation_bindings(tmp_path, manifest_sha256)
    monkeypatch.setattr(
        release_pairwise,
        "load_production_model",
        lambda _: pytest.fail("model loaded before pair labels were validated"),
    )
    monkeypatch.setattr(
        release_pairwise,
        "_load_sealed_anddata",
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
    bindings = _write_evaluation_bindings(tmp_path, _sha256(manifest))
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
    bindings = _write_evaluation_bindings(tmp_path, manifest_sha256)
    name_counts_index, name_tuples, orcid_prefix_counts, artifact_args = _patch_evaluation_artifacts(monkeypatch)
    complete_loader_calls: list[Path] = []
    sealed_dataset_calls: list[dict[str, Any]] = []
    clusterer = argparse.Namespace(
        featurizer_info=object(),
        nameless_featurizer_info=object(),
        classifier=_StubBinaryClassifier([0.1, 0.9]),
        nameless_classifier=_StubBinaryClassifier([0.3, 0.7]),
        feature_contract={
            "name_counts_manifest_sha256": name_counts_index.manifest_sha256,
            "name_tuples_data_sha256": name_tuples.data_sha256,
        },
    )
    expected_hashes = {
        "name_counts_manifest_sha256": name_counts_index.manifest_sha256,
        "name_tuples_data_sha256": name_tuples.data_sha256,
        "orcid_prefix_counts_data_sha256": orcid_prefix_counts.data_sha256,
        "orcid_prefix_counts_manifest_sha256": orcid_prefix_counts.manifest_sha256,
    }

    def load_complete(path: Path, *, expected_artifact_hashes: dict[str, str]) -> object:
        complete_loader_calls.append(path)
        assert expected_artifact_hashes == expected_hashes
        return clusterer

    monkeypatch.setattr(release_pairwise, "load_production_model", load_complete)

    def load_sealed_dataset(*args: Any, **kwargs: Any) -> object:
        sealed_dataset_calls.append({"args": args, "kwargs": kwargs})
        return object()

    monkeypatch.setattr(release_pairwise, "_load_sealed_anddata", load_sealed_dataset)
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
            **artifact_args,
            **bindings,
        )
    )

    assert complete_loader_calls == [Path(bindings["model"]).resolve()]
    assert len(sealed_dataset_calls) == 1
    assert sealed_dataset_calls[0]["kwargs"]["name_counts_index"] is name_counts_index
    assert sealed_dataset_calls[0]["kwargs"]["name_tuples"] is name_tuples.pairs
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["release_spec_sha256"] == bindings["expected_release_spec_sha256"]
    assert report["model_manifest_sha256"] == bindings["expected_model_manifest_sha256"]
    assert report["name_counts_manifest_sha256"] == name_counts_index.manifest_sha256
    assert report["name_tuples_data_sha256"] == name_tuples.data_sha256
    assert report["orcid_prefix_counts_data_sha256"] == orcid_prefix_counts.data_sha256
    assert report["orcid_prefix_counts_manifest_sha256"] == orcid_prefix_counts.manifest_sha256
    assert report["population_manifest_sha256"] == manifest_sha256
    assert set(report["datasets"]["toy"]) == {"metrics"}
    assert "pairs" not in report["datasets"]["toy"]
    assert not list(tmp_path.glob("*unblind*"))
    assert not list(tmp_path.glob("*evaluation_start*"))
    assert not list(tmp_path.glob("*promotion*"))


def test_pair_evaluator_rejects_explicit_artifact_that_differs_from_model(
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
    name_counts_index, _name_tuples, _orcid_prefix_counts, artifact_args = _patch_evaluation_artifacts(monkeypatch)

    def reject_model(_path: Path, *, expected_artifact_hashes: dict[str, str]) -> None:
        assert expected_artifact_hashes["name_counts_manifest_sha256"] == name_counts_index.manifest_sha256
        raise ValueError("name_counts_manifest_sha256 does not match the explicit artifact authority")

    monkeypatch.setattr(release_pairwise, "load_production_model", reject_model)
    monkeypatch.setattr(
        release_pairwise,
        "many_pairs_featurize",
        lambda *args, **kwargs: pytest.fail("scoring began after an artifact mismatch"),
    )

    with pytest.raises(ValueError, match="name_counts_manifest_sha256 does not match"):
        release_pairwise.evaluate_pairs(
            argparse.Namespace(
                output_report=tmp_path / "report.json",
                manifest=manifest,
                expected_manifest_sha256=manifest_sha256,
                n_jobs=1,
                total_ram_bytes=None,
                **artifact_args,
                **_write_evaluation_bindings(tmp_path, manifest_sha256),
            )
        )


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
    bindings = _write_evaluation_bindings(
        tmp_path,
        manifest_sha256,
        manifest_schema=release_pairwise.CLUSTER_MANIFEST_SCHEMA,
    )
    name_counts_index, name_tuples, orcid_prefix_counts, artifact_args = _patch_evaluation_artifacts(monkeypatch)

    class StubClusterer:
        n_jobs = 1
        feature_contract = {
            "name_counts_manifest_sha256": name_counts_index.manifest_sha256,
            "name_tuples_data_sha256": name_tuples.data_sha256,
        }

        @staticmethod
        def predict(blocks: dict[str, list[str]], dataset: object) -> tuple[dict[str, list[str]], None]:
            return {"c1": ["s1"], "c2": ["s2"]}, None

    dataset = argparse.Namespace(construct_cluster_to_signatures=lambda blocks: {"c1": ["s1"], "c2": ["s2"]})
    observed_authorities: list[dict[str, str]] = []

    def load_complete(_path: Path, *, expected_artifact_hashes: dict[str, str]) -> StubClusterer:
        observed_authorities.append(expected_artifact_hashes)
        return StubClusterer()

    monkeypatch.setattr(release_pairwise, "load_production_model", load_complete)
    monkeypatch.setattr(
        release_pairwise,
        "_load_pairwise_staging_model",
        lambda _: pytest.fail("cluster evaluator used the staging-only loader"),
    )
    monkeypatch.setattr(release_pairwise, "_load_sealed_anddata", lambda *args, **kwargs: dataset)
    output = tmp_path / "cluster_report.json"

    report = release_pairwise.evaluate_clusters(
        argparse.Namespace(
            output_report=output,
            manifest=manifest,
            expected_manifest_sha256=manifest_sha256,
            n_jobs=2,
            **artifact_args,
            **bindings,
        )
    )

    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["release_spec_sha256"] == bindings["expected_release_spec_sha256"]
    assert report["model_manifest_sha256"] == bindings["expected_model_manifest_sha256"]
    assert report["name_counts_manifest_sha256"] == name_counts_index.manifest_sha256
    assert report["name_tuples_data_sha256"] == name_tuples.data_sha256
    assert report["orcid_prefix_counts_data_sha256"] == orcid_prefix_counts.data_sha256
    assert report["orcid_prefix_counts_manifest_sha256"] == orcid_prefix_counts.manifest_sha256
    assert observed_authorities == [
        {
            "name_counts_manifest_sha256": name_counts_index.manifest_sha256,
            "name_tuples_data_sha256": name_tuples.data_sha256,
            "orcid_prefix_counts_data_sha256": orcid_prefix_counts.data_sha256,
            "orcid_prefix_counts_manifest_sha256": orcid_prefix_counts.manifest_sha256,
        }
    ]
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
    bindings = _write_evaluation_bindings(
        tmp_path,
        manifest_sha256,
        manifest_schema=release_pairwise.CLUSTER_MANIFEST_SCHEMA,
    )
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


def test_release_spec_population_binding_fails_before_model_load(
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
            "pairs": [{"signature_id_1": "s1", "signature_id_2": "s2", "label": 0}],
        },
    )
    monkeypatch.setattr(
        release_pairwise,
        "load_production_model",
        lambda *_args, **_kwargs: pytest.fail("model loaded before population binding was validated"),
    )

    with pytest.raises(ValueError, match=r"populations\.pairwise_manifest_sha256 does not match"):
        release_pairwise.evaluate_pairs(
            argparse.Namespace(
                output_report=tmp_path / "report.json",
                manifest=manifest,
                expected_manifest_sha256=manifest_sha256,
                n_jobs=1,
                total_ram_bytes=None,
                **_write_evaluation_bindings(tmp_path, "0" * 64),
            )
        )


@pytest.mark.parametrize(
    ("binding", "message"),
    [
        ("expected_release_spec_sha256", "Release-spec SHA-256 mismatch"),
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
        **_write_evaluation_bindings(tmp_path, manifest_sha256),
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
