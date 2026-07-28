from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from s2and.arrow_inputs import PUBLIC_DATA_KIND
from s2and.consts import PUBLIC_DATA_FORMAT_VERSION
from s2and.incremental_linking.contracts import canonical_json_digest
from s2and.subblocking import GraphSubblockingConfig
from scripts.production.model import release_pairwise
from scripts.production.model.run_binding import (
    build_run_binding_payload,
    evaluation_plan_content_identity,
    load_run_binding,
    model_plan_content_identity,
    require_run_binding_matches,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def _write_text(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _write_public_data_root(root: Path, *, release_version: str = "1.3") -> tuple[Path, Path]:
    dataset_manifest = _write_json(root / "fixture" / "manifest.json", {})
    root_manifest = _write_json(
        root / "manifest.json",
        {
            "kind": PUBLIC_DATA_KIND,
            "release_version": release_version,
            "format_version": PUBLIC_DATA_FORMAT_VERSION,
            "dataset_manifests": {
                "fixture": {
                    "path": "fixture/manifest.json",
                    "sha256": _sha256(dataset_manifest),
                }
            },
        },
    )
    return root_manifest, dataset_manifest


def _dataset_files(root: Path, name: str, contents: dict[str, Any]) -> dict[str, str]:
    files: dict[str, str] = {}
    for role, content in contents.items():
        path = root / f"{name}_{role}{'.csv' if isinstance(content, str) else '.json'}"
        if isinstance(content, str):
            _write_text(path, content)
        else:
            _write_json(path, content)
        files[role] = str(path.resolve())
    return files


def _release_fixture(
    tmp_path: Path,
    *,
    fixed_test_pair: tuple[str, str] = ("f5", "f6"),
    random_test_pair: tuple[str, str] = ("rt1", "rt2"),
    heldout_labels: tuple[Any, Any] = (0, 1),
    eps_grid: list[float] | None = None,
    minimum_dataset_f1: float = 0.0,
    minimum_signature_weighted_f1: float = 0.0,
) -> argparse.Namespace:
    inputs = tmp_path / "inputs"
    model = {
        "random": _dataset_files(
            inputs,
            "model_random",
            {
                "signatures": {"r1": {}, "r2": {}},
                "papers": {},
                "specter_embeddings": {},
                "clusters": {"c": {"signature_ids": ["r1", "r2"]}},
            },
        ),
        "fixed": _dataset_files(
            inputs,
            "model_fixed",
            {
                "signatures": {"f1": {}, "f2": {}, "f3": {}, "f4": {}},
                "papers": {},
                "specter_embeddings": {},
                "train_pairs": "signature_id_1,signature_id_2,label\nf1,f2,YES\n",
                "val_pairs": "signature_id_1,signature_id_2,label\nf3,f4,0\n",
            },
        ),
    }
    pairwise = {
        "random": _dataset_files(
            inputs,
            "evaluation_pair_random",
            {
                "signatures": {random_test_pair[0]: {}, random_test_pair[1]: {}},
                "papers": {},
                "specter_embeddings": {},
                "pairs": [
                    {
                        "signature_id_1": random_test_pair[0],
                        "signature_id_2": random_test_pair[1],
                        "label": heldout_labels[0],
                    }
                ],
            },
        ),
        "fixed": _dataset_files(
            inputs,
            "evaluation_pair_fixed",
            {
                "signatures": {fixed_test_pair[0]: {}, fixed_test_pair[1]: {}},
                "papers": {},
                "specter_embeddings": {},
                "pairs": [
                    {
                        "signature_id_1": fixed_test_pair[0],
                        "signature_id_2": fixed_test_pair[1],
                        "label": heldout_labels[1],
                    }
                ],
            },
        ),
    }
    cluster = {
        "random": _dataset_files(
            inputs,
            "evaluation_cluster_random",
            {
                "signatures": {"ct1": {}, "ct2": {}},
                "papers": {},
                "specter_embeddings": {},
                "clusters": {"deliberately": "not validated during preparation"},
                "blocks": {"b": ["ct1", "ct2"]},
            },
        )
    }
    arrow_root = tmp_path / "arrow"
    arrow_root.mkdir()
    arrow_manifest = _write_json(arrow_root / "manifest.json", {"datasets": ["random"]})
    parity_dir = inputs / "parity"
    parity_signatures = _write_json(parity_dir / "signatures.json", {"p1": {}})
    parity_papers = _write_json(parity_dir / "papers.json", {})
    parity_specter = _write_text(parity_dir / "specter.pkl", "fixture")
    _write_json(
        parity_dir / "meta.json",
        {
            "block": "reviewed",
            "dataset": "random",
            "paths": {
                "papers": parity_papers.name,
                "signatures": parity_signatures.name,
                "specter": parity_specter.name,
            },
        },
    )
    component_members = _write_text(inputs / "candidate_components.parquet", "reviewed components")
    baseline_metrics = {
        "cluster_signature_weighted_b3_f1": 0.804,
        "pairwise_aggregate": {"auroc": 0.9005, "macro_f1": 0.804},
        "pairwise_datasets": {name: {"auroc": 0.9005, "macro_f1": 0.804} for name in pairwise},
        "predict_seconds_p50": 10.0,
    }

    def population_identity(datasets: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
        return {
            name: {role: _sha256(Path(path)) for role, path in sorted(files.items())}
            for name, files in sorted(datasets.items())
        }

    workload = {"dataset": "random", "n_jobs": 1}
    baseline_record = _write_json(
        tmp_path / "reviewed_v1.21_baseline.json",
        {
            "metric_contract": release_pairwise._BASELINE_METRIC_CONTRACT,
            "metrics": baseline_metrics,
            "performance": {
                "arrow_root_manifest_sha256": _sha256(arrow_manifest),
                "workload": workload,
            },
            "populations": {
                "cluster": population_identity(cluster),
                "pairwise": population_identity(pairwise),
            },
            "provenance": {
                "commands": ["uv run historical-baseline"],
                "data": {"release": "v1.21"},
                "environment": {"python": "3.11"},
                "model": {"eps": 0.65},
                "source_commit": "e54c6ba9c0e3ca4c2b5a40dcaa9a55c2c771d87d",
            },
        },
    )
    release = {
        "release_version": "1.3",
        "model": {
            "datasets": model,
            "eps": {
                "grid": eps_grid or [0.3, 0.6],
                "minimum_dataset_f1": minimum_dataset_f1,
                "minimum_signature_weighted_f1": minimum_signature_weighted_f1,
            },
        },
        "evaluation": {
            "pairwise": pairwise,
            "cluster": cluster,
            "performance": {
                "arrow_root": str(arrow_root),
                "workload": workload,
            },
            "parity": {
                "fixture_dir": str(parity_dir),
                "workload": {
                    "block_size": 100,
                    "compare_features": True,
                    "include_specter": True,
                    "n_jobs": 1,
                    "total_ram_bytes": 1_000_000,
                    "use_cluster_seeds": False,
                },
            },
            "subblocking": {
                "component_members_parquet": str(component_members),
                "dataset": "random",
                "workload": {
                    "allow_full": True,
                    "comparison_mode": "python-vs-rust",
                    "graph_config": asdict(GraphSubblockingConfig()),
                    "limit": None,
                    "maximum_size": 100,
                    "orcid_subblocking": True,
                    "python_source": "arrow",
                    "sample_mode": "random",
                    "seed": 42,
                    "top_diff_subblocks": 30,
                },
            },
            "baseline": {"path": str(baseline_record), "sha256": _sha256(baseline_record)},
            "gates": {
                "cluster_signature_weighted_b3_f1_max_drop": 0.005,
                "pairwise_aggregate_auroc_max_drop": 0.001,
                "pairwise_aggregate_macro_f1_max_drop": 0.005,
                "pairwise_dataset_auroc_max_drop": 0.001,
                "pairwise_dataset_macro_f1_max_drop": 0.005,
                "peak_rss_absolute_max_gb": 4.0,
                "runtime_max_ratio": 1.1,
                "subblocking_maximum_size": 100,
            },
        },
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    release_path = _write_json(run_dir / "release.json", release)
    return argparse.Namespace(
        release=release_path,
        run_dir=run_dir,
        payload=release,
        model=model,
        pairwise=pairwise,
        cluster=cluster,
        arrow_root=arrow_root,
        parity_dir=parity_dir,
        component_members=component_members,
        baseline_record=baseline_record,
    )


def _prepare(fixture: argparse.Namespace) -> tuple[Path, Path]:
    release_pairwise.prepare_run(argparse.Namespace(release=fixture.release))
    return fixture.run_dir / "model_plan.json", fixture.run_dir / "evaluation_plan.json"


def test_prepare_run_writes_three_simple_authorities_without_reading_heldout_answers(tmp_path: Path) -> None:
    fixture = _release_fixture(
        tmp_path,
        heldout_labels=({"preparation": "must ignore this"}, ["and", "this"]),
    )

    model_path, evaluation_path = _prepare(fixture)

    model = json.loads(model_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert set(fixture.run_dir.iterdir()) == {
        fixture.release,
        model_path,
        evaluation_path,
        fixture.run_dir / "stages",
        fixture.run_dir / "reports",
        fixture.run_dir / "final",
    }
    assert set(model) == {"release_version", "datasets", "eps"}
    assert model["release_version"] == "1.3"
    assert set(evaluation) == {
        "baseline_record_sha256",
        "baselines",
        "cluster",
        "gates",
        "parity",
        "pairwise",
        "performance",
        "subblocking",
    }
    for datasets in (model["datasets"], evaluation["pairwise"], evaluation["cluster"]):
        for files in datasets.values():
            for spec in files.values():
                assert set(spec) == {"path", "sha256"}
                assert Path(spec["path"]).is_absolute()
                assert spec["sha256"] == _sha256(Path(spec["path"]))
    assert evaluation["performance"] == {
        "arrow_root": str(fixture.arrow_root.resolve()),
        "arrow_root_manifest_sha256": _sha256(fixture.arrow_root / "manifest.json"),
        "workload": {"dataset": "random", "n_jobs": 1},
    }
    assert evaluation["parity"]["fixture_dir"] == str(fixture.parity_dir.resolve())
    assert evaluation["parity"]["dataset"] == "random"
    assert evaluation["parity"]["block"] == "reviewed"
    assert evaluation["subblocking"]["component_members"] == {
        "path": str(fixture.component_members.resolve()),
        "sha256": _sha256(fixture.component_members),
    }
    assert evaluation["subblocking"]["dataset"] == "random"
    assert evaluation["baseline_record_sha256"] == _sha256(fixture.baseline_record)


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    (
        ({"fixed_test_pair": ("f2", "f1")}, "train/test unordered pairs overlap"),
        ({"random_test_pair": ("r1", "rt2")}, "contains test signatures"),
    ),
)
def test_prepare_run_rejects_leakage_before_writes(tmp_path, fixture_kwargs, message) -> None:
    fixture = _release_fixture(tmp_path, **fixture_kwargs)

    with pytest.raises(ValueError, match=message):
        release_pairwise.prepare_run(argparse.Namespace(release=fixture.release))

    assert {path.name for path in fixture.run_dir.iterdir()} == {"release.json"}


def test_prepare_run_requires_a_fresh_directory_and_code_owned_gate_maxima(tmp_path: Path) -> None:
    fixture = _release_fixture(tmp_path)
    (fixture.run_dir / "notes.txt").write_text("not fresh", encoding="utf-8")
    with pytest.raises(ValueError, match="initially contain only"):
        release_pairwise.prepare_run(argparse.Namespace(release=fixture.release))
    (fixture.run_dir / "notes.txt").unlink()
    fixture.payload["evaluation"]["gates"]["runtime_max_ratio"] = 1.11
    _write_json(fixture.release, fixture.payload)
    with pytest.raises(ValueError, match="runtime_max_ratio"):
        release_pairwise.prepare_run(argparse.Namespace(release=fixture.release))


def test_prepare_run_verifies_the_reviewed_baseline_file_digest(tmp_path: Path) -> None:
    fixture = _release_fixture(tmp_path)
    _write_json(fixture.baseline_record, {"changed": True})

    with pytest.raises(ValueError, match="baseline SHA-256 mismatch"):
        release_pairwise.prepare_run(argparse.Namespace(release=fixture.release))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda record: record["populations"]["pairwise"]["fixed"].__setitem__("pairs", "f" * 64),
            "pairwise populations",
        ),
        (
            lambda record: record["performance"].__setitem__("workload", {"different": True}),
            "performance workload",
        ),
        (
            lambda record: record["metric_contract"].__setitem__("performance_statistic", "mean"),
            "metric_contract",
        ),
    ),
)
def test_prepare_run_rejects_a_baseline_for_a_different_contract(tmp_path, mutation, message) -> None:
    fixture = _release_fixture(tmp_path)
    record = json.loads(fixture.baseline_record.read_text(encoding="utf-8"))
    mutation(record)
    _write_json(fixture.baseline_record, record)
    fixture.payload["evaluation"]["baseline"]["sha256"] = _sha256(fixture.baseline_record)
    _write_json(fixture.release, fixture.payload)

    with pytest.raises(ValueError, match=message):
        release_pairwise.prepare_run(argparse.Namespace(release=fixture.release))


def test_bind_candidate_writes_one_content_based_run_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release_fixture(tmp_path)
    model_plan, evaluation_plan = _prepare(fixture)
    candidate = fixture.run_dir / "final" / "production_model_v1.3"
    training_config = _write_json(
        candidate / "reproducibility" / "pairwise_training_config.json",
        {
            "input_artifact_hashes": _artifact_authority().hashes,
            "model_plan_sha256": _sha256(model_plan),
        },
    )
    training_summary = _write_json(
        candidate / "reproducibility" / "pairwise_training_summary.json",
        {"pair_count": 1},
    )
    candidate_manifest = _write_json(
        candidate / "manifest.json",
        {
            "sha256": {
                "reproducibility/pairwise_training_config.json": _sha256(training_config),
                "reproducibility/pairwise_training_summary.json": _sha256(training_summary),
            }
        },
    )

    public_data_root = fixture.run_dir / "stages" / "public_data_root"
    public_manifest, _dataset_manifest = _write_public_data_root(public_data_root)
    observed_authority_roots: list[Path] = []
    validated_public_roots: list[Path] = []

    def load_authority(*, name_counts_index_root: Path) -> release_pairwise.ProductionArtifactAuthority:
        observed_authority_roots.append(name_counts_index_root)
        return _artifact_authority()

    monkeypatch.setattr(release_pairwise, "load_packaged_artifact_authority", load_authority)
    monkeypatch.setattr(
        release_pairwise,
        "validate_release_root",
        lambda root: validated_public_roots.append(root),
    )
    monkeypatch.setattr(
        release_pairwise,
        "load_production_model",
        lambda path, *, expected_artifact_hashes: argparse.Namespace(
            path=path,
            expected_artifact_hashes=expected_artifact_hashes,
            production_model_release_version="1.3",
        ),
    )

    result = release_pairwise.bind_candidate(
        argparse.Namespace(
            run_root=fixture.run_dir,
            candidate_model=candidate,
            public_data_root=public_data_root,
        )
    )

    binding_path = fixture.run_dir / "run_binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    model = json.loads(model_plan.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_plan.read_text(encoding="utf-8"))
    assert result == {
        "run_binding": str(binding_path),
        "run_binding_sha256": binding["run_binding_sha256"],
    }
    assert binding["candidate_model_manifest_sha256"] == _sha256(candidate_manifest)
    assert binding["baseline_record_sha256"] == evaluation["baseline_record_sha256"]
    assert binding["model_plan_content_sha256"] == canonical_json_digest(model_plan_content_identity(model))
    assert model_plan_content_identity(model)["release_version"] == "1.3"
    assert binding["public_data_root_manifest_sha256"] == _sha256(public_manifest)
    assert binding["public_data_root_manifest_sha256"] != evaluation["performance"]["arrow_root_manifest_sha256"]
    assert validated_public_roots == [public_data_root]
    assert observed_authority_roots == [public_data_root / "name_counts_index"]
    assert set(binding) == {
        "baseline_record_sha256",
        "candidate_model_manifest_sha256",
        "evaluation_plan_content_sha256",
        "model_plan_content_sha256",
        "public_data_root_manifest_sha256",
        "run_binding_sha256",
    }


def test_bind_candidate_rejects_mismatched_release_authorities_and_training_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release_fixture(tmp_path)
    model_plan, _evaluation_plan = _prepare(fixture)
    candidate = fixture.run_dir / "final" / "production_model_v1.3"
    training_config = _write_json(
        candidate / "reproducibility" / "pairwise_training_config.json",
        {
            "input_artifact_hashes": _artifact_authority().hashes,
            "model_plan_sha256": _sha256(model_plan),
        },
    )
    training_summary = _write_json(
        candidate / "reproducibility" / "pairwise_training_summary.json",
        {"pair_count": 1},
    )
    candidate_manifest = _write_json(
        candidate / "manifest.json",
        {
            "sha256": {
                "reproducibility/pairwise_training_config.json": _sha256(training_config),
                "reproducibility/pairwise_training_summary.json": _sha256(training_summary),
            }
        },
    )

    def refresh_training_config_binding() -> None:
        payload = json.loads(candidate_manifest.read_text(encoding="utf-8"))
        payload["sha256"]["reproducibility/pairwise_training_config.json"] = _sha256(training_config)
        _write_json(candidate_manifest, payload)

    public_data_root = fixture.run_dir / "stages" / "public_data_root"
    public_manifest, public_dataset_manifest = _write_public_data_root(public_data_root)
    loaded_model = argparse.Namespace(path=candidate, production_model_release_version="different")
    model_loads: list[dict[str, str]] = []

    def load_model(_path: Path, *, expected_artifact_hashes: dict[str, str]) -> argparse.Namespace:
        model_loads.append(expected_artifact_hashes)
        return loaded_model

    monkeypatch.setattr(release_pairwise, "load_packaged_artifact_authority", lambda **_kwargs: _artifact_authority())
    monkeypatch.setattr(release_pairwise, "load_production_model", load_model)
    monkeypatch.setattr(release_pairwise, "validate_release_root", lambda _root: {})

    args = argparse.Namespace(
        run_root=fixture.run_dir,
        candidate_model=candidate,
        public_data_root=public_data_root,
    )
    monkeypatch.setattr(
        release_pairwise,
        "validate_release_root",
        lambda _root: (_ for _ in ()).throw(ValueError("invalid final publication")),
    )
    with pytest.raises(ValueError, match="invalid final publication"):
        release_pairwise.bind_candidate(args)
    assert not model_loads
    monkeypatch.setattr(release_pairwise, "validate_release_root", lambda _root: {})

    with pytest.raises(ValueError, match="Candidate model release_version"):
        release_pairwise.bind_candidate(args)

    loaded_model.production_model_release_version = "1.3"
    public_payload = json.loads(public_manifest.read_text(encoding="utf-8"))
    public_payload["release_version"] = "different"
    _write_json(public_manifest, public_payload)
    with pytest.raises(ValueError, match="Public-data release_version"):
        release_pairwise.bind_candidate(args)

    public_payload["release_version"] = "1.3"
    public_payload["format_version"] = 2
    _write_json(public_manifest, public_payload)
    with pytest.raises(ValueError, match="format_version"):
        release_pairwise.bind_candidate(args)

    public_payload["format_version"] = PUBLIC_DATA_FORMAT_VERSION
    _write_json(public_manifest, public_payload)
    _write_json(public_dataset_manifest, {"dataset": "mutated"})
    with pytest.raises(ValueError, match=r"dataset_manifests\.fixture\.sha256 mismatch"):
        release_pairwise.bind_candidate(args)

    _write_json(public_dataset_manifest, {})
    public_payload["dataset_manifests"]["fixture"]["sha256"] = _sha256(public_dataset_manifest)
    _write_json(public_manifest, public_payload)
    config_payload = json.loads(training_config.read_text(encoding="utf-8"))
    config_payload["input_artifact_hashes"]["name_counts_manifest_sha256"] = "0" * 64
    _write_json(training_config, config_payload)
    refresh_training_config_binding()
    model_load_count = len(model_loads)
    with pytest.raises(ValueError, match="different artifact authority"):
        release_pairwise.bind_candidate(args)
    assert len(model_loads) == model_load_count

    config_payload["input_artifact_hashes"] = _artifact_authority().hashes
    config_payload["model_plan_sha256"] = "f" * 64
    _write_json(training_config, config_payload)
    refresh_training_config_binding()
    with pytest.raises(ValueError, match="different model plan"):
        release_pairwise.bind_candidate(args)

    candidate_payload = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    candidate_payload["sha256"].pop("reproducibility/pairwise_training_config.json")
    _write_json(candidate_manifest, candidate_payload)
    with pytest.raises(ValueError, match="must bind both pairwise training reproducibility files"):
        release_pairwise.bind_candidate(args)


def test_run_binding_rejects_a_conflicting_readable_baseline_identity(tmp_path: Path) -> None:
    fixture = _release_fixture(tmp_path)
    _model_plan, evaluation_plan = _prepare(fixture)
    evaluation = json.loads(evaluation_plan.read_text(encoding="utf-8"))
    binding_path = fixture.run_dir / "conflicting_binding.json"
    _write_json(
        binding_path,
        build_run_binding_payload(
            {
                "baseline_record_sha256": "f" * 64,
                "candidate_model_manifest_sha256": "a" * 64,
                "evaluation_plan_content_sha256": canonical_json_digest(evaluation_plan_content_identity(evaluation)),
                "model_plan_content_sha256": "b" * 64,
                "public_data_root_manifest_sha256": "c" * 64,
            }
        ),
    )

    with pytest.raises(ValueError, match="baseline record"):
        require_run_binding_matches(
            load_run_binding(binding_path),
            evaluation_plan=evaluation_plan,
        )


def test_evaluation_content_identity_excludes_only_operational_paths(tmp_path: Path) -> None:
    fixture = _release_fixture(tmp_path)
    _model_plan, evaluation_plan = _prepare(fixture)
    first = json.loads(evaluation_plan.read_text(encoding="utf-8"))
    relocated = json.loads(json.dumps(first))
    relocated_root = (tmp_path / "relocated").resolve()
    relocated["performance"]["arrow_root"] = str(relocated_root / "performance")
    relocated["parity"]["fixture_dir"] = str(relocated_root / "parity")
    for spec in relocated["parity"]["files"].values():
        spec["path"] = str(relocated_root / "parity" / "input")
    relocated["subblocking"]["component_members"]["path"] = str(relocated_root / "components.parquet")
    for population in ("pairwise", "cluster"):
        for files in relocated[population].values():
            for spec in files.values():
                spec["path"] = str(relocated_root / "evaluation" / "input")

    assert evaluation_plan_content_identity(first) == evaluation_plan_content_identity(relocated)
    relocated["subblocking"]["dataset"] = "different"
    assert evaluation_plan_content_identity(first) != evaluation_plan_content_identity(relocated)


class _StubCalibrationClusterer:
    def __init__(self) -> None:
        self.cluster_model = argparse.Namespace(eps=0.5)
        self.n_jobs = 1
        self.events: list[tuple[Any, ...]] = []
        self.distance_ram: list[int | None] = []

    @staticmethod
    def filter_blocks(blocks: dict[str, list[str]], num_to_keep: int | None = None) -> dict[str, list[str]]:
        assert num_to_keep is None
        return {key: values for key, values in blocks.items() if len(values) > 1}

    def make_distance_matrices(
        self,
        blocks: dict[str, list[str]],
        dataset: Any,
        total_ram_bytes: int | None = None,
    ) -> dict[str, object]:
        self.events.append(("build", dataset.name))
        self.distance_ram.append(total_ram_bytes)
        return {key: object() for key in blocks}

    def predict(
        self,
        blocks: dict[str, list[str]],
        dataset: Any,
        dists: dict[str, object] | None = None,
        total_ram_bytes: int | None = None,
    ) -> tuple[dict[str, list[str]], None]:
        assert dists is not None
        self.events.append(("predict", dataset.name, float(self.cluster_model.eps), total_ram_bytes))
        return {"cluster": ["s1", "s2"]}, None


def _artifact_authority() -> release_pairwise.ProductionArtifactAuthority:
    tuples = argparse.Namespace(data_sha256="d" * 64, pairs=frozenset({("a", "b")}))
    return release_pairwise.ProductionArtifactAuthority(
        name_counts_index=argparse.Namespace(manifest_sha256="c" * 64),
        name_tuples=tuples,
        orcid_prefix_counts=argparse.Namespace(data_sha256="e" * 64, name_tuples_sha256=tuples.data_sha256),
    )


def _calibration_bundle(tmp_path: Path, model_plan: Path) -> Path:
    bundle = tmp_path / "source_bundle"
    (bundle / "reproducibility").mkdir(parents=True)
    _write_json(
        bundle / "reproducibility" / "pairwise_training_config.json",
        {
            "training_scope": "production_full",
            "data_random_seed": 1111,
            "model_plan_sha256": _sha256(model_plan),
        },
    )
    _write_json(bundle / "manifest.json", {})
    return bundle


def _patch_calibration(
    monkeypatch: pytest.MonkeyPatch,
    clusterer: _StubCalibrationClusterer,
    scores: dict[float, float],
) -> list[dict[str, Any]]:
    finalizations: list[dict[str, Any]] = []
    monkeypatch.setattr(release_pairwise, "_load_release_artifacts", lambda _args: _artifact_authority())

    def load_pending(*_args: Any, **kwargs: Any) -> _StubCalibrationClusterer:
        assert kwargs["expected_eps_calibration"] == "pending"
        return clusterer

    monkeypatch.setattr(
        release_pairwise,
        "_load_pairwise_staging_model",
        load_pending,
    )
    monkeypatch.setattr(
        release_pairwise,
        "_load_anddata",
        lambda name, *_args, **_kwargs: argparse.Namespace(
            name=name,
            split_cluster_signatures=lambda: ({}, {"b1": ["s1", "s2"], "b2": ["s3"]}, {}),
            construct_cluster_to_signatures=lambda _blocks: {"c": ["s1", "s2"]},
        ),
    )
    monkeypatch.setattr(
        release_pairwise,
        "_b3_report",
        lambda *_args: {
            "precision": 1.0,
            "recall": 1.0,
            "f1": scores[float(clusterer.cluster_model.eps)],
            "signature_count": 2,
        },
    )

    def finalize(**kwargs: Any) -> None:
        finalizations.append(kwargs)
        shutil.copytree(kwargs["source_bundle_dir"], kwargs["output_bundle_dir"])

    monkeypatch.setattr(release_pairwise, "finalize_pairwise_eps", finalize)
    return finalizations


def test_calibrate_eps_binds_plan_reuses_matrices_applies_floors_and_smallest_tie_break(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release_fixture(
        tmp_path,
        eps_grid=[0.6, 0.4, 0.2],
        minimum_dataset_f1=0.75,
        minimum_signature_weighted_f1=0.75,
    )
    model_plan, _ = _prepare(fixture)
    bundle = _calibration_bundle(tmp_path, model_plan)
    clusterer = _StubCalibrationClusterer()
    finalizations = _patch_calibration(monkeypatch, clusterer, {0.2: 0.8, 0.4: 0.8, 0.6: 0.7})
    args = argparse.Namespace(
        source_bundle=bundle,
        model_plan=model_plan,
        output_bundle=tmp_path / "calibrated",
        output_report=tmp_path / "calibration.json",
        name_counts_index_root=tmp_path / "counts",
        n_jobs=5,
        total_ram_bytes=2048,
    )

    report = release_pairwise.calibrate_eps(args)

    assert [event for event in clusterer.events if event[0] == "build"] == [("build", "random")]
    assert [event[2] for event in clusterer.events if event[0] == "predict"] == [0.2, 0.4, 0.6]
    assert clusterer.distance_ram == [2048]
    assert clusterer.cluster_model.eps == pytest.approx(0.5)
    assert clusterer.n_jobs == 5
    assert report["selected_eps"] == pytest.approx(0.2)
    assert report["model_plan_sha256"] == _sha256(model_plan)
    assert report["source_bundle_manifest_sha256"] == _sha256(bundle / "manifest.json")
    assert report["output_bundle_manifest_sha256"] == _sha256(args.output_bundle / "manifest.json")
    assert report["input_artifact_hashes"] == _artifact_authority().hashes
    assert finalizations[0]["new_eps"] == pytest.approx(0.2)
    assert "expected_old_eps" not in finalizations[0]
    assert args.output_bundle.is_dir() and args.output_report.is_file()


def test_calibrate_eps_rejects_a_different_model_plan_binding_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release_fixture(tmp_path)
    model_plan, _ = _prepare(fixture)
    bundle = _calibration_bundle(tmp_path, model_plan)
    config = bundle / "reproducibility" / "pairwise_training_config.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["model_plan_sha256"] = "0" * 64
    _write_json(config, payload)
    monkeypatch.setattr(
        release_pairwise,
        "_load_pairwise_staging_model",
        lambda *_args, **_kwargs: pytest.fail("model must not load"),
    )

    with pytest.raises(ValueError, match="different model plan"):
        release_pairwise.calibrate_eps(
            argparse.Namespace(
                source_bundle=bundle,
                model_plan=model_plan,
                output_bundle=tmp_path / "calibrated",
                output_report=tmp_path / "calibration.json",
            )
        )


def _evaluation_args(
    fixture: argparse.Namespace,
    evaluation_plan: Path,
    *,
    output: Path,
) -> argparse.Namespace:
    model = fixture.run_dir / "final" / "model"
    model.mkdir(parents=True, exist_ok=True)
    model_manifest = _write_json(model / "manifest.json", {"candidate": "test"})
    evaluation_payload = json.loads(evaluation_plan.read_text(encoding="utf-8"))
    binding_path = fixture.run_dir / "run_binding.json"
    binding = build_run_binding_payload(
        {
            "baseline_record_sha256": evaluation_payload["baseline_record_sha256"],
            "candidate_model_manifest_sha256": _sha256(model_manifest),
            "evaluation_plan_content_sha256": canonical_json_digest(
                evaluation_plan_content_identity(evaluation_payload)
            ),
            "model_plan_content_sha256": "0" * 64,
            "public_data_root_manifest_sha256": evaluation_payload["performance"]["arrow_root_manifest_sha256"],
        }
    )
    _write_json(binding_path, binding)
    return argparse.Namespace(
        model=model,
        evaluation_plan=evaluation_plan,
        run_binding=binding_path,
        name_counts_index_root=Path("counts"),
        output_report=output,
        n_jobs=2,
        total_ram_bytes=None,
        _model_dir=model,
    )


def _patch_evaluation_model(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> argparse.Namespace:
    model = argparse.Namespace(
        n_jobs=1,
        predict=lambda blocks, _dataset: ({"predicted": next(iter(blocks.values()))}, None),
    )
    monkeypatch.setattr(
        release_pairwise,
        "_load_release_artifacts",
        lambda _args: events.append("artifacts") or _artifact_authority(),
    )
    monkeypatch.setattr(
        release_pairwise,
        "load_production_model",
        lambda *_args, **_kwargs: events.append("model") or model,
    )
    return model


def test_pair_evaluator_loads_complete_model_before_plan_and_ignores_cluster_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release_fixture(tmp_path)
    _, evaluation_plan = _prepare(fixture)
    _write_json(Path(fixture.cluster["random"]["blocks"]), {"changed": ["not", "verified"]})
    events: list[str] = []
    _patch_evaluation_model(monkeypatch, events)
    real_load = release_pairwise._load_evaluation_plan
    monkeypatch.setattr(
        release_pairwise,
        "_load_evaluation_plan",
        lambda path: events.append("plan") or real_load(path),
    )
    monkeypatch.setattr(
        release_pairwise,
        "_load_anddata",
        lambda name, *_args, **_kwargs: argparse.Namespace(name=name),
    )

    def predictions(pairs: list[tuple[str, str, int]], dataset: Any, *_args: Any) -> tuple[Any, ...]:
        label = np.asarray([pairs[0][2]])
        probability = np.asarray([0.8 if label[0] else 0.2])
        return {"rows": 1, "auroc": 1.0, "macro_f1": 1.0}, label, probability, probability

    monkeypatch.setattr(release_pairwise, "_pairwise_predictions", predictions)
    args = _evaluation_args(fixture, evaluation_plan, output=tmp_path / "pairs_report.json")

    report = release_pairwise.evaluate_pairs(args)

    assert events[:3] == ["artifacts", "model", "plan"]
    assert set(report["datasets"]) == {"fixed", "random"}
    assert report["aggregate"]["rows"] == 2


def test_cluster_evaluator_loads_complete_model_before_plan_and_ignores_pair_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release_fixture(tmp_path)
    _, evaluation_plan = _prepare(fixture)
    _write_json(Path(fixture.pairwise["fixed"]["pairs"]), [{"changed": True}])
    events: list[str] = []
    _patch_evaluation_model(monkeypatch, events)
    real_load = release_pairwise._load_evaluation_plan
    monkeypatch.setattr(
        release_pairwise,
        "_load_evaluation_plan",
        lambda path: events.append("plan") or real_load(path),
    )
    monkeypatch.setattr(
        release_pairwise,
        "_load_anddata",
        lambda name, *_args, **_kwargs: argparse.Namespace(
            name=name,
            construct_cluster_to_signatures=lambda blocks: {"truth": next(iter(blocks.values()))},
        ),
    )
    monkeypatch.setattr(
        release_pairwise,
        "_b3_report",
        lambda *_args: {"precision": 1.0, "recall": 1.0, "f1": 1.0, "signature_count": 2},
    )
    args = _evaluation_args(fixture, evaluation_plan, output=tmp_path / "clusters_report.json")

    report = release_pairwise.evaluate_clusters(args)

    assert events[:3] == ["artifacts", "model", "plan"]
    assert set(report["datasets"]) == {"random"}
    assert report["signature_weighted"]["f1"] == 1.0


def test_pair_labels_are_checked_only_after_complete_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release_fixture(tmp_path, heldout_labels=({"bad": "label"}, 1))
    _, evaluation_plan = _prepare(fixture)
    events: list[str] = []
    _patch_evaluation_model(monkeypatch, events)

    with pytest.raises(ValueError, match="JSON integer"):
        release_pairwise.evaluate_pairs(
            _evaluation_args(fixture, evaluation_plan, output=tmp_path / "pairs_report.json")
        )

    assert events == ["artifacts", "model"]


@pytest.mark.parametrize(
    ("command_name", "changed_path"),
    (
        ("evaluate_pairs", ("pairwise", "fixed", "pairs")),
        ("evaluate_clusters", ("cluster", "random", "blocks")),
    ),
    ids=("pairs", "clusters"),
)
def test_evaluators_reject_a_changed_heldout_file_after_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command_name: str,
    changed_path: tuple[str, str, str],
) -> None:
    fixture = _release_fixture(tmp_path)
    _, evaluation_plan = _prepare(fixture)
    section, dataset, role = changed_path
    _write_text(Path(getattr(fixture, section)[dataset][role]), "changed")
    events: list[str] = []
    _patch_evaluation_model(monkeypatch, events)

    command = getattr(release_pairwise, command_name)
    with pytest.raises(ValueError, match="changed after release preparation"):
        command(_evaluation_args(fixture, evaluation_plan, output=tmp_path / "report.json"))

    assert events == ["artifacts", "model"]


def _write_release_reports(fixture: argparse.Namespace, evaluation_plan: Path) -> argparse.Namespace:
    plan = release_pairwise._load_evaluation_plan(evaluation_plan)
    run_binding = _evaluation_args(
        fixture,
        evaluation_plan,
        output=fixture.run_dir / "reports" / "unused.json",
    ).run_binding
    run_binding_sha256 = json.loads(run_binding.read_text(encoding="utf-8"))["run_binding_sha256"]
    report_dir = fixture.run_dir / "reports"
    payloads = {
        "pairwise_evaluation_report": {
            "run_binding_sha256": run_binding_sha256,
            "aggregate": {"auroc": 0.9, "macro_f1": 0.8},
            "datasets": {name: {"metrics": {"auroc": 0.9, "macro_f1": 0.8}} for name in plan.pairwise},
        },
        "cluster_evaluation_report": {
            "run_binding_sha256": run_binding_sha256,
            "datasets": {"random": {"metrics": {"f1": 0.8}}},
            "signature_weighted": {"f1": 0.8},
        },
        "performance_evaluation_report": {
            "run_binding_sha256": run_binding_sha256,
            "workload": dict(plan.workload),
            "arrow_root": str(plan.arrow_root),
            "summary": {"predict_seconds": {"p50": 10.5}, "peak_rss_gb": {"max": 2.5}},
        },
        "subblocking_evaluation_report": {
            "run_binding_sha256": run_binding_sha256,
            "rust": {
                "partition": {"max_subblock_size": 50},
                "component_preservation": {"component_pair_recall": 1.0},
            },
        },
        "parity_evaluation_report": {
            "run_binding_sha256": run_binding_sha256,
            "clusters_exact_match": True,
        },
    }
    paths = {}
    for role, payload in payloads.items():
        paths[role] = _write_json(report_dir / f"{role}.json", payload)
    return argparse.Namespace(plan=plan, report_dir=report_dir, paths=paths, run_binding=run_binding)


def _gate_args(evaluation_plan: Path, reports: argparse.Namespace, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        evaluation_plan=evaluation_plan,
        run_binding=reports.run_binding,
        report_dir=reports.report_dir,
        output_report=output,
    )


def test_evaluate_release_reads_five_reports_and_all_gates_pass(tmp_path: Path) -> None:
    fixture = _release_fixture(tmp_path)
    _, evaluation_plan = _prepare(fixture)
    reports = _write_release_reports(fixture, evaluation_plan)

    result = release_pairwise.build_evaluation_report(_gate_args(evaluation_plan, reports, tmp_path / "decision.json"))

    assert result["passed"] is True
    assert result["checks"] and all(check["passed"] for check in result["checks"])
    assert (
        result["run_binding_sha256"]
        == json.loads(reports.run_binding.read_text(encoding="utf-8"))["run_binding_sha256"]
    )
    assert set(path.name for path in reports.report_dir.iterdir()) == set(
        release_pairwise._RELEASE_GATE_INPUTS.values()
    )
    assert not (reports.report_dir / "release_spec.json").exists()


@pytest.mark.parametrize(
    ("role", "mutation", "message"),
    (
        (
            "pairwise_evaluation_report",
            lambda payload: payload["datasets"].pop("fixed"),
            "datasets must exactly match",
        ),
        (
            "performance_evaluation_report",
            lambda payload: payload.__setitem__("workload", {"different": True}),
            "workload does not match",
        ),
        (
            "performance_evaluation_report",
            lambda payload: payload.__setitem__("arrow_root", "different"),
            "arrow_root does not match",
        ),
        (
            "pairwise_evaluation_report",
            lambda payload: payload.__setitem__("run_binding_sha256", "f" * 64),
            "wrong run binding",
        ),
    ),
    ids=("pairwise-datasets", "performance-workload", "performance-arrow-root", "pairwise-run-binding"),
)
def test_evaluate_release_rejects_identity_mismatch(
    tmp_path: Path,
    role: str,
    mutation: Any,
    message: str,
) -> None:
    fixture = _release_fixture(tmp_path)
    _, evaluation_plan = _prepare(fixture)
    reports = _write_release_reports(fixture, evaluation_plan)
    payload = json.loads(reports.paths[role].read_text(encoding="utf-8"))
    mutation(payload)
    _write_json(reports.paths[role], payload)

    with pytest.raises(ValueError, match=message):
        release_pairwise.build_evaluation_report(_gate_args(evaluation_plan, reports, tmp_path / "decision.json"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda _payload: [], "must be a JSON object"),
        (
            lambda payload: (payload["aggregate"].pop("auroc"), payload)[1],
            "missing 'aggregate.auroc'",
        ),
        (
            lambda payload: (payload["aggregate"].__setitem__("auroc", float("nan")), payload)[1],
            "contains a non-finite number",
        ),
    ),
    ids=("non-object", "missing-auroc", "nonfinite-auroc"),
)
def test_evaluate_release_rejects_malformed_consumed_report_content(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    fixture = _release_fixture(tmp_path)
    _, evaluation_plan = _prepare(fixture)
    reports = _write_release_reports(fixture, evaluation_plan)
    report_path = reports.paths["pairwise_evaluation_report"]
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    _write_json(report_path, mutation(payload))

    with pytest.raises(ValueError, match=message):
        release_pairwise.build_evaluation_report(_gate_args(evaluation_plan, reports, tmp_path / "decision.json"))


def test_evaluate_release_records_every_failed_gate(tmp_path: Path) -> None:
    fixture = _release_fixture(tmp_path)
    _, evaluation_plan = _prepare(fixture)
    reports = _write_release_reports(fixture, evaluation_plan)
    pairwise = json.loads(reports.paths["pairwise_evaluation_report"].read_text(encoding="utf-8"))
    pairwise["aggregate"] = {"auroc": 0.0, "macro_f1": 0.0}
    for dataset in pairwise["datasets"].values():
        dataset["metrics"] = {"auroc": 0.0, "macro_f1": 0.0}
    _write_json(reports.paths["pairwise_evaluation_report"], pairwise)
    cluster = json.loads(reports.paths["cluster_evaluation_report"].read_text(encoding="utf-8"))
    cluster["signature_weighted"]["f1"] = 0.0
    _write_json(reports.paths["cluster_evaluation_report"], cluster)
    performance = json.loads(reports.paths["performance_evaluation_report"].read_text(encoding="utf-8"))
    performance["summary"] = {"predict_seconds": {"p50": 12.0}, "peak_rss_gb": {"max": 5.0}}
    _write_json(reports.paths["performance_evaluation_report"], performance)
    subblocking = json.loads(reports.paths["subblocking_evaluation_report"].read_text(encoding="utf-8"))
    subblocking["rust"]["partition"]["max_subblock_size"] = 101
    subblocking["rust"]["component_preservation"]["component_pair_recall"] = 0.99
    _write_json(reports.paths["subblocking_evaluation_report"], subblocking)
    parity = json.loads(reports.paths["parity_evaluation_report"].read_text(encoding="utf-8"))
    parity["clusters_exact_match"] = False
    _write_json(reports.paths["parity_evaluation_report"], parity)

    result = release_pairwise.build_evaluation_report(_gate_args(evaluation_plan, reports, tmp_path / "decision.json"))

    assert result["passed"] is False
    assert result["checks"] and all(not check["passed"] for check in result["checks"])
    assert len(result["checks"]) == 12


def test_every_command_refuses_an_existing_output_before_other_inputs(tmp_path: Path) -> None:
    output = _write_json(tmp_path / "exists.json", {})
    with pytest.raises(FileExistsError):
        release_pairwise.evaluate_pairs(argparse.Namespace(output_report=output))
    with pytest.raises(FileExistsError):
        release_pairwise.evaluate_clusters(argparse.Namespace(output_report=output))
    with pytest.raises(FileExistsError):
        release_pairwise.build_evaluation_report(argparse.Namespace(output_report=output))
