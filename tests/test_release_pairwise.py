from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from s2and.production_training_contract import block_membership_sha256, load_model_plan
from scripts.production.model import release_pairwise


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Stubbed datasets expose these development blocks; candidates record their digest.
_FAKE_BLOCKS = {"b1": ["s1", "s2"], "b2": ["s3"]}
_RECORDED_POPULATIONS = {
    "random": {"block_membership_sha256": block_membership_sha256(_FAKE_BLOCKS), "test_block_ids": ["b2"]}
}


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


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

    def make_distance_matrices_from_rust_featurizer(
        self,
        blocks: dict[str, list[str]],
        rust_featurizer: Any,
        total_ram_bytes: int | None = None,
    ) -> dict[str, object]:
        self.events.append(("build", rust_featurizer.dataset_name))
        self.distance_ram.append(total_ram_bytes)
        return {key: object() for key in blocks}

    def predict_eps_grid(
        self,
        blocks: dict[str, list[str]],
        dataset: Any,
        dists: dict[str, object],
        eps_grid: list[float],
    ) -> tuple[list[str], dict[float, np.ndarray]]:
        assert dists is not None
        self.events.append(("predict_eps_grid", dataset.name, tuple(float(eps) for eps in eps_grid)))
        # One cluster holds both signatures; its id encodes the EPS for the stub scorer.
        return ["s1", "s2"], {float(eps): np.full(2, round(eps * 1000), dtype=np.int64) for eps in eps_grid}


class _FakeRustFeaturizer:
    """Stand-in featurizer that records the cluster seeds overlaid on it."""

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        self.seeds: list[tuple[Any, Any]] = []

    def with_cluster_seeds(self, require: Any, disallow: Any) -> _FakeRustFeaturizer:
        self.seeds.append((require, disallow))
        return self


def _artifact_authority() -> release_pairwise.ProductionArtifactAuthority:
    tuples = argparse.Namespace(data_sha256="d" * 64, pairs=frozenset({("a", "b")}))
    return release_pairwise.ProductionArtifactAuthority(
        name_counts_index=argparse.Namespace(manifest_sha256="c" * 64),
        name_tuples=tuples,
        orcid_prefix_counts=argparse.Namespace(data_sha256="e" * 64, name_tuples_sha256=tuples.data_sha256),
    )


def _arrow_model_plan(
    tmp_path: Path,
    *,
    eps_grid: list[float] | None = None,
    minimum_dataset_f1: float = 0.0,
    minimum_signature_weighted_f1: float = 0.0,
    unknown: list[str] | None = None,
) -> Path:
    """Write a one-dataset Arrow plan; calibration stubs replace the Arrow data itself."""

    dataset = tmp_path / "dataset" / "random"
    files = {
        "arrow": _write_json(dataset / "manifest.json", {}),
        "clusters": _write_json(dataset / "random_clusters.json", {"c": {"signature_ids": ["s1", "s2"]}}),
        "heldout_blocks": _write_json(tmp_path / "dataset" / "heldout" / "test_blocks.json", {"b9": ["t1"]}),
        "heldout_pairs": _write_json(tmp_path / "dataset" / "heldout" / "test_pairs.json", []),
    }
    if unknown is not None:
        files["unknown_signatures"] = _write_json(dataset / "restored_unknown_signature_ids_20261009.json", unknown)
    return _write_json(
        tmp_path / "model_plan.json",
        {
            "release_version": "1.3",
            "datasets": {
                "random": {role: {"path": str(path), "sha256": _sha256(path)} for role, path in files.items()}
            },
            "eps": {
                "grid": eps_grid or [0.3, 0.6],
                "minimum_dataset_f1": minimum_dataset_f1,
                "minimum_signature_weighted_f1": minimum_signature_weighted_f1,
            },
        },
    )


def _calibration_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "source_bundle"
    (bundle / "reproducibility").mkdir(parents=True)
    _write_json(
        bundle / "reproducibility" / "pairwise_training_config.json",
        {"training_scope": "production_full", "data_random_seed": 1111},
    )
    _write_json(
        bundle / "reproducibility" / "pairwise_training_summary.json",
        {"cluster_test_splits": _RECORDED_POPULATIONS},
    )
    _write_json(bundle / "manifest.json", {})
    return bundle


def _patch_calibration(
    monkeypatch: pytest.MonkeyPatch,
    clusterer: _StubCalibrationClusterer,
    scores: dict[float, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    finalizations: list[dict[str, Any]] = []
    featurizer_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(release_pairwise, "_load_release_artifacts", lambda _args: _artifact_authority())

    def load_pending(*_args: Any, **kwargs: Any) -> _StubCalibrationClusterer:
        assert kwargs["expected_eps_calibration"] == "pending"
        return clusterer

    @contextmanager
    def open_arrow(dataset: Any):
        yield f"arrow:{dataset.files['arrow'].parent.name}"

    def build_featurizer(dataset: Any, arrow_dataset: Any, **kwargs: Any) -> _FakeRustFeaturizer:
        featurizer_calls.append({"name": dataset.name, "arrow": arrow_dataset, **kwargs})
        return _FakeRustFeaturizer(dataset.name)

    monkeypatch.setattr(release_pairwise, "_load_pairwise_staging_model", load_pending)
    monkeypatch.setattr(release_pairwise, "open_plan_arrow", open_arrow)
    monkeypatch.setattr(
        release_pairwise,
        "load_development_dataset",
        lambda name, *_args, **_kwargs: argparse.Namespace(
            name=name,
            get_blocks=lambda: _FAKE_BLOCKS,
            split_cluster_signatures=lambda: ({}, {"b1": ["s1", "s2"], "b2": ["s3"]}, {}),
            construct_cluster_to_signatures=lambda _blocks: {"c": ["s1", "s2"]},
            cluster_seeds_require={},
            cluster_seeds_disallow=set(),
        ),
    )
    monkeypatch.setattr(release_pairwise, "build_verified_rust_featurizer", build_featurizer)

    class StubScorer:
        def __init__(self, *_args: Any) -> None:
            pass

        @staticmethod
        def report(ids: np.ndarray) -> dict[str, Any]:
            return {"precision": 1.0, "recall": 1.0, "f1": scores[int(ids[0]) / 1000], "signature_count": 2}

    monkeypatch.setattr(release_pairwise, "_B3Scorer", StubScorer)

    def finalize(**kwargs: Any) -> None:
        finalizations.append(kwargs)
        shutil.copytree(kwargs["source_bundle_dir"], kwargs["output_bundle_dir"])

    monkeypatch.setattr(release_pairwise, "finalize_pairwise_eps", finalize)
    return finalizations, featurizer_calls


def _calibration_args(tmp_path: Path, bundle: Path, model_plan: Path, *, n_jobs: int = 1) -> argparse.Namespace:
    return argparse.Namespace(
        source_bundle=bundle,
        model_plan=model_plan,
        output_bundle=tmp_path / "calibrated",
        output_report=tmp_path / "calibration.json",
        name_counts_index_root=tmp_path / "counts",
        n_jobs=n_jobs,
        total_ram_bytes=2048,
    )


def test_calibrate_eps_binds_plan_reuses_matrices_applies_floors_and_smallest_tie_break(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_plan = _arrow_model_plan(
        tmp_path,
        eps_grid=[0.6, 0.4, 0.2],
        minimum_dataset_f1=0.75,
        minimum_signature_weighted_f1=0.75,
    )
    bundle = _calibration_bundle(tmp_path)
    clusterer = _StubCalibrationClusterer()
    finalizations, featurizer_calls = _patch_calibration(monkeypatch, clusterer, {0.2: 0.8, 0.4: 0.8, 0.6: 0.7})
    args = _calibration_args(tmp_path, bundle, model_plan, n_jobs=5)

    report = release_pairwise.calibrate_eps(args)

    # The plan's own Arrow dataset builds features for the filtered validation blocks only.
    assert [(call["name"], call["arrow"], call["num_threads"]) for call in featurizer_calls] == [
        ("random", "arrow:random", 5)
    ]
    assert featurizer_calls[0]["name_counts_manifest_sha256"] == "c" * 64
    # "b2" has one signature and is dropped.
    assert featurizer_calls[0]["signature_ids"] == {"s1", "s2"}
    assert [event for event in clusterer.events if event[0] == "build"] == [("build", "random")]
    # One call cuts the shared matrices at every EPS.
    assert [event for event in clusterer.events if event[0] == "predict_eps_grid"] == [
        ("predict_eps_grid", "random", (0.2, 0.4, 0.6))
    ]
    assert clusterer.distance_ram == [2048]
    assert clusterer.cluster_model.eps == pytest.approx(0.5)
    assert clusterer.n_jobs == 5
    assert report["selected_eps"] == pytest.approx(0.2)
    assert finalizations[0]["new_eps"] == pytest.approx(0.2)
    assert "expected_old_eps" not in finalizations[0]
    assert args.output_bundle.is_dir() and args.output_report.is_file()


@pytest.mark.parametrize("sidecar_change", [None, "relocated", "changed", "removed", "unbound"])
def test_eps_retains_unknown_inputs_scores_knowns_and_rejects_sidecar_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sidecar_change: str | None,
) -> None:
    from scripts.production.model.plan_datasets import plan_unknown_signatures

    actual_scorer = release_pairwise._B3Scorer
    model_plan = _arrow_model_plan(tmp_path, unknown=["s2"])
    bundle = _calibration_bundle(tmp_path)
    dataset = load_model_plan(model_plan).datasets["random"]
    sidecar = dataset.files["unknown_signatures"]
    config_path = bundle / "reproducibility" / "pairwise_training_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sidecar_change != "unbound":
        config["unknown_signature_inputs"] = {"random": plan_unknown_signatures(dataset).binding()}
        if sidecar_change == "relocated":
            # The training machine recorded another location for the same sidecar bytes.
            config["unknown_signature_inputs"]["random"]["path"] = str(tmp_path / "elsewhere" / sidecar.name)
        _write_json(config_path, config)
    plan_payload = json.loads(model_plan.read_text(encoding="utf-8"))
    if sidecar_change == "changed":
        # A fresh plan declares the new list, so only the training record disagrees.
        _write_json(sidecar, [])
        plan_payload["datasets"]["random"]["unknown_signatures"]["sha256"] = _sha256(sidecar)
    elif sidecar_change == "removed":
        sidecar.unlink()
        del plan_payload["datasets"]["random"]["unknown_signatures"]
    _write_json(model_plan, plan_payload)
    clusterer = _StubCalibrationClusterer()
    finalizations, featurizer_calls = _patch_calibration(monkeypatch, clusterer, {0.3: 1.0, 0.6: 1.0})
    monkeypatch.setattr(release_pairwise, "_B3Scorer", actual_scorer)
    args = _calibration_args(tmp_path, bundle, model_plan)
    if sidecar_change not in {None, "relocated"}:
        message = "trained with unknown-signature" if sidecar_change == "unbound" else "sidecar changed"
        with pytest.raises(ValueError, match=message):
            release_pairwise.calibrate_eps(args)
        assert not featurizer_calls
        assert not finalizations
        assert not args.output_bundle.exists()
        return
    report = release_pairwise.calibrate_eps(args)
    assert featurizer_calls[0]["signature_ids"] == {"s1", "s2"}
    assert report["validation_identities"]["random"]["signature_count"] == 2
    for trial in report["trials"]:
        assert trial["datasets"]["random"]["signature_count"] == 1
        assert trial["signature_count"] == 1
        assert trial["signature_weighted"]["f1"] == 1.0
    assert [event[2] for event in clusterer.events if event[0] == "predict_eps_grid"] == [(0.3, 0.6)]


@pytest.mark.parametrize("unknown", [frozenset(), frozenset({"s3", "s7", "s20"})], ids=["all-known", "unknowns"])
def test_b3_scorer_matches_b3_report(unknown: frozenset[str]) -> None:
    """Calibration scores each EPS with the scorer; it must report what _b3_report reports."""

    rng = np.random.default_rng(11)
    signatures = [f"s{index}" for index in range(40)]

    def partition(ids: np.ndarray, prefix: str) -> dict[str, list[str]]:
        clusters: dict[str, list[str]] = {}
        for signature, cluster in zip(signatures, ids.tolist(), strict=True):
            clusters.setdefault(f"{prefix}{cluster}", []).append(signature)
        return clusters

    true_clusters = partition(rng.integers(0, 9, len(signatures)), "t")
    scorer = release_pairwise._B3Scorer(true_clusters, signatures, unknown)
    for cluster_count in (1, 2, 5, 9, 15, 40):
        predicted_ids = rng.integers(0, cluster_count, len(signatures))
        expected = release_pairwise._b3_report(true_clusters, partition(predicted_ids, "p"), unknown)
        assert scorer.report(predicted_ids) == expected
    everyone = frozenset(signatures)
    with pytest.raises(ValueError, match="at least one known signature"):
        release_pairwise._B3Scorer(true_clusters, signatures, everyone).report(np.zeros(len(signatures)))


def test_calibrate_eps_rejects_a_model_trained_on_other_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_plan = _arrow_model_plan(tmp_path)
    bundle = _calibration_bundle(tmp_path)
    _write_json(
        bundle / "reproducibility" / "pairwise_training_summary.json",
        {"cluster_test_splits": {"random": {"block_membership_sha256": "0" * 64, "test_block_ids": ["b2"]}}},
    )
    clusterer = _StubCalibrationClusterer()
    finalizations, featurizer_calls = _patch_calibration(monkeypatch, clusterer, {0.3: 1.0, 0.6: 1.0})
    args = _calibration_args(tmp_path, bundle, model_plan)

    with pytest.raises(ValueError, match="differ from the model's training population for 'random'"):
        release_pairwise.calibrate_eps(args)

    assert not featurizer_calls
    assert not finalizations
    assert not args.output_bundle.exists()


def test_calibrate_eps_rejects_a_plan_without_published_arrow_data(tmp_path: Path) -> None:
    json_plan = _arrow_model_plan(tmp_path)
    payload = json.loads(json_plan.read_text(encoding="utf-8"))
    source = _write_json(tmp_path / "random_signatures.json", {})
    spec = {"path": str(source), "sha256": _sha256(source)}
    payload["datasets"]["random"] = {role: spec for role in ("signatures", "papers", "specter_embeddings", "clusters")}
    json_plan.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must name published Arrow data"):
        release_pairwise.calibrate_eps(_calibration_args(tmp_path, _calibration_bundle(tmp_path), json_plan))


def test_pairwise_predictions_forward_the_ram_budget_to_rust_featurization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[dict[str, Any]] = []

    def featurize(rust_featurizer: Any, pairs: list[Any], featurizer_info: Any, **kwargs: Any) -> tuple[Any, ...]:
        received.append({"featurizer": rust_featurizer, "pairs": pairs, "info": featurizer_info, **kwargs})
        rows = len(pairs)
        return np.zeros((rows, 2)), np.asarray([pair[2] for pair in pairs], dtype=float), np.zeros((rows, 1))

    monkeypatch.setattr(release_pairwise, "featurize_pairs_with_rust", featurize)
    classifier = argparse.Namespace(predict_proba=lambda rows: np.tile([0.2, 0.8], (len(rows), 1)))
    clusterer = argparse.Namespace(
        featurizer_info="main-info",
        nameless_featurizer_info="nameless-info",
        classifier=classifier,
        nameless_classifier=classifier,
    )
    featurizer = _FakeRustFeaturizer("pairs")

    release_pairwise._pairwise_predictions(
        [("a", "b", 1), ("c", "d", 0)],
        featurizer,
        clusterer,
        argparse.Namespace(n_jobs=3, total_ram_bytes=4096),
    )

    assert len(received) == 1
    assert received[0]["featurizer"] is featurizer
    assert received[0]["info"] == "main-info"
    assert received[0]["nameless_featurizer_info"] == "nameless-info"
    assert received[0]["n_jobs"] == 3
    assert received[0]["total_ram_bytes"] == 4096
