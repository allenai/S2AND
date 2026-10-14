"""Check paired features, source exclusions, metric deltas, and block units."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest

from s2and._sha256 import sha256_file
from s2and.featurizer import FeaturizationInfo
from s2and.production_bundle_contract import CALIBRATED_EPS_CALIBRATION, PENDING_EPS_CALIBRATION
from s2and.production_training_contract import load_packaged_artifact_authority
from scripts.verification import compare_pairwise_models as comparison
from tests.pairwise_release_helpers import convert_release_arrow, write_pairwise_artifacts, write_pairwise_dataset
from tests.promoted_linking_helpers import write_synthetic_pairwise_bundle


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


@pytest.fixture
def comparison_fixture(tmp_path: Path) -> argparse.Namespace:
    _, index_root, _ = write_pairwise_artifacts(tmp_path)
    authority = load_packaged_artifact_authority(name_counts_index_root=index_root)
    source_root = tmp_path / "sources"
    files = write_pairwise_dataset(source_root)
    signatures = json.loads(files["signatures"].read_text(encoding="utf-8"))
    blocks: dict[str, list[str]] = {}
    for signature_id, signature in signatures.items():
        blocks.setdefault(signature["author_info"]["block"], []).append(signature_id)
    rows = [
        {"signature_id_1": ids[left], "signature_id_2": ids[right], "label": label}
        for ids in blocks.values()
        for left, right, label in ((0, 1, 1), (0, 2, 0), (2, 3, 1))
    ]
    rows.append({"signature_id_1": "signature-0-0", "signature_id_2": "signature-1-0", "label": 0})
    files.update(
        blocks=files["signatures"].parent / "test_blocks.json",
        pairs=files["signatures"].parent / "test_pairs.json",
        unknown_signatures=files["signatures"].parent / "restored_unknown_signature_ids_test.json",
    )
    _write(files["blocks"], blocks)
    _write(files["pairs"], rows)
    _write(files["unknown_signatures"], ["signature-0-3"])
    augmented_files = {}
    for role in ("papers", "specter_embeddings", "signatures"):
        path = source_root / "augmented" / files[role].name.replace("qian_", "augmented_")
        path.parent.mkdir(parents=True, exist_ok=True)
        if role == "signatures":
            augmented = {}
            for signature_id, signature in signatures.items():
                signature = copy.deepcopy(signature)
                signature["signature_id"] = f"qian___{signature_id}"
                augmented[signature["signature_id"]] = signature
            _write(path, augmented)
        else:
            path.write_bytes(files[role].read_bytes())
        augmented_files[role] = path
    augmented_files["pairs"] = source_root / "augmented" / "test_pairs.json"
    _write(
        augmented_files["pairs"],
        [
            {
                "signature_id_1": f"qian___{row['signature_id_1']}",
                "signature_id_2": f"qian___{row['signature_id_2']}",
                "label": row["label"],
            }
            for row in rows
        ],
    )
    arrow_root = tmp_path / "arrow"
    for name in ("qian", "augmented"):
        convert_release_arrow(source_root, name, arrow_root, index_root)
    arrow_manifest = json.loads((arrow_root / "manifest.json").read_text(encoding="utf-8"))
    arrow_manifest["dataset_manifests"][comparison.PATENT_DATASET] = arrow_manifest["dataset_manifests"]["qian"]
    _write(arrow_root / "manifest.json", arrow_manifest)
    inputs_path = tmp_path / "inputs.json"
    specs = {
        name: {role: {"path": str(path), "sha256": sha256_file(path)} for role, path in source.items()}
        for name, source in (("qian", files), ("augmented", augmented_files))
    }
    specs[comparison.PATENT_DATASET] = copy.deepcopy(specs["qian"])
    for spec in specs[comparison.PATENT_DATASET].values():
        spec["path"] = str(tmp_path / "patent_files_must_not_open.json")
    _write(inputs_path, {"release_version": "9.9", "sources": {"model": {}, "evaluation": specs}})
    model = tmp_path / "model"
    write_synthetic_pairwise_bundle(
        model,
        artifact_hashes=authority.hashes,
        release_version="9.9",
        eps_calibration=PENDING_EPS_CALIBRATION,
    )
    return argparse.Namespace(
        model_with_patents=model,
        model_without_patents=model,
        source_inputs=inputs_path,
        arrow_root=arrow_root,
        index_root=index_root,
        output_dir=tmp_path / "comparison",
        n_jobs=1,
        total_ram_bytes=1_000_000_000,
        bootstrap_resamples=20,
        bootstrap_seed=13,
        files=files,
        authority=authority,
    )


def test_real_same_model_has_zero_deltas_and_one_feature_pass(
    comparison_fixture: argparse.Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = comparison_fixture
    before = {path: path.read_bytes() for path in fixture.source_inputs.parent.rglob("*") if path.is_file()}
    calls = []
    featurize = comparison.featurize_pairs_with_rust
    score_report = comparison._score_report

    def record_features(*args: Any, **kwargs: Any) -> Any:
        calls.append(len(args[1]))
        return featurize(*args, **kwargs)

    def check_point_outputs(*args: Any, **kwargs: Any) -> Any:
        if kwargs["resamples"]:
            assert (fixture.output_dir / "point_metrics.json").is_file()
            assert (fixture.output_dir / "point_metrics.md").is_file()
            assert len(list(fixture.output_dir.glob("pairs_*.parquet"))) == 2
        return score_report(*args, **kwargs)

    monkeypatch.setattr(comparison, "featurize_pairs_with_rust", record_features)
    monkeypatch.setattr(comparison, "_score_report", check_point_outputs)
    report = comparison.compare(fixture)

    assert calls == [30, 30]
    assert set(report["datasets"]) == {"qian", "augmented"}
    assert report["aggregate"]["rows"] == 60
    assert report["aggregate"]["paired_block_bootstrap"]["units"] == 9
    assert report["aggregate"]["paired_block_bootstrap"]["valid_resamples"] == 20
    for population in [report["aggregate"], *report["datasets"].values()]:
        for head in comparison.HEADS:
            scores = population["heads"][head]
            assert scores["with_patents"] == scores["without_patents"]
            assert set(scores["delta_with_minus_without"].values()) == {0.0}
            assert set(map(tuple, population["paired_block_bootstrap"]["delta_percentile_95"][head].values())) == {
                (0.0, 0.0)
            }
    for name, dataset_report in report["datasets"].items():
        assert dataset_report["unknown_pairs_removed"] == 1
        assert dataset_report["cross_block_pairs"] == 1
        audit = pq.read_table(fixture.output_dir / dataset_report["pair_scores_file"]).to_pydict()
        assert set(audit["dataset"]) == {name}
        assert len(audit["label"]) == 30
        assert len(set(zip(audit["signature_id_1"], audit["signature_id_2"], strict=True))) == 30
        assert all("signature-0-3" not in value for value in audit["signature_id_1"] + audit["signature_id_2"])
        assert audit["with_patents_production"] == audit["without_patents_production"]
        assert np.array_equal(
            audit["with_patents_production"],
            (np.asarray(audit["with_patents_main"]) + audit["with_patents_nameless"]) / 2,
        )
    assert json.loads((fixture.output_dir / "comparison.json").read_text(encoding="utf-8")) == report
    point_metrics = json.loads((fixture.output_dir / "point_metrics.json").read_text(encoding="utf-8"))
    assert point_metrics["aggregate"]["heads"] == report["aggregate"]["heads"]
    assert point_metrics["aggregate"]["paired_block_bootstrap"]["resamples"] == 0
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("role", ["pairs", "unknown_signatures"])
def test_changed_pair_authority_fails_before_features(
    comparison_fixture: argparse.Namespace, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    fixture = comparison_fixture
    fixture.files[role].write_bytes(fixture.files[role].read_bytes() + b" ")

    def fail_features(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Features must not use changed test authority")

    monkeypatch.setattr(comparison, "featurize_pairs_with_rust", fail_features)
    with pytest.raises(ValueError, match="changed after release preparation"):
        comparison.compare(fixture)
    assert not fixture.output_dir.exists()


def test_calibrated_staging_model_is_supported(comparison_fixture: argparse.Namespace, tmp_path: Path) -> None:
    path = tmp_path / "calibrated"
    write_synthetic_pairwise_bundle(
        path,
        artifact_hashes=comparison_fixture.authority.hashes,
        release_version="9.9",
        eps_calibration=CALIBRATED_EPS_CALIBRATION,
    )
    model = comparison._load_model(path, comparison_fixture.authority.hashes, 2)
    assert model.n_jobs == 2
    assert model.classifier.n_jobs == 2


def test_feature_labels_cannot_drift_from_test_pairs(
    comparison_fixture: argparse.Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    featurize = comparison.featurize_pairs_with_rust

    def change_label(*args: Any, **kwargs: Any) -> Any:
        features, labels, nameless = featurize(*args, **kwargs)
        labels[0] = 1 - labels[0]
        return features, labels, nameless

    monkeypatch.setattr(comparison, "featurize_pairs_with_rust", change_label)
    with pytest.raises(ValueError, match="ordered test pair labels"):
        comparison.compare(comparison_fixture)
    assert not comparison_fixture.output_dir.exists()


def test_feature_contract_rejects_different_ordered_columns() -> None:
    left = SimpleNamespace(
        featurizer_info=FeaturizationInfo(["name_similarity"]),
        nameless_featurizer_info=FeaturizationInfo(["year_diff"]),
    )
    right = SimpleNamespace(
        featurizer_info=FeaturizationInfo(["year_diff"]), nameless_featurizer_info=left.nameless_featurizer_info
    )
    with pytest.raises(ValueError, match="different ordered main feature"):
        comparison._feature_contract(left, right)


def test_metric_contract_and_signed_delta() -> None:
    labels = np.asarray([0, 0, 1, 1])
    lower_scores = np.asarray([0.5, 0.6, 0.5, 0.9])
    metrics = comparison._metrics(labels, lower_scores)
    assert metrics["auroc"] == 0.625
    assert metrics["average_precision"] == 0.75
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.5
    report = comparison._score_report(
        labels,
        dict.fromkeys(comparison.HEADS, np.asarray([0.1, 0.2, 0.8, 0.9])),
        dict.fromkeys(comparison.HEADS, lower_scores),
        [("qian", "a"), ("qian", "a"), ("qian", "b"), ("qian", "b")],
        resamples=40,
        seed=13,
    )
    assert report["heads"]["production"]["delta_with_minus_without"]["auroc"] == 0.375
    assert report["heads"]["production"]["delta_with_minus_without"]["f1"] == 0.5
    assert 0 < report["paired_block_bootstrap"]["valid_resamples"] < 40


def test_cross_block_components_and_augmented_copies_share_units() -> None:
    a, b, c, d = [("qian", key) for key in "abcd"]
    edges = [(b, c), (a, b), (d, d), (c, a)]
    assert comparison._block_components(edges) == [a, a, d, a]
    assert comparison._block_components(list(reversed(edges))) == [a, d, a, a]
    assert comparison._source_block("augmented", "qian___s1", "altered", {"qian": {"s1": "a"}}) == a
    with pytest.raises(ValueError, match="augmented patent"):
        comparison._source_block("augmented", "inventors_s2and___s1", "a", {})


def test_cli_has_bounded_defaults() -> None:
    argv = [
        value
        for flag in (
            "model-with-patents",
            "model-without-patents",
            "source-inputs",
            "arrow-root",
            "index-root",
            "output-dir",
        )
        for value in (f"--{flag}", "unused")
    ]
    args = comparison.build_parser().parse_args(argv)
    assert args.bootstrap_resamples == 500
    assert args.bootstrap_seed == 13
