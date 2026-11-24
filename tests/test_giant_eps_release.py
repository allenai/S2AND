from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from s2and._sha256 import sha256_file
from s2and.arrow_inputs import ArrowDataset
from s2and.metrics import b3_precision_recall_fscore
from s2and.production_bundle import finalize_production_bundle
from s2and.production_model import canonical_artifact_hashes, load_production_model
from s2and.rust_calls import get_constraint_labels_index_arrays_rust
from scripts.common.block_roles import load_block_roles
from scripts.common.initial_query_sources import MAP_SHA256_FIELD, derived_query_id
from scripts.eps import apply_reviewed_eps as apply_reviewed_eps_module
from scripts.eps import orcid_b3, plot_b3_eps_sweeps, sweep_eps_on_linking_gold
from scripts.eps.apply_reviewed_eps import apply_reviewed_eps
from scripts.eps.sweep_curves import (
    CLUSTERING_SOURCE,
    DISTANCE_POLICY_VERSION,
    RELEASE_ROUTE,
    arrow_dataset_dir,
    read_sweep_curve,
    write_json,
)
from scripts.intake.assign_block_roles import BlockSpec, assign_block_roles
from tests.helpers import write_arrow_tables_bundle, write_test_arrow_artifact_manifest
from tests.promoted_linking_helpers import write_synthetic_pairwise_bundle
from tests.raw_arrow_helpers import base_arrow_paths, write_ipc
from tests.test_production_model import _write_synthetic_linker


def _source_snapshot(root: Path, *, source: str = "q1", with_clone: bool = False) -> Path:
    dataset = root / "datasets" / "unit"
    ids, papers = [source, "other"], ["source-paper", "other-paper"]
    if with_clone:
        ids.append(derived_query_id("visibleq:", source, "source-paper"))
        papers.append(derived_query_id("visiblep:", source, "source-paper"))
    write_ipc(dataset / "signatures.arrow", pa.table({"signature_id": ids, "paper_id": papers}))
    write_json(dataset / "manifest.json", {"paths": {"signatures": "signatures.arrow"}})
    path = root / "incremental_patent_tasks" / "initial_query_source_map.jsonl"
    if with_clone:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"dataset": "unit", "source_query_signature_id": source, "initial_query_signature_id": ids[-1]})
            + "\n",
            encoding="utf-8",
        )
    return path


def _write_sweep_assignments(
    root: Path, calibration_datasets: tuple[str, ...] = ("unit",), *, test_signature: str = "q1"
) -> None:
    assignments = root / "splits" / "assignments.csv"
    assignments.parent.mkdir(parents=True, exist_ok=True)
    extra_datasets = (*calibration_datasets, "held_out")
    rows = [
        (dataset, "fit" if index == 0 else "check", f"{dataset}_eval")
        for index, dataset in enumerate(calibration_datasets)
    ]
    rows.extend([("s2and", "fit", "s2and_eval"), ("held_out", "test", "held_out_eval"), ("unit", "test", "unit_eval")])
    frame = pd.DataFrame(rows, columns=pd.Index(["dataset", "split", "source_key"]))
    frame["query_group_id"] = frame["dataset"] + ":" + frame["split"]
    frame["base_group_id"] = frame["query_group_id"]
    frame.to_csv(assignments, index=False)
    candidates = frame.drop(columns=["source_key", "split"]).assign(
        query_signature_id="calibration",
        candidate_component_key="candidate",
        retrieval_rank=1,
        label=1,
    )
    candidates.loc[frame["split"].eq("test"), "query_signature_id"] = test_signature
    for dataset, dataset_rows in candidates.groupby("dataset"):
        dataset_rows.to_parquet(root / f"{dataset}.parquet", index=False)
    candidates.iloc[:1].assign(query_group_id="unassigned").to_parquet(root / "calibration.parquet", index=False)
    write_json(
        root / "bundle.json",
        {
            "bundle_name": "snapshot",
            "assets": {},
            "expected_metrics": {},
            "models": {
                "classic": {
                    "classic_gate_source_path": "calibration.parquet",
                    "s2and_eval_path": "s2and.parquet",
                    "hwang_eval_path": "hwang.parquet",
                    "extra_eval_paths": {dataset: f"{dataset}.parquet" for dataset in extra_datasets},
                    "promoted_stratified_gate": {"calibration_splits": ["fit", "check"]},
                    "stratified_eval_test_split": {"assignments_path": "splits/assignments.csv", "test_split": "test"},
                }
            },
        },
    )


def _curve(
    root: Path,
    model_hash: str = "a" * 64,
    limited: bool = False,
    *,
    with_source_map: bool = False,
    test_signature: str = "q1",
) -> Path:
    dataset = root / "unit"
    dataset.mkdir(parents=True)
    reference = dataset / "orcid_reference.parquet"
    pd.DataFrame({"signature_id": ["r1", "r2"], "orcid": ["o", "o"], "repeated": [True, True]}).to_parquet(
        reference, index=False
    )
    snapshot_root = root / "snapshot"
    map_path = _source_snapshot(snapshot_root, source=test_signature, with_clone=with_source_map)
    test_query = derived_query_id("visibleq:", test_signature, "source-paper") if with_source_map else test_signature
    _write_sweep_assignments(snapshot_root, test_signature=test_query)
    map_digest = sha256_file(map_path) if with_source_map else None
    plan = {
        "dataset": "unit",
        "model_manifest_sha256": model_hash,
        "score_mode": "orcid-b3",
        "limited_run": limited,
        "clustering_source": CLUSTERING_SOURCE,
        "route": RELEASE_ROUTE,
        "distance_policy_version": DISTANCE_POLICY_VERSION,
        "batching_threshold": 15000,
        "eps_values": [0.55, 0.65],
        "use_orcid_subblocking": False,
        "suppress_orcid_constraints": True,
        "arrow_root": str(snapshot_root),
        MAP_SHA256_FIELD: map_digest,
        "reference": {
            "reference_policy": orcid_b3.REFERENCE_POLICY,
            "sha256": sha256_file(reference),
            MAP_SHA256_FIELD: map_digest,
        },
    }
    write_json(dataset / "plan.json", plan)
    diagnostic = dataset / "eps_b3_with_singletons.tsv"
    pd.DataFrame(
        {"eps": [0.55, 0.65], "precision": [0.9, 0.7], "recall": [0.8, 1], "f1": [0.85, 0.82], "reference_coverage": 1}
    ).to_csv(diagnostic, sep="\t", index=False)
    scores = dataset / "eps_b3.tsv"
    pd.DataFrame(
        {
            "eps": [0.55, 0.65],
            "precision": [1, 0.8],
            "recall": [0.8, 1],
            "f1": [8 / 9, 8 / 9],
            "reference_coverage": [1, 1],
        }
    ).to_csv(scores, sep="\t", index=False)
    write_json(
        dataset / "summary.json",
        {
            "plan_sha256": sha256_file(dataset / "plan.json"),
            "score_sha256": sha256_file(scores),
            "singleton_inclusive_score_sha256": sha256_file(diagnostic),
        },
    )
    return dataset


def _apply(tmp_path: Path, source: Path, *, datasets: tuple[str, ...] = ("unit",)):
    return apply_reviewed_eps(
        source,
        tmp_path / "output",
        tmp_path / "curves",
        list(datasets),
        0.65,
        "Fixture review",
        tmp_path / "review.json",
        expected_batching_threshold=15000,
    )


def test_orcid_b3_matches_standard_metric_and_retains_boundary_splits() -> None:
    reference = {"a": "one", "b": "one", "c": "two", "d": "two"}
    assignments = {"a": "left", "b": "right", "c": "left", "d": "left", "unknown": "left"}
    result = orcid_b3.score_assignments(reference, assignments, 0.65)
    expected = b3_precision_recall_fscore(
        {"one": ["a", "b"], "two": ["c", "d"]}, {"left": ["a", "c", "d"], "right": ["b"]}
    )[3]
    precision = np.mean([row[0] for row in expected.values()])
    recall = np.mean([row[1] for row in expected.values()])
    assert result["precision"] == pytest.approx(precision)
    assert result["recall"] == pytest.approx(recall)
    assert result["f1"] == pytest.approx(2 * precision * recall / (precision + recall))
    assert result["reference_coverage"] == 1


def test_orcid_reference_normalizes_ids_and_excludes_copies(tmp_path: Path) -> None:
    raw = base_arrow_paths(tmp_path / "raw")
    with pa.OSFile(raw["signatures"], "rb") as source:
        table = pa.ipc.open_file(source).read_all()
    rows = table.to_pylist()
    clone = derived_query_id("visibleq:", "q1", "p_q")
    clone_paper = derived_query_id("visiblep:", "q1", "p_q")
    valid = "0000-0002-1694-233X"
    rows[0]["author_orcid"] = "https://orcid.org/" + valid
    rows[1]["author_orcid"] = "000000021694233X"
    rows[2]["author_orcid"] = "https://orcid.org/0000-0002-1825-0097"
    rows.append({**rows[2], "signature_id": "bad_copy"})
    rows.append({**rows[0], "signature_id": clone, "paper_id": clone_paper})
    rows.append({**rows[0], "signature_id": "lonely", "author_orcid": "0000-0001-5000-0007"})
    paths = write_arrow_tables_bundle(tmp_path / "raw", {"signatures": pa.Table.from_pylist(rows, schema=table.schema)})
    write_test_arrow_artifact_manifest(tmp_path / "raw", {**raw, **paths})
    map_path = tmp_path / "incremental_patent_tasks" / "initial_query_source_map.jsonl"
    map_path.parent.mkdir()
    map_path.write_text(
        json.dumps({"dataset": "unit", "source_query_signature_id": "q1", "initial_query_signature_id": clone}) + "\n",
        encoding="utf-8",
    )
    other = "0000-0001-5000-0007"
    with ArrowDataset.open(Path(paths["signatures"]).parent) as dataset:
        reference = orcid_b3.read_reference(
            dataset, tmp_path, "unit", test_identities=frozenset(), error_records=frozenset()
        )
        assert reference.repeated == {"q1": valid, "s1": valid}
        assert reference.with_singletons == {"q1": valid, "s1": valid, "lonely": other}
        assert reference.excluded_copies == {clone}
        assert reference.stats[MAP_SHA256_FIELD] == sha256_file(map_path)
        assert reference.stats["excluded_mixed_orcid_rows"] == 2
        assert reference.stats["reference_policy"] == orcid_b3.REFERENCE_POLICY
        # A final-test record removes its complete ORCID group, not only its own row.
        separated = orcid_b3.read_reference(
            dataset, tmp_path, "unit", test_identities=frozenset({"lonely"}), error_records=frozenset()
        )
        assert separated.with_singletons == {"q1": valid, "s1": valid}
        assert separated.stats["excluded_final_test_orcid_groups"] == 1
        assert separated.stats["excluded_final_test_group_signatures"] == 1
        with pytest.raises(ValueError, match="No repeated ORCID"):
            orcid_b3.read_reference(
                dataset, tmp_path, "unit", test_identities=frozenset({"q1"}), error_records=frozenset()
            )
        # A reviewed wrong ORCID removes only that record.
        corrected = orcid_b3.read_reference(
            dataset, tmp_path, "unit", test_identities=frozenset(), error_records=frozenset({"lonely"})
        )
        assert corrected.with_singletons == {"q1": valid, "s1": valid}
        assert corrected.stats["excluded_reviewed_error_records"] == 1
        map_path.write_text(
            json.dumps({"dataset": "unit", "source_query_signature_id": "s1", "initial_query_signature_id": clone})
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Derived query IDs differ"):
            orcid_b3.read_reference(dataset, tmp_path, "unit", test_identities=frozenset(), error_records=frozenset())


def test_final_test_identities_use_label_dataset_when_assignment_is_stale(tmp_path: Path) -> None:
    _source_snapshot(tmp_path, source="origin", with_clone=True)
    _write_sweep_assignments(tmp_path, test_signature=derived_query_id("visibleq:", "origin", "source-paper"))
    assignments_path = tmp_path / "splits" / "assignments.csv"
    assignments = pd.read_csv(assignments_path)
    assignments.loc[assignments["query_group_id"].eq("unit:test"), "dataset"] = "stale_dataset"
    assignments.to_csv(assignments_path, index=False)

    assert orcid_b3.final_test_query_identities(tmp_path, "unit") == frozenset({"origin"})


def test_final_test_identities_use_promoted_gate_test_split(tmp_path: Path) -> None:
    _source_snapshot(tmp_path)
    _write_sweep_assignments(tmp_path)
    bundle_path = tmp_path / "bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["models"]["classic"]["promoted_stratified_gate"]["test_split"] = "final"
    write_json(bundle_path, bundle)
    assignments_path = tmp_path / "splits" / "assignments.csv"
    assignments = pd.read_csv(assignments_path)
    final = assignments.loc[assignments["query_group_id"].eq("unit:test")].assign(
        split="final", query_group_id="unit:final", base_group_id="unit:final"
    )
    pd.concat([assignments, final]).to_csv(assignments_path, index=False)
    labels_path = tmp_path / "unit.parquet"
    rows = pd.read_parquet(labels_path)
    final_rows = rows.loc[rows["query_group_id"].eq("unit:test")].assign(
        query_group_id="unit:final", base_group_id="unit:final", query_signature_id="held-record"
    )
    pd.concat([rows, final_rows]).to_parquet(labels_path, index=False)

    assert orcid_b3.final_test_query_identities(tmp_path, "unit") == frozenset({"held-record"})


def test_final_test_identities_reject_groups_only_in_calibration_source(tmp_path: Path) -> None:
    _source_snapshot(tmp_path)
    _write_sweep_assignments(tmp_path)
    assignments_path = tmp_path / "splits" / "assignments.csv"
    assignments = pd.read_csv(assignments_path)
    assignments.loc[assignments["query_group_id"].eq("unit:test"), "source_key"] = "s2and_eval"
    assignments.to_csv(assignments_path, index=False)
    rows = pd.read_parquet(tmp_path / "unit.parquet")
    rows.loc[rows["query_group_id"].eq("unit:test")].to_parquet(tmp_path / "calibration.parquet", index=False)

    with pytest.raises(ValueError, match="did not all match source rows.*unit:test"):
        orcid_b3.final_test_query_identities(tmp_path, "unit")


def test_final_test_identities_resolve_copies_and_reject_unmapped_copies() -> None:
    rows = pd.DataFrame(
        {
            "dataset": ["unit", "unit", "unit", "other"],
            "query_group_id": ["g1", "g1", "g2", "g3"],
            "query_signature_id": ["s1", "s1", "visibleq:c", "s9"],
        }
    )
    assert orcid_b3.resolve_test_query_identities(rows, {"visibleq:c": "s2"}, "unit") == {"s1", "s2"}
    with pytest.raises(ValueError, match="no source record"):
        orcid_b3.resolve_test_query_identities(rows, {}, "unit")


def test_native_eps_featurizer_hides_orcid_ids(tmp_path: Path) -> None:
    raw = base_arrow_paths(tmp_path / "raw")
    with pa.OSFile(raw["signatures"], "rb") as source:
        table = pa.ipc.open_file(source).read_all()
    rows = table.to_pylist()
    for row in rows[:2]:
        row["author_orcid"] = "0000-0002-1694-233X"
    changed = write_arrow_tables_bundle(
        tmp_path / "raw", {"signatures": pa.Table.from_pylist(rows, schema=table.schema)}
    )
    paths = {**raw, **changed}
    root = Path(changed["signatures"]).parent
    write_test_arrow_artifact_manifest(root, paths)
    with ArrowDataset.open(root) as dataset:
        labels = []
        for suppress in (False, True):
            featurizer = sweep_eps_on_linking_gold._build_arrow_featurizer(
                SimpleNamespace(n_jobs=1, suppress_orcid=suppress), dataset, ["q1", "s1"]
            )
            labels.append(
                get_constraint_labels_index_arrays_rust(
                    np.array([0], dtype=np.uint32),
                    np.array([1], dtype=np.uint32),
                    featurizer=featurizer,
                    suppress_orcid=False,
                )[0]
            )
    assert np.isfinite(labels[0])
    assert np.isnan(labels[1])


def test_completed_curve_rejects_stale_or_incomplete_files(tmp_path: Path) -> None:
    dataset = _curve(tmp_path)
    plan = dataset / "plan.json"
    plan_bytes = plan.read_bytes()
    plan.write_bytes(plan_bytes + b"\n")
    with pytest.raises(ValueError, match="plan"):
        read_sweep_curve(tmp_path, "unit")
    plan.write_bytes(plan_bytes)

    scores = dataset / "eps_b3.tsv"
    pd.read_csv(scores, sep="\t").iloc[:1].to_csv(scores, sep="\t", index=False)
    with pytest.raises(ValueError, match="scores change"):
        read_sweep_curve(tmp_path, "unit")

    summary = dataset / "summary.json"
    completed = json.loads(summary.read_bytes())
    write_json(summary, completed | {"score_sha256": sha256_file(scores)})
    with pytest.raises(ValueError, match="Incomplete EPS"):
        plot_b3_eps_sweeps.plot(tmp_path, ["unit"], tmp_path / "plot.png")


def test_review_rejects_proxy_or_bounded_curves(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    dataset = _curve(tmp_path / "curves", sha256_file(manifest), limited=True)
    with pytest.raises(ValueError, match="bounded smoke"):
        _apply(tmp_path, source)

    plan = json.loads((dataset / "plan.json").read_bytes())
    _update_curve_plan(dataset, {**plan, "score_mode": "safe-action-proxy"})
    with pytest.raises(ValueError, match="must use ORCID B3"):
        _apply(tmp_path, source)


def _update_curve_plan(dataset: Path, plan: dict) -> None:
    plan_path = dataset / "plan.json"
    write_json(plan_path, plan)
    summary_path = dataset / "summary.json"
    summary = json.loads(summary_path.read_bytes())
    write_json(summary_path, summary | {"plan_sha256": sha256_file(plan_path)})


def test_review_requires_a_production_sweep_with_current_policies(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    dataset = _curve(tmp_path / "curves", sha256_file(manifest))
    plan = json.loads((dataset / "plan.json").read_bytes())

    _update_curve_plan(dataset, {key: value for key, value in plan.items() if key != "clustering_source"})
    with pytest.raises(ValueError, match="production block clustering"):
        _apply(tmp_path, source)
    _update_curve_plan(dataset, plan | {"route": "bulk"})
    with pytest.raises(ValueError, match="attachment passes"):
        _apply(tmp_path, source)
    _update_curve_plan(dataset, plan | {"distance_policy_version": 11})
    with pytest.raises(ValueError, match="another distance policy"):
        _apply(tmp_path, source)
    _update_curve_plan(dataset, plan | {"batching_threshold": 5000})
    with pytest.raises(ValueError, match="batching_threshold=5000, expected 15000"):
        _apply(tmp_path, source)
    _update_curve_plan(dataset, {key: value for key, value in plan.items() if key != "batching_threshold"})
    with pytest.raises(ValueError, match="batching_threshold=None, expected 15000"):
        _apply(tmp_path, source)
    _update_curve_plan(dataset, plan | {"reference": plan["reference"] | {"reference_policy": "old"}})
    with pytest.raises(ValueError, match="does not exclude final-test"):
        _apply(tmp_path, source)


def test_review_requires_frozen_test_separated_curves(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    dataset = _curve(tmp_path / "curves", sha256_file(manifest), with_source_map=True, test_signature="r2")

    with pytest.raises(ValueError, match="must match the snapshot's calibration assignments"):
        _apply(tmp_path, source, datasets=("unit", "unit"))
    with pytest.raises(ValueError, match="contains final-test records"):
        _apply(tmp_path, source)

    _write_sweep_assignments(tmp_path / "curves" / "snapshot")
    diagnostic = dataset / "eps_b3_with_singletons.tsv"
    diagnostic.write_bytes(diagnostic.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="singleton-inclusive EPS scores change"):
        _apply(tmp_path, source)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "threshold_args", [[], ["--expected-batching-threshold", "0"], ["--expected-batching-threshold", "-1"]]
)
def test_snapshot_sweep_review_requires_a_positive_batching_threshold(
    tmp_path: Path, threshold_args: list[str]
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="positive --expected-batching-threshold"):
        apply_reviewed_eps_module.main(
            [
                "--source-model",
                str(source),
                "--output-model",
                str(tmp_path / "output"),
                "--sweep-root",
                str(tmp_path / "curves"),
                "--datasets",
                "unit",
                "--eps",
                "0.65",
                "--review-note",
                "Fixture review",
                "--output-report",
                str(tmp_path / "review.json"),
                *threshold_args,
            ]
        )
    assert not (tmp_path / "output").exists()


def test_release_review_rejects_a_sweep_missing_a_snapshot_calibration_dataset(tmp_path: Path) -> None:
    source = _complete_model(tmp_path)
    root = tmp_path / "curves"
    _curve(root, sha256_file(source / "manifest.json"))
    _write_sweep_assignments(root / "snapshot", ("unit", "other"))

    with pytest.raises(ValueError, match="calibration assignments.*other"):
        _apply(tmp_path, source)

    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "review.json").exists()


def test_release_review_rejects_changed_source_map_before_test_resolution(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    _curve(tmp_path / "curves", sha256_file(source / "manifest.json"), with_source_map=True)
    map_path = tmp_path / "curves/snapshot/incremental_patent_tasks/initial_query_source_map.jsonl"
    map_path.write_text(" " + map_path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        apply_reviewed_eps(
            source,
            tmp_path / "output",
            tmp_path / "curves",
            ["unit"],
            0.65,
            "Fixture review",
            tmp_path / "review.json",
            expected_batching_threshold=15000,
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("with_source_map", [False, True])
def test_eps_sweep_rejects_an_unmatched_expected_source_map(tmp_path: Path, with_source_map: bool) -> None:
    _source_snapshot(tmp_path, with_clone=with_source_map)
    args = [
        "--output-dir",
        str(tmp_path / "output"),
        "--expected-initial-query-map-sha256",
        "0" * 64,
    ]
    with pytest.raises(ValueError, match="SHA-256 differs"):
        parsed = sweep_eps_on_linking_gold.parse_args(
            [
                "--dataset",
                "unit",
                "--arrow-root",
                str(tmp_path),
                "--model-path",
                str(tmp_path / "model"),
                "--batching-threshold",
                "15000",
                *args,
            ]
        )
        sweep_eps_on_linking_gold.run(parsed)
    assert not (tmp_path / "output").exists()


def _complete_model(tmp_path: Path) -> Path:
    hashes = canonical_artifact_hashes()
    pairwise = tmp_path / "pairwise"
    write_synthetic_pairwise_bundle(pairwise, artifact_hashes=hashes, release_version="9.9")
    linker = tmp_path / "linker"
    target = _write_synthetic_linker(pairwise, linker)
    return finalize_production_bundle(
        pairwise_bundle_dir=pairwise,
        output_bundle_dir=tmp_path / "source",
        incremental_linker_artifact_dir=linker,
        target_json=target,
    ).bundle_dir


def test_internal_review_rejects_a_batching_threshold(tmp_path: Path) -> None:
    source = _complete_model(tmp_path)
    output = tmp_path / "output"
    report = tmp_path / "review.json"

    with pytest.raises(ValueError, match="--expected-batching-threshold requires --sweep-root"):
        apply_reviewed_eps_module.main(
            [
                "--source-model",
                str(source),
                "--output-model",
                str(output),
                "--eps",
                "0.65",
                "--review-note",
                "Fixture review",
                "--output-report",
                str(report),
                "--expected-batching-threshold",
                "15000",
            ]
        )

    assert not output.exists()
    assert not report.exists()


def test_release_review_rejects_a_completed_curve_copied_under_another_name(tmp_path: Path) -> None:
    source = _complete_model(tmp_path)
    root = tmp_path / "curves"
    dataset = _curve(root, sha256_file(source / "manifest.json"))
    shutil.copytree(dataset, root / "other")
    _write_sweep_assignments(root / "snapshot", ("unit", "other"))
    snapshot_datasets = root / "snapshot" / "datasets"
    shutil.copytree(snapshot_datasets / "unit", snapshot_datasets / "other")
    output = tmp_path / "output"
    report = tmp_path / "review.json"

    with pytest.raises(ValueError, match=r"for 'other' records dataset='unit'"):
        apply_reviewed_eps(
            source,
            output,
            root,
            ["unit", "other"],
            0.65,
            "Fixture review",
            report,
            expected_batching_threshold=15000,
            test_identities=lambda _root, _dataset, _sources: frozenset(),
        )

    assert not output.exists()
    assert not report.exists()


def test_release_review_rejects_curves_from_different_snapshots(tmp_path: Path) -> None:
    source = _complete_model(tmp_path)
    root = tmp_path / "curves"
    dataset = _curve(root, sha256_file(source / "manifest.json"))
    snapshot = root / "snapshot"
    _write_sweep_assignments(snapshot, ("unit", "other"))
    shutil.copytree(snapshot / "datasets" / "unit", snapshot / "datasets" / "other")
    other_snapshot = tmp_path / "other_snapshot"
    shutil.copytree(snapshot, other_snapshot)
    other = root / "other"
    shutil.copytree(dataset, other)
    plan_path, summary_path = other / "plan.json", other / "summary.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    write_json(plan_path, plan | {"dataset": "other", "arrow_root": str(other_snapshot)})
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    write_json(summary_path, summary | {"plan_sha256": sha256_file(plan_path)})

    with pytest.raises(ValueError, match="must use one snapshot"):
        _apply(tmp_path, source, datasets=("unit", "other"))

    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "review.json").exists()


def test_release_review_reports_and_ignores_an_extra_sweep_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    source = _complete_model(tmp_path)
    root = tmp_path / "curves"
    dataset = _curve(root, sha256_file(source / "manifest.json"))
    other = root / "unit_archived"
    shutil.copytree(dataset, other)

    report = _apply(tmp_path, source)

    assert report["sweep_plans"].keys() == {"unit"}
    assert "unit_archived" in capsys.readouterr().err


def test_release_review_rejects_an_assignment_dataset_that_differs_from_eval_rows(tmp_path: Path) -> None:
    source = _complete_model(tmp_path)
    root = tmp_path / "curves"
    _curve(root, sha256_file(source / "manifest.json"))
    snapshot = root / "snapshot"
    _write_sweep_assignments(snapshot, ("unit", "other"))
    assignments_path = snapshot / "splits" / "assignments.csv"
    assignments = pd.read_csv(assignments_path)
    assignments.loc[assignments["source_key"].eq("other_eval"), "dataset"] = "unit"
    assignments.to_csv(assignments_path, index=False)

    with pytest.raises(ValueError, match="Calibration assignment dataset does not match.*other"):
        _apply(tmp_path, source)

    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "review.json").exists()


def test_plot_rejects_a_completed_curve_copied_under_another_name(tmp_path: Path) -> None:
    dataset = _curve(tmp_path)
    shutil.copytree(dataset, tmp_path / "other")

    with pytest.raises(ValueError, match=r"for 'other' records dataset='unit'"):
        plot_b3_eps_sweeps.plot(tmp_path, ["other"], tmp_path / "plot.png")


def _changed_files(source: Path, output: Path) -> set[str]:
    before = {path.relative_to(source).as_posix(): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    after = {path.relative_to(output).as_posix(): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert before.keys() == after.keys()
    return {path for path in before if before[path] != after[path]}


def test_eps_only_update_accepts_an_explicit_dataset_list_and_preserves_artifacts(tmp_path: Path) -> None:
    source = _complete_model(tmp_path)
    _curve(tmp_path / "curves", sha256_file(source / "manifest.json"), with_source_map=True)
    output = tmp_path / "output"
    report = _apply(tmp_path, source)
    assert report["sweep_plans"]["unit"]["reference"]["reference_policy"] == orcid_b3.REFERENCE_POLICY
    assert _changed_files(source, output) == {"clusterer.json", "manifest.json"}
    assert load_production_model(output).cluster_model.eps == 0.65
    assert load_production_model(source).cluster_model.eps == 0.5


def test_eps_release_accepts_shared_orcid_only_when_sweep_masks_it(tmp_path: Path) -> None:
    shared = "0000-0002-1694-233X"
    local = "0000-0001-5000-0007"
    role_path = tmp_path / "assignments.json"
    result = assign_block_roles(
        [
            BlockSpec("unit", 3, frozenset({shared, local}), source_split="development"),
            BlockSpec("sealed", 2, frozenset({shared}), source_split="holdout"),
        ],
        {"eps_cal": 1, "final_test_a": 1},
        closure_mode="home_pool",
        enforce_source_splits=True,
    )
    write_json(role_path, result)
    assert result["verification"]["raw_cross_role_orcid_overlap"] == 1
    assert result["verification"]["cross_role_orcid_overlap"] == 0
    block_roles = load_block_roles(role_path)
    source = _complete_model(tmp_path)
    sweep_root = tmp_path / "curves"
    sweep = _curve(sweep_root, sha256_file(source / "manifest.json"))
    plan_path = sweep / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    raw = base_arrow_paths(tmp_path / "raw")
    with pa.OSFile(raw["signatures"], "rb") as handle:
        table = pa.ipc.open_file(handle).read_all()
    table = table.set_column(
        table.schema.get_field_index("author_orcid"), "author_orcid", pa.array([shared, local, local])
    )
    table = table.append_column("author_block", pa.array(["unit"] * 3))
    paths = write_arrow_tables_bundle(tmp_path / "raw", {"signatures": table})
    write_test_arrow_artifact_manifest(tmp_path / "raw", {**raw, **paths})
    with ArrowDataset.open(tmp_path / "raw") as arrow:
        reference = orcid_b3.read_reference(
            arrow,
            Path(plan["arrow_root"]),
            "unit",
            test_identities=frozenset(),
            error_records=frozenset(),
            allow_all_identities=True,
            block_roles=block_roles,
        )
    assert reference.with_singletons == {"s1": local, "s2": local}
    assert reference.stats["excluded_non_home_role_records"] == 1

    reference_path = sweep / "orcid_reference.parquet"
    summary_path = sweep / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    def set_sweep(labels: dict[str, str], *, bind_manifest: bool) -> None:
        pd.DataFrame({"signature_id": list(labels), "orcid": list(labels.values())}).to_parquet(reference_path)
        plan["reference"] = {**reference.stats, "sha256": sha256_file(reference_path)}
        if not bind_manifest:
            del plan["reference"][orcid_b3.BLOCK_ROLE_SHA256_FIELD]
        write_json(plan_path, plan)
        write_json(summary_path, {**summary, "plan_sha256": sha256_file(plan_path)})

    def apply() -> dict:
        return apply_reviewed_eps(
            source,
            tmp_path / "output",
            sweep_root,
            ["unit"],
            0.65,
            "Fixture review",
            tmp_path / "review.json",
            expected_batching_threshold=15000,
            block_role_assignments=role_path,
        )

    set_sweep(reference.with_singletons, bind_manifest=False)
    with pytest.raises(ValueError, match="same frozen block roles"):
        apply()
    set_sweep({**reference.with_singletons, "q1": shared}, bind_manifest=True)
    with pytest.raises(ValueError, match="outside its home role"):
        apply()
    set_sweep(reference.with_singletons, bind_manifest=True)
    assert apply()["sweep_plans"]["unit"]["reference"][orcid_b3.BLOCK_ROLE_SHA256_FIELD] == block_roles.sha256
    assert load_production_model(tmp_path / "output").cluster_model.eps == 0.65


def test_owner_review_can_set_the_close_pair_bonus_and_release_version(tmp_path: Path) -> None:
    source = _complete_model(tmp_path)
    output = tmp_path / "output"
    report = apply_reviewed_eps(
        source,
        output,
        None,
        [],
        0.41,
        "Internal review",
        tmp_path / "review.json",
        close_pair=(8.0, 0.05),
        release_version="9.9.1",
    )
    assert _changed_files(source, output) == {"clusterer.json", "manifest.json"}
    model = load_production_model(output)
    assert (model.cluster_model.eps, model.cluster_model.close_pair_bonus, model.cluster_model.close_pair_distance) == (
        0.41,
        8.0,
        0.05,
    )
    assert report["sweep_plans"] is None
    assert model.production_model_release_version == "9.9.1" == report["release_version"]
    assert load_production_model(source).cluster_model.close_pair_bonus == 0.0


def test_snapshot_sweep_review_cannot_change_the_close_pair_bonus(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="before the sweep"):
        apply_reviewed_eps(
            source,
            tmp_path / "output",
            tmp_path / "curves",
            ["unit"],
            0.41,
            "Review",
            tmp_path / "review.json",
            close_pair=(8.0, 0.05),
        )
    assert not (tmp_path / "output").exists()


def _uneven_curve(tmp_path: Path, source: Path) -> None:
    """A sweep whose F1 is 2/3 at EPS 0.55 and 8/9 at 0.65."""
    dataset = _curve(tmp_path / "curves", sha256_file(source / "manifest.json"))
    pd.DataFrame(
        {"eps": [0.55, 0.65], "precision": [1, 0.8], "recall": [0.5, 1], "f1": [2 / 3, 8 / 9], "reference_coverage": 1}
    ).to_csv(dataset / "eps_b3.tsv", sep="\t", index=False)
    summary = json.loads((dataset / "summary.json").read_text(encoding="utf-8"))
    write_json(dataset / "summary.json", summary | {"score_sha256": sha256_file(dataset / "eps_b3.tsv")})


def test_sweep_review_of_a_bonus_bundle_requires_the_f1_tolerance(tmp_path: Path) -> None:
    # The owner review writes a bundle with the bonus; its sweep review may not skip the F1 check.
    plain = _complete_model(tmp_path)
    bonus = tmp_path / "bonus"
    apply_reviewed_eps(
        plain, bonus, None, [], 0.65, "Internal review", tmp_path / "bonus_review.json", close_pair=(8.0, 0.05)
    )
    _uneven_curve(tmp_path, bonus)

    def review(tolerance: float | None, eps: float = 0.65) -> dict:
        return apply_reviewed_eps(
            bonus,
            tmp_path / "output",
            tmp_path / "curves",
            ["unit"],
            eps,
            "Fixture review",
            tmp_path / "review.json",
            expected_batching_threshold=15000,
            test_identities=lambda _root, _dataset, _sources: frozenset(),
            precision_f1_tolerance=tolerance,
        )

    with pytest.raises(ValueError, match="needs --precision-f1-tolerance"):
        review(None)
    assert not (tmp_path / "output").exists()
    with pytest.raises(ValueError, match="below the best swept F1"):
        review(0.01, eps=0.55)
    assert not (tmp_path / "output").exists()

    record = review(0.01)["giant_f1_tolerance"]
    assert (record["best_eps"], record["tolerance"]) == (0.65, 0.01)
    assert record["selected_f1"] == pytest.approx(8 / 9)


def _second_block(tmp_path: Path, *, eps: list[float], recall: list[float], references: int) -> None:
    """Add block "other" to the fixture sweep, with its own EPS grid, recall and reference count."""
    source, dataset = tmp_path / "curves" / "unit", tmp_path / "curves" / "other"
    shutil.copytree(source, dataset)
    pd.DataFrame({"signature_id": [f"r{i}" for i in range(references)], "orcid": "o", "repeated": True}).to_parquet(
        dataset / "orcid_reference.parquet", index=False
    )
    rows = {"eps": eps, "precision": [1.0] * len(eps), "recall": recall, "f1": [0.5] * len(eps)}
    pd.DataFrame(rows | {"reference_coverage": 1}).to_csv(dataset / "eps_b3.tsv", sep="\t", index=False)
    plan = json.loads((dataset / "plan.json").read_text(encoding="utf-8"))
    plan["dataset"] = "other"
    plan["eps_values"] = eps
    plan["reference"]["sha256"] = sha256_file(dataset / "orcid_reference.parquet")
    write_json(dataset / "plan.json", plan)
    summary = json.loads((dataset / "summary.json").read_text(encoding="utf-8"))
    write_json(
        dataset / "summary.json",
        summary
        | {"plan_sha256": sha256_file(dataset / "plan.json"), "score_sha256": sha256_file(dataset / "eps_b3.tsv")},
    )


def test_pooled_curve_weights_each_scored_reference_equally(tmp_path: Path) -> None:
    _curve(tmp_path / "curves")
    # "unit" has 2 references with recall 0.8 and 1.0; "other" has 6 with recall 0.4 and 0.6.
    _second_block(tmp_path, eps=[0.55, 0.65], recall=[0.4, 0.6], references=6)

    pooled = apply_reviewed_eps_module._pooled_curve(tmp_path / "curves", ["unit", "other"])

    assert pooled["recall"].tolist() == pytest.approx([(2 * 0.8 + 6 * 0.4) / 8, (2 * 1.0 + 6 * 0.6) / 8])
    assert pooled["precision"].tolist() == pytest.approx([1.0, (2 * 0.8 + 6 * 1.0) / 8])

    shutil.rmtree(tmp_path / "curves" / "other")
    _second_block(tmp_path, eps=[0.65, 0.75], recall=[0.4, 0.6], references=6)

    with pytest.raises(ValueError, match="different EPS values"):
        apply_reviewed_eps_module._pooled_curve(tmp_path / "curves", ["unit", "other"])


def test_giant_f1_tolerance_needs_a_snapshot_sweep(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="needs a snapshot sweep"):
        apply_reviewed_eps(
            source,
            tmp_path / "output",
            None,
            [],
            0.53,
            "Review",
            tmp_path / "review.json",
            precision_f1_tolerance=0.001,
        )


def test_review_requires_sweep_root_and_datasets_together(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="together"):
        apply_reviewed_eps(
            source, tmp_path / "output", tmp_path / "curves", [], 0.53, "Review", tmp_path / "review.json"
        )
    assert not (tmp_path / "output").exists()


def test_completed_output_cannot_be_rebound_by_dry_run(tmp_path: Path) -> None:
    dataset = _curve(tmp_path)
    before = (dataset / "plan.json").read_bytes()
    args = sweep_eps_on_linking_gold.parse_args(
        [
            "--dataset",
            "unit",
            "--arrow-root",
            "absent",
            "--model-path",
            "absent",
            "--batching-threshold",
            "15000",
            "--output-dir",
            str(dataset),
            "--dry-run",
        ]
    )
    with pytest.raises(FileExistsError, match="fresh sweep"):
        sweep_eps_on_linking_gold.run(args)
    assert (dataset / "plan.json").read_bytes() == before


def test_eps_resolves_hash_bound_replay_data_in_combined_snapshot(tmp_path: Path) -> None:
    dataset_manifest = tmp_path / "datasets" / "giant" / "manifest.json"
    write_json(dataset_manifest, {})
    replay = tmp_path / "replay_manifest.json"
    benchmark = tmp_path / "benchmarks" / "unit" / "manifest.json"
    write_json(benchmark, {})
    write_json(
        replay,
        {
            "kind": "s2and_arrow_collection",
            "format_version": 1,
            "dataset_manifests": {
                "giant": {"path": "datasets/giant/manifest.json", "sha256": sha256_file(dataset_manifest)}
            },
        },
    )
    write_json(
        tmp_path / "manifest.json",
        {
            "kind": "s2and_public_data",
            "format_version": 1,
            "release_version": "1.3",
            "dataset_manifests": {"giant": {"path": "benchmarks/unit/manifest.json", "sha256": sha256_file(benchmark)}},
            "replay_bundles": {"incremental": {"path": "replay_manifest.json", "sha256": sha256_file(replay)}},
        },
    )
    assert arrow_dataset_dir(tmp_path, "giant", replay_bundle="incremental") == dataset_manifest.parent
    assert arrow_dataset_dir(tmp_path, "giant") == benchmark.parent
    replay.write_bytes(replay.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256|checksum|sha256"):
        arrow_dataset_dir(tmp_path, "giant")


def test_manual_curve_reader_rejects_hash_valid_old_distance_policy(tmp_path: Path) -> None:
    dataset = _curve(tmp_path)
    plan = json.loads((dataset / "plan.json").read_bytes())
    _update_curve_plan(dataset, plan | {"distance_policy_version": DISTANCE_POLICY_VERSION - 1})
    with pytest.raises(ValueError, match="another distance policy"):
        read_sweep_curve(tmp_path, "unit")


@pytest.mark.parametrize("mode", ["strict", "home_pool"])
def test_eps_release_rejects_nonmember_without_shared_orcids(tmp_path: Path, mode: str) -> None:
    from types import SimpleNamespace

    from s2and._sha256 import sha256_file
    from scripts.common.block_roles import load_block_roles
    from scripts.common.initial_query_sources import MAP_SHA256_FIELD
    from scripts.eps.apply_reviewed_eps import _require_test_separated_reference
    from scripts.eps.orcid_b3 import BLOCK_ROLE_SHA256_FIELD, REFERENCE_POLICY_ALL_IDENTITIES
    from scripts.intake.assign_block_roles import BlockSpec, assign_block_roles

    block = "a ross"
    home, sealed = "0000-0001-5000-0007", "0000-0002-1694-233X"
    role_path = tmp_path / "roles.json"
    role_path.write_text(
        json.dumps(
            assign_block_roles(
                [BlockSpec(block, 2, frozenset({home})), BlockSpec("b ross", 2, frozenset({sealed}))],
                {"eps_cal": 1, "final_test_a": 1},
                pinned_roles={block: "eps_cal", "b ross": "final_test_a"},
                closure_mode=mode,
            )
        ),
        encoding="utf-8",
    )
    roles = load_block_roles(role_path)
    assert not roles.orcid_home_role
    ref = tmp_path / block / "orcid_reference.parquet"
    ref.parent.mkdir()
    pd.DataFrame({"signature_id": ["one", "two"], "orcid": [sealed, sealed]}).to_parquet(ref, index=False)
    digest = "a" * 64
    report = {
        "arrow_root": str(tmp_path),
        MAP_SHA256_FIELD: digest,
        "reference": {
            "reference_policy": REFERENCE_POLICY_ALL_IDENTITIES,
            "sha256": sha256_file(ref),
            MAP_SHA256_FIELD: digest,
            BLOCK_ROLE_SHA256_FIELD: roles.sha256,
        },
    }
    with pytest.raises(ValueError, match="ORCID .* absent .* frozen block"):
        _require_test_separated_reference(
            tmp_path,
            block,
            report,
            lambda *_: frozenset(),
            SimpleNamespace(sha256=digest),
            verified_eps_cal_blocks=frozenset({block}),
            block_roles=roles,
        )
