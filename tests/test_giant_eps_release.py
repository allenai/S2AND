from __future__ import annotations

import json
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
from scripts.eps_sweep import orcid_b3, plot_b3_eps_sweeps, sweep_eps_on_linking_gold
from scripts.eps_sweep.apply_reviewed_eps import apply_reviewed_eps
from scripts.eps_sweep.build_linking_eps_gold import _build_dataset_constraints, parse_args
from scripts.eps_sweep.common import arrow_dataset_dir, read_sweep_curve, write_json
from tests.helpers import write_arrow_tables_bundle, write_test_arrow_artifact_manifest
from tests.promoted_linking_helpers import write_synthetic_pairwise_bundle
from tests.raw_arrow_helpers import base_arrow_paths
from tests.test_production_model import _write_synthetic_linker


def _curve(root: Path, model_hash: str = "a" * 64, limited: bool = False, mode: str = "orcid-b3") -> Path:
    dataset = root / "unit"
    dataset.mkdir(parents=True)
    reference = dataset / "orcid_reference.parquet"
    pd.DataFrame({"signature_id": ["r1", "r2"], "orcid": ["o", "o"], "repeated": [True, True]}).to_parquet(
        reference, index=False
    )
    plan = {
        "model_manifest_sha256": model_hash,
        "score_mode": mode,
        "limited_run": limited,
        "eps_values": [0.55, 0.65],
        "use_orcid_subblocking": False,
        "suppress_orcid_constraints": True,
        "arrow_root": str(root / "snapshot"),
        "reference": {"reference_policy": orcid_b3.REFERENCE_POLICY, "sha256": sha256_file(reference)},
    }
    write_json(dataset / "plan.json", plan)
    diagnostic = dataset / "eps_b3_with_singletons.tsv"
    pd.DataFrame(
        {"eps": [0.55, 0.65], "precision": [0.9, 0.7], "recall": [0.8, 1], "f1": [0.85, 0.82], "reference_coverage": 1}
    ).to_csv(diagnostic, sep="\t", index=False)
    if mode == "orcid-b3":
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
    else:
        scores = dataset / "eps_overall.tsv"
        pd.DataFrame(
            {
                "eps": [0.55, 0.65],
                "weight": "weight_query_class_balanced",
                "scope_type": "overall",
                "scope_value": "all",
                "positive_recall": [0.8, 1],
                "negative_separation": [1, 0.8],
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


def _apply(
    tmp_path: Path, source: Path, *, test_ids: frozenset[str] = frozenset(), datasets: tuple[str, ...] = ("unit",)
):
    return apply_reviewed_eps(
        source,
        tmp_path / "output",
        tmp_path / "curves",
        list(datasets),
        0.65,
        "Fixture review",
        tmp_path / "review.json",
        release_datasets=("unit",),
        test_identities=lambda _root, _dataset: test_ids,
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
    valid = "0000-0002-1694-233X"
    rows[0]["author_orcid"] = "https://orcid.org/" + valid
    rows[1]["author_orcid"] = "000000021694233X"
    rows[2]["author_orcid"] = "https://orcid.org/0000-0002-1825-0097"
    rows.append({**rows[2], "signature_id": "bad_copy"})
    rows.append({**rows[0], "signature_id": "visibleq:copy"})
    rows.append({**rows[0], "signature_id": "lonely", "author_orcid": "0000-0001-5000-0007"})
    paths = write_arrow_tables_bundle(tmp_path / "raw", {"signatures": pa.Table.from_pylist(rows, schema=table.schema)})
    write_test_arrow_artifact_manifest(tmp_path / "raw", {**raw, **paths})
    map_path = tmp_path / "incremental_patent_tasks" / "initial_query_source_map.jsonl"
    map_path.parent.mkdir()
    map_path.write_text(
        json.dumps(
            {"dataset": "unit", "source_query_signature_id": "q1", "initial_query_signature_id": "visibleq:copy"}
        )
        + "\n",
        encoding="utf-8",
    )
    other = "0000-0001-5000-0007"
    with ArrowDataset.open(Path(paths["signatures"]).parent) as dataset:
        reference = orcid_b3.read_reference(
            dataset, tmp_path, "unit", test_identities=frozenset(), error_records=frozenset()
        )
        assert reference.repeated == {"q1": valid, "s1": valid}
        assert reference.with_singletons == {"q1": valid, "s1": valid, "lonely": other}
        assert reference.excluded_copies == {"visibleq:copy"}
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


def test_final_test_identities_resolve_copies_and_reject_unmapped_or_missing_rows() -> None:
    assignments = pd.DataFrame({"dataset": ["unit", "unit", "other"], "query_group_id": ["g1", "g2", "g3"]})
    rows = pd.DataFrame(
        {
            "dataset": ["unit", "unit", "unit", "other"],
            "query_group_id": ["g1", "g1", "g2", "g3"],
            "query_signature_id": ["s1", "s1", "visibleq:c", "s9"],
        }
    )
    assert orcid_b3.resolve_test_query_identities(rows, assignments, {"visibleq:c": "s2"}, "unit") == {"s1", "s2"}
    with pytest.raises(ValueError, match="no source record"):
        orcid_b3.resolve_test_query_identities(rows, assignments, {}, "unit")
    with pytest.raises(ValueError, match="no label rows"):
        orcid_b3.resolve_test_query_identities(rows[rows["query_group_id"] != "g2"], assignments, {}, "unit")


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


@pytest.mark.parametrize("field", ["plan", "scores"])
def test_completed_curve_rejects_stale_bytes(tmp_path: Path, field: str) -> None:
    dataset = _curve(tmp_path)
    path = dataset / ("plan.json" if field == "plan" else "eps_b3.tsv")
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="does not match|change after"):
        read_sweep_curve(tmp_path, "unit", "orcid-b3")


def test_plot_rejects_incomplete_grid_even_with_updated_score_hash(tmp_path: Path) -> None:
    dataset = _curve(tmp_path)
    path = dataset / "eps_b3.tsv"
    pd.read_csv(path, sep="\t").iloc[:1].to_csv(path, sep="\t", index=False)
    write_json(
        dataset / "summary.json", {"plan_sha256": sha256_file(dataset / "plan.json"), "score_sha256": sha256_file(path)}
    )
    with pytest.raises(ValueError, match="Incomplete EPS"):
        plot_b3_eps_sweeps.plot(tmp_path, ["unit"], tmp_path / "plot.png")


@pytest.mark.parametrize(
    "mode,limited,message", [("safe-action-proxy", False, "diagnostic"), ("orcid-b3", True, "bounded smoke")]
)
def test_review_rejects_diagnostic_or_bounded_curves(tmp_path: Path, mode: str, limited: bool, message: str) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    _curve(tmp_path / "curves", sha256_file(source / "manifest.json"), limited, mode)
    with pytest.raises(ValueError, match=message):
        _apply(tmp_path, source)


@pytest.mark.parametrize(
    "change", ["partial-datasets", "old-policy", "test-overlap", "stale-diagnostic", "no-diagnostic"]
)
def test_review_requires_frozen_test_separated_curves(tmp_path: Path, change: str) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    dataset = _curve(tmp_path / "curves", sha256_file(source / "manifest.json"))
    test_ids = frozenset()
    datasets: tuple[str, ...] = ("unit",)
    message = {
        "partial-datasets": "exactly the frozen datasets",
        "old-policy": "does not exclude final-test",
        "test-overlap": "contains final-test records",
        "stale-diagnostic": "singleton-inclusive EPS scores change",
        "no-diagnostic": "singleton-inclusive EPS scores change|No such file",
    }[change]
    if change == "partial-datasets":
        datasets = ("unit", "unit")
    elif change == "old-policy":
        plan = json.loads((dataset / "plan.json").read_text(encoding="utf-8"))
        del plan["reference"]["reference_policy"]
        write_json(dataset / "plan.json", plan)
        summary = json.loads((dataset / "summary.json").read_text(encoding="utf-8"))
        summary["plan_sha256"] = sha256_file(dataset / "plan.json")
        write_json(dataset / "summary.json", summary)
    elif change == "test-overlap":
        test_ids = frozenset({"r2"})
    elif change == "stale-diagnostic":
        path = dataset / "eps_b3_with_singletons.tsv"
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        (dataset / "eps_b3_with_singletons.tsv").unlink()
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        _apply(tmp_path, source, test_ids=test_ids, datasets=datasets)
    assert not (tmp_path / "output").exists()


def test_release_review_requires_every_frozen_giant(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    _curve(tmp_path / "curves", sha256_file(source / "manifest.json"))
    with pytest.raises(ValueError, match="exactly the frozen datasets"):
        apply_reviewed_eps(
            source, tmp_path / "output", tmp_path / "curves", ["unit"], 0.65, "Fixture review", tmp_path / "review.json"
        )


def test_eps_only_update_preserves_real_complete_artifacts(tmp_path: Path) -> None:
    hashes = canonical_artifact_hashes()
    pairwise = tmp_path / "pairwise"
    write_synthetic_pairwise_bundle(pairwise, artifact_hashes=hashes, release_version="9.9")
    linker = tmp_path / "linker"
    target = _write_synthetic_linker(pairwise, linker)
    source = finalize_production_bundle(
        pairwise_bundle_dir=pairwise,
        output_bundle_dir=tmp_path / "source",
        incremental_linker_artifact_dir=linker,
        target_json=target,
    ).bundle_dir
    before = {path.relative_to(source).as_posix(): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    _curve(tmp_path / "curves", sha256_file(source / "manifest.json"))
    output = tmp_path / "output"
    report = _apply(tmp_path, source)
    assert report["sweep_plans"]["unit"]["reference"]["reference_policy"] == orcid_b3.REFERENCE_POLICY
    after = {path.relative_to(output).as_posix(): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert before.keys() == after.keys()
    assert {path for path in before if before[path] != after[path]} == {"clusterer.json", "manifest.json"}
    assert load_production_model(output).cluster_model.eps == 0.65
    assert load_production_model(source).cluster_model.eps == 0.5


def test_proxy_builder_excludes_initial_query_origin(tmp_path: Path) -> None:
    components = tmp_path / "components"
    components.mkdir()
    pd.DataFrame(
        {
            "dataset": ["unit", "unit"],
            "candidate_component_key": ["profile", "profile"],
            "member_index": [0, 1],
            "signature_id": ["origin", "other"],
        }
    ).to_parquet(components / "unit_members.parquet", index=False)
    origins = tmp_path / "incremental_patent_tasks" / "initial_query_source_map.jsonl"
    origins.parent.mkdir()
    origins.write_text(
        json.dumps({"dataset": "unit", "source_query_signature_id": "origin", "initial_query_signature_id": "copy"})
        + "\n",
        encoding="utf-8",
    )
    labels = pd.DataFrame(
        [
            {
                "dataset": "unit",
                "table_name": "dev.parquet",
                "split": "calibration_fit",
                "source_key": "dev",
                "source_kind": "manual",
                "supervision_type": "manual",
                "query_group_id": "query",
                "query_signature_id": "copy",
                "query_view": "initial_only",
                "candidate_component_key": "profile",
                "label": 1,
            }
        ]
    )
    pairs, conflicts, summary = _build_dataset_constraints(tmp_path, "unit", labels)
    assert pairs["member_signature_id"].tolist() == ["other"]
    assert conflicts.empty
    assert summary["excluded_initial_origin_rows"] == 1


def test_proxy_builder_cli_rejects_test_assignment() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--bundle-root", "data", "--output-dir", "out", "--datasets", "unit", "--task-splits", "test"])


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
