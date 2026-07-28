from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pyarrow as pa
import pytest

from s2and.incremental_linking.feature_block import write_name_counts_index
from scripts.verification import profile_promoted_incremental_arrow as profile
from tests.helpers import (
    tiny_name_counts_tuple,
    write_minimal_arrow_prediction_bundle,
    write_test_arrow_artifact_manifest,
)


def _args(tmp_path, **workload_overrides):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    dataset_root = tmp_path / "arrow" / "dummy"
    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text("{}\n", encoding="utf-8")
    workload = {
        "dataset": "dummy",
        "target_block": "",
        "query_limit": 25,
        "max_seed_clusters": 25,
        "seed_source": "clusters",
        "runs": 1,
        "n_jobs": 4,
        "batching_threshold": None,
        "total_ram_bytes": None,
        "synthetic_seeds_when_clusters_missing": False,
    }
    workload.update(workload_overrides)
    evaluation_plan = tmp_path / "evaluation_plan.json"
    evaluation_plan.write_text(
        json.dumps(
            {
                "pairwise": {},
                "cluster": {},
                "performance": {
                    "arrow_root": str((tmp_path / "arrow").resolve()),
                    "workload": workload,
                },
                "baselines": {},
                "gates": {},
            }
        ),
        encoding="utf-8",
    )
    return profile.parse_args(
        [
            "--evaluation-plan",
            str(evaluation_plan),
            "--model-path",
            str(model_dir),
            "--write-json",
            str(tmp_path / "report.json"),
        ]
    )


def _stub_prerun(monkeypatch):
    monkeypatch.setattr(profile, "_require_psutil", lambda: object())
    monkeypatch.setattr(profile, "_rust_extension_identity", lambda _required: {})


def _stub_inputs(monkeypatch, tmp_path, block_signatures):
    dataset_root = tmp_path / "arrow" / "dummy"
    paths = write_minimal_arrow_prediction_bundle(dataset_root, include_specter=True)
    name_counts_index, _metrics = write_name_counts_index(dataset_root, tiny_name_counts_tuple())
    paths["name_counts_index"] = name_counts_index
    write_test_arrow_artifact_manifest(dataset_root, paths)
    (dataset_root / "dummy_clusters.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(profile, "read_signature_blocks", lambda _path: {"block": block_signatures})
    monkeypatch.setattr(profile, "_read_signature_to_cluster_id", lambda _path: {"seed": "cluster"})


def test_read_signature_blocks_reads_only_block_membership(tmp_path) -> None:
    path = tmp_path / "signatures.arrow"
    table = pa.table(
        {
            "signature_id": ["s1", "s2", "s3"],
            "author_block": ["a", "b", "a"],
            "unused": [1, 2, 3],
        }
    )
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)

    with path.open("rb") as source_file:
        assert profile.read_signature_blocks(source_file) == {"a": ["s1", "s3"], "b": ["s2"]}


def test_process_tree_rss_monitor_sums_parent_and_children() -> None:
    child_a = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=20))
    child_b = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=30))
    parent = SimpleNamespace(
        children=lambda **_kwargs: [child_a, child_b],
        memory_info=lambda: SimpleNamespace(rss=10),
    )
    monitor = profile.ProcessTreeRSSMonitor(SimpleNamespace(Process=lambda: parent, Error=Exception))

    monitor._sample()

    assert monitor.peak_gb == 60 / (1024**3)


def test_select_profile_workload_uses_largest_block_and_stable_seed_queries() -> None:
    workload = profile.select_profile_workload(
        blocks={"small": ["x"], "large": ["a", "b", "c", "d", "e"]},
        signature_to_cluster_id={
            "a": "cluster-1",
            "b": "cluster-1",
            "c": "cluster-2",
            "d": "cluster-3",
            "e": "cluster-3",
        },
        target_block="",
        query_limit=2,
        max_seed_clusters=2,
    )

    assert workload.target_block == "large"
    assert workload.block_signature_count == 5
    assert workload.seed_signature_to_cluster == {"a": "cluster-1", "c": "cluster-2"}
    assert workload.query_signature_ids == ["b", "d"]
    assert workload.block_signatures == ["a", "c", "b", "d"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("query_limit", 0, "query_limit must be 1"),
        ("query_limit", -1, "query_limit must be >= 0"),
        ("max_seed_clusters", 0, "max_seed_clusters must be 1"),
        ("max_seed_clusters", -1, "max_seed_clusters must be >= 0"),
    ],
)
def test_run_rejects_unbounded_or_negative_workload_without_full_run(
    tmp_path,
    field: str,
    value: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        profile.run(_args(tmp_path, **{field: value}))


def test_main_writes_fresh_report(tmp_path, monkeypatch, capsys) -> None:
    report_path = tmp_path / "report.json"
    payload = {"schema_version": profile.REPORT_SCHEMA, "summary": {"run_count": 1}}
    monkeypatch.setattr(profile, "run", lambda _args: payload)

    exit_code = profile.main(
        [
            "--evaluation-plan",
            "evaluation-plan.json",
            "--model-path",
            "model",
            "--write-json",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(report_path.read_text(encoding="utf-8")) == payload
    assert json.loads(capsys.readouterr().out) == payload


def test_main_rejects_existing_report_before_run(tmp_path, monkeypatch) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(profile, "run", lambda _args: pytest.fail("profile must not run"))

    with pytest.raises(FileExistsError, match="already exists"):
        profile.main(
            [
                "--evaluation-plan",
                "evaluation-plan.json",
                "--model-path",
                "model",
                "--write-json",
                str(report_path),
            ]
        )


def test_invalid_data_manifest_fails_from_content(tmp_path, monkeypatch) -> None:
    from s2and import production_model

    _stub_prerun(monkeypatch)
    monkeypatch.setattr(production_model, "load_production_model", lambda _path: SimpleNamespace(n_jobs=0))

    with pytest.raises(ValueError, match="requires normalization_version"):
        profile.run(_args(tmp_path, synthetic_seeds_when_clusters_missing=True))


def test_invalid_model_manifest_fails_from_content(tmp_path, monkeypatch) -> None:
    _stub_prerun(monkeypatch)
    _stub_inputs(monkeypatch, tmp_path, ["seed", "query"])

    with pytest.raises(ValueError, match="Unsupported production model bundle schema_version=None"):
        profile.run(_args(tmp_path))


def test_report_contains_release_schema_and_workload(tmp_path, monkeypatch) -> None:
    from s2and import production_model

    _stub_prerun(monkeypatch)
    _stub_inputs(monkeypatch, tmp_path, ["seed", "query"])
    clusterer = SimpleNamespace(n_jobs=0)
    clusterer.predict_incremental_from_arrow = lambda signature_ids, *_args, **_kwargs: {
        "incremental_linker_telemetry": {"arrow_promoted_incremental": 1},
        "incremental_linker_query_view": "raw_arrow",
        "clusters": {"cluster": list(signature_ids)},
    }
    monkeypatch.setattr(production_model, "load_production_model", lambda _path: clusterer)
    monkeypatch.setattr(
        profile,
        "ProcessTreeRSSMonitor",
        lambda _psutil: nullcontext(SimpleNamespace(peak_gb=1.25)),
    )
    monkeypatch.setattr(profile, "_run_metadata", lambda: {})

    payload = profile.run(_args(tmp_path))

    assert payload["schema_version"] == "s2and_performance_evaluation_report_v1"
    assert payload["runner"] == "promoted_incremental_arrow_profile"
    assert payload["summary"]["predict_seconds"]["p50"] >= 0
    assert payload["summary"]["peak_rss_gb"]["max"] == 1.25
    assert payload["workload"] == {
        "dataset": "dummy",
        "target_block": "",
        "query_limit": 25,
        "max_seed_clusters": 25,
        "seed_source": "clusters",
        "runs": 1,
        "n_jobs": 4,
        "batching_threshold": None,
        "total_ram_bytes": None,
        "synthetic_seeds_when_clusters_missing": False,
    }


def test_invalid_workload_fails(tmp_path, monkeypatch) -> None:
    from s2and import production_model

    _stub_prerun(monkeypatch)
    _stub_inputs(monkeypatch, tmp_path, ["seed"])
    monkeypatch.setattr(production_model, "load_production_model", lambda _path: SimpleNamespace(n_jobs=0))

    with pytest.raises(ValueError, match="no query signatures"):
        profile.run(_args(tmp_path))
