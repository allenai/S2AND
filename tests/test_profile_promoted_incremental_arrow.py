from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pyarrow as pa
import pytest

from s2and.arrow_inputs import ARROW_COLLECTION_KIND
from s2and.consts import PUBLIC_DATA_FORMAT_VERSION
from s2and.incremental_linking.contracts import canonical_json_digest
from s2and.incremental_linking.feature_block import write_name_counts_index
from scripts.production.model.run_binding import (
    build_run_binding_payload,
    evaluation_plan_content_identity,
)
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
    root_manifest = tmp_path / "arrow" / "manifest.json"
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text("{}\n", encoding="utf-8")
    root_manifest.write_text(
        json.dumps(
            {
                "kind": ARROW_COLLECTION_KIND,
                "format_version": PUBLIC_DATA_FORMAT_VERSION,
                "dataset_manifests": {
                    "dummy": {
                        "path": "dummy/manifest.json",
                        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
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
    identity_file = tmp_path / "identity.json"
    identity_file.write_text("{}\n", encoding="utf-8")
    identity_sha256 = hashlib.sha256(identity_file.read_bytes()).hexdigest()
    root_sha256 = hashlib.sha256(root_manifest.read_bytes()).hexdigest()
    file_spec = {"path": str(identity_file.resolve()), "sha256": identity_sha256}
    evaluation_payload = {
        "pairwise": {"dummy": {"pairs": file_spec}},
        "cluster": {"dummy": {"blocks": file_spec}},
        "parity": {
            "block": "dummy",
            "dataset": "dummy",
            "files": {"signatures": file_spec},
            "fixture_dir": str(tmp_path.resolve()),
            "workload": {"fixture": True},
        },
        "performance": {
            "arrow_root": str((tmp_path / "arrow").resolve()),
            "arrow_root_manifest_sha256": root_sha256,
            "workload": workload,
        },
        "baseline_record_sha256": "b" * 64,
        "baselines": {},
        "gates": {},
        "subblocking": {
            "component_members": file_spec,
            "dataset": "dummy",
            "workload": {"full": True},
        },
    }
    evaluation_plan.write_text(json.dumps(evaluation_payload), encoding="utf-8")
    binding = build_run_binding_payload(
        {
            "baseline_record_sha256": "b" * 64,
            "candidate_model_manifest_sha256": hashlib.sha256((model_dir / "manifest.json").read_bytes()).hexdigest(),
            "evaluation_plan_content_sha256": canonical_json_digest(
                evaluation_plan_content_identity(evaluation_payload)
            ),
            "model_plan_content_sha256": "a" * 64,
            "public_data_root_manifest_sha256": "c" * 64,
        }
    )
    run_binding = tmp_path / "run_binding.json"
    run_binding.write_text(json.dumps(binding), encoding="utf-8")
    return profile.parse_args(
        [
            "--evaluation-plan",
            str(evaluation_plan),
            "--model-path",
            str(model_dir),
            "--run-binding",
            str(run_binding),
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


def test_resolve_dataset_root_rejects_undeclared_sibling(tmp_path) -> None:
    arrow_root = tmp_path / "arrow"
    declared_manifest = arrow_root / "declared" / "manifest.json"
    undeclared_manifest = arrow_root / "dummy" / "manifest.json"
    declared_manifest.parent.mkdir(parents=True)
    undeclared_manifest.parent.mkdir()
    declared_manifest.write_text("{}\n", encoding="utf-8")
    undeclared_manifest.write_text("{}\n", encoding="utf-8")
    (arrow_root / "manifest.json").write_text(
        json.dumps(
            {
                "kind": ARROW_COLLECTION_KIND,
                "format_version": PUBLIC_DATA_FORMAT_VERSION,
                "dataset_manifests": {
                    "declared": {
                        "path": "declared/manifest.json",
                        "sha256": hashlib.sha256(declared_manifest.read_bytes()).hexdigest(),
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not declare dataset='dummy'"):
        profile._resolve_dataset_root(arrow_root, "dummy")  # noqa: SLF001


def test_run_rejects_unbounded_or_negative_workload_without_full_run(tmp_path) -> None:
    cases = (
        ("zero-query-limit", "query_limit", 0, "query_limit must be 1"),
        ("negative-query-limit", "query_limit", -1, "query_limit must be >= 0"),
        ("zero-seed-clusters", "max_seed_clusters", 0, "max_seed_clusters must be 1"),
        ("negative-seed-clusters", "max_seed_clusters", -1, "max_seed_clusters must be >= 0"),
    )
    for case_id, field, value, message in cases:
        case_root = tmp_path / case_id
        case_root.mkdir()
        try:
            profile.run(_args(case_root, **{field: value}))
        except ValueError as error:
            assert message in str(error), f"{case_id}: {error}"
        else:
            raise AssertionError(f"{case_id}: unsafe workload was accepted")


def test_main_writes_fresh_report_and_rejects_reuse(tmp_path, monkeypatch, capsys) -> None:
    report_path = tmp_path / "report.json"
    payload = {"summary": {"run_count": 1}}
    monkeypatch.setattr(profile, "run", lambda _args: payload)

    exit_code = profile.main(
        [
            "--evaluation-plan",
            "evaluation-plan.json",
            "--model-path",
            "model",
            "--run-binding",
            "run_binding.json",
            "--write-json",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(report_path.read_text(encoding="utf-8")) == payload
    assert json.loads(capsys.readouterr().out) == payload

    monkeypatch.setattr(profile, "run", lambda _args: pytest.fail("profile must not run"))
    with pytest.raises(FileExistsError, match="already exists"):
        profile.main(
            [
                "--evaluation-plan",
                "evaluation-plan.json",
                "--model-path",
                "model",
                "--run-binding",
                "run_binding.json",
                "--write-json",
                str(report_path),
            ]
        )


def test_invalid_data_or_model_manifest_fails_from_content(tmp_path, monkeypatch) -> None:
    from s2and import production_model

    data_root = tmp_path / "data"
    data_root.mkdir()
    with monkeypatch.context() as data_patch:
        _stub_prerun(data_patch)
        data_patch.setattr(production_model, "load_production_model", lambda _path: SimpleNamespace(n_jobs=0))
        with pytest.raises(ValueError, match="Arrow artifact manifest .* fields mismatch"):
            profile.run(_args(data_root, synthetic_seeds_when_clusters_missing=True))

    model_root = tmp_path / "model"
    model_root.mkdir()
    _stub_prerun(monkeypatch)
    _stub_inputs(monkeypatch, model_root, ["seed", "query"])
    with pytest.raises(ValueError, match="Production model manifest field mismatch"):
        profile.run(_args(model_root))


def test_report_contains_release_workload_and_metrics(tmp_path, monkeypatch) -> None:
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

    assert payload["runner"] == "promoted_incremental_arrow_profile"
    assert len(payload["run_binding_sha256"]) == 64
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
