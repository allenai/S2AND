import hashlib
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from s2and.arrow_inputs import (
    PUBLIC_DATA_KIND,
    ArrowDataset,
    build_arrow_artifact_manifest,
    write_arrow_artifact_manifest,
)
from s2and.consts import PUBLIC_DATA_FORMAT_VERSION
from s2and.incremental_linking.contracts import canonical_json_digest
from s2and.incremental_linking.feature_block import write_arrow_batch_lookup_index, write_name_counts_index
from s2and.subblocking import GraphSubblockingConfig
from scripts.production.model.run_binding import (
    build_run_binding_payload,
    evaluation_plan_content_identity,
)
from scripts.verification import compare_graph_subblocking_arrow_quality as script
from tests.helpers import tiny_name_counts_tuple, write_minimal_arrow_prediction_bundle


def _base_argv() -> list[str]:
    return [
        "compare_graph_subblocking_arrow_quality.py",
        "--raw-root",
        "raw",
        "--specter-pickle",
        "specter.pkl",
        "--arrow-root",
        "arrow",
        "--public-data-root",
        "public",
        "--evaluation-plan",
        "evaluation_plan.json",
        "--run-binding",
        "run_binding.json",
        "--output-dir",
        "out",
        "--maximum-size",
        "2500",
    ]


def _release_argv(arrow_root, public_data_root, component_members, output_dir) -> list[str]:
    return [
        "compare_graph_subblocking_arrow_quality.py",
        "--arrow-root",
        str(arrow_root),
        "--public-data-root",
        str(public_data_root),
        "--evaluation-plan",
        str(output_dir.parent / "evaluation_plan.json"),
        "--run-binding",
        str(output_dir.parent / "run_binding.json"),
        "--output-dir",
        str(output_dir),
        "--comparison-mode",
        "python-vs-rust",
        "--python-source",
        "arrow",
        "--component-members-parquet",
        str(component_members),
        "--maximum-size",
        "2",
        "--allow-full",
    ]


def test_compare_graph_subblocking_parser_requires_bounded_or_explicit_full_run(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", _base_argv())

    with pytest.raises(SystemExit):
        script.parse_args()


def _write_table(path, table) -> None:
    import pyarrow as pa

    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def _write_indexed_table(path, table, *, key_column: str, table_name: str) -> tuple[str, str]:
    _write_table(path, table)
    index_key = f"{table_name}_batch_index"
    index_path = path.parent / f"{table_name}.{index_key}.bin"
    write_arrow_batch_lookup_index(
        path,
        index_path,
        key_column=key_column,
        table_name=table_name,
    )
    return str(path), str(index_path)


def _write_two_signature_arrow_bundle(arrow_root: Path) -> dict[str, str]:
    import pyarrow as pa

    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01]], dtype=np.float32)
    paths = write_minimal_arrow_prediction_bundle(arrow_root, include_specter=True)
    paths["signatures"], paths["signatures_batch_index"] = _write_indexed_table(
        arrow_root / "signatures.arrow",
        pa.table(
            {
                "signature_id": pa.array(["s1", "s2"], type=pa.string()),
                "paper_id": pa.array(["p1", "p2"], type=pa.string()),
                "author_first": pa.array(["Hui", "Hui"], type=pa.string()),
                "author_middle": pa.array(["", ""], type=pa.string()),
                "author_last": pa.array(["Wang", "Wang"], type=pa.string()),
                "author_suffix": pa.array(["", ""], type=pa.string()),
                "author_affiliations": pa.array([["AI Lab"], ["AI Lab"]], type=pa.list_(pa.string())),
                "author_orcid": pa.array([None, None], type=pa.string()),
                "author_position": pa.array([0, 0], type=pa.int64()),
            }
        ),
        key_column="signature_id",
        table_name="signatures",
    )
    paths["papers"], paths["papers_batch_index"] = _write_indexed_table(
        arrow_root / "papers.arrow",
        pa.table(
            {
                "paper_id": pa.array(["p1", "p2"], type=pa.string()),
                "title": pa.array(["First paper", "Second paper"], type=pa.string()),
                "venue": pa.array(["", ""], type=pa.string()),
                "journal_name": pa.array(["", ""], type=pa.string()),
            }
        ),
        key_column="paper_id",
        table_name="papers",
    )
    paths["paper_authors"], paths["paper_authors_batch_index"] = _write_indexed_table(
        arrow_root / "paper_authors.arrow",
        pa.table(
            {
                "paper_id": pa.array(["p1", "p1", "p1", "p2", "p2", "p2"], type=pa.string()),
                "position": pa.array([0, 1, 2, 0, 1, 2], type=pa.int64()),
                "author_name": pa.array(
                    ["Hui Wang", "Ada Lovelace", "", "Hui Wang", "Ada Lovelace", "   "],
                    type=pa.string(),
                ),
            }
        ),
        key_column="paper_id",
        table_name="paper_authors",
    )
    paths["specter"], paths["specter_batch_index"] = _write_indexed_table(
        arrow_root / "specter.arrow",
        pa.table(
            {
                "paper_id": pa.array(["p1", "p2"], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(pa.array(np.ravel(embeddings), type=pa.float32()), 2),
            }
        ),
        key_column="paper_id",
        table_name="specter",
    )
    write_arrow_artifact_manifest(build_arrow_artifact_manifest(paths, arrow_root), arrow_root)
    return paths


def test_load_lightweight_dataset_from_arrow_builds_python_subblocking_view(tmp_path) -> None:
    arrow_root = tmp_path / "arrow"
    _write_two_signature_arrow_bundle(arrow_root)

    with ArrowDataset.open(arrow_root, require_specter=True) as arrow_dataset:
        with arrow_dataset.use(require_specter=True) as arrow_lease:
            dataset, signature_ids = script.load_lightweight_dataset_from_arrow(
                arrow_lease,
                limit=2,
                sample_mode="first",
                seed=7,
            )

    assert signature_ids == ["s1", "s2"]
    assert dataset.signatures["s1"].author_info_first == "Hui"
    assert dataset.signatures["s1"].author_info_coauthor_blocks == ("a lovelace",)
    assert dataset.papers["p1"]["authors"][2] == {"position": 2, "author_name": ""}
    assert dataset.papers["p2"]["authors"][2] == {"position": 2, "author_name": "   "}
    assert np.allclose(dataset.specter_embeddings["p1"], np.array([1.0, 0.0], dtype=np.float32))


def test_direct_research_loaders_keep_bounded_input_support(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    (raw_root / "signatures.json").write_text(
        json.dumps(
            {
                signature_id: {"paper_id": "p1", "author_info": {"first": "Hui", "position": 0}}
                for signature_id in ("s1", "s2")
            }
        ),
        encoding="utf-8",
    )
    (raw_root / "papers.json").write_text(
        json.dumps({"p1": {"authors": [{"position": 0, "author_name": "Hui Wang"}]}}),
        encoding="utf-8",
    )
    specter_path = tmp_path / "specter.pkl"
    specter_path.write_bytes(pickle.dumps({"p1": np.array([1.0, 0.0], dtype=np.float32)}))

    dataset, signature_ids = script.load_lightweight_dataset(
        raw_root, specter_path, limit=1, sample_mode="first", seed=7
    )

    assert signature_ids == ["s1"]
    assert set(dataset.signatures) == {"s1"}
    assert dataset.signatures["s1"].author_info_first == "Hui"
    assert np.array_equal(dataset.specter_embeddings["p1"], [1.0, 0.0])

    arrow_root = tmp_path / "arrow"
    _write_two_signature_arrow_bundle(arrow_root)
    with ArrowDataset.open(arrow_root) as arrow_dataset:
        assert script.load_signature_ids_from_arrow(arrow_dataset, limit=1, sample_mode="first", seed=7) == ["s1"]


@pytest.fixture
def release_inputs(tmp_path: Path) -> SimpleNamespace:
    import pyarrow as pa
    import pyarrow.parquet as pq

    public_data_root = tmp_path / "public"
    arrow_root = public_data_root / "dummy"
    paths = _write_two_signature_arrow_bundle(arrow_root)
    paths["name_counts_index"], _metrics = write_name_counts_index(public_data_root, tiny_name_counts_tuple())
    write_arrow_artifact_manifest(build_arrow_artifact_manifest(paths, arrow_root), arrow_root)
    (public_data_root / "LICENSE.txt").write_text("Test fixture\n", encoding="utf-8")
    (public_data_root / "manifest.json").write_text(
        json.dumps(
            {
                "kind": PUBLIC_DATA_KIND,
                "release_version": "1.3",
                "format_version": PUBLIC_DATA_FORMAT_VERSION,
                "dataset_manifests": {
                    "dummy": {
                        "path": "dummy/manifest.json",
                        "sha256": hashlib.sha256((arrow_root / "manifest.json").read_bytes()).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    component_members = tmp_path / "components.parquet"
    pq.write_table(
        pa.table({"signature_id": ["s1", "s2"], "candidate_component_key": ["same", "same"]}),
        component_members,
    )
    public_manifest_sha256 = hashlib.sha256((public_data_root / "manifest.json").read_bytes()).hexdigest()
    output_dir = tmp_path / "report"
    identity_file = tmp_path / "identity.json"
    identity_file.write_text("{}\n", encoding="utf-8")
    identity_sha256 = hashlib.sha256(identity_file.read_bytes()).hexdigest()
    file_spec = {"path": str(identity_file.resolve()), "sha256": identity_sha256}
    component_spec = {
        "path": str(component_members.resolve()),
        "sha256": hashlib.sha256(component_members.read_bytes()).hexdigest(),
    }
    evaluation_payload = {
        "baseline_record_sha256": "a" * 64,
        "baselines": {
            "cluster_signature_weighted_b3_f1": 1.0,
            "pairwise_aggregate": {"auroc": 1.0, "macro_f1": 1.0},
            "pairwise_datasets": {"dummy": {"auroc": 1.0, "macro_f1": 1.0}},
            "predict_seconds_p50": 1.0,
        },
        "cluster": {
            "dummy": {role: file_spec for role in ("signatures", "papers", "specter_embeddings", "clusters", "blocks")}
        },
        "gates": {
            "cluster_signature_weighted_b3_f1_max_drop": 0.005,
            "pairwise_aggregate_auroc_max_drop": 0.001,
            "pairwise_aggregate_macro_f1_max_drop": 0.005,
            "pairwise_dataset_auroc_max_drop": 0.001,
            "pairwise_dataset_macro_f1_max_drop": 0.005,
            "peak_rss_absolute_max_gb": 4.0,
            "runtime_max_ratio": 1.1,
            "subblocking_maximum_size": 2,
        },
        "pairwise": {"dummy": {role: file_spec for role in ("signatures", "papers", "specter_embeddings", "pairs")}},
        "parity": {
            "block": "dummy",
            "dataset": "dummy",
            "files": {"signatures": file_spec, "papers": file_spec},
            "fixture_dir": str(tmp_path.resolve()),
            "workload": {
                "block_size": 2,
                "compare_features": True,
                "include_specter": False,
                "n_jobs": 1,
                "total_ram_bytes": 1_000_000,
                "use_cluster_seeds": False,
            },
        },
        "performance": {
            "arrow_root": str(public_data_root.resolve()),
            "arrow_root_manifest_sha256": public_manifest_sha256,
            "workload": {"dataset": "dummy"},
        },
        "subblocking": {
            "component_members": component_spec,
            "dataset": "dummy",
            "workload": {
                "allow_full": True,
                "comparison_mode": "python-vs-rust",
                "graph_config": asdict(GraphSubblockingConfig()),
                "limit": None,
                "maximum_size": 2,
                "orcid_subblocking": True,
                "python_source": "arrow",
                "sample_mode": "random",
                "seed": 42,
                "top_diff_subblocks": 30,
            },
        },
    }
    evaluation_plan = tmp_path / "evaluation_plan.json"
    evaluation_plan.write_text(json.dumps(evaluation_payload), encoding="utf-8")
    (tmp_path / "run_binding.json").write_text(
        json.dumps(
            build_run_binding_payload(
                {
                    "baseline_record_sha256": "a" * 64,
                    "candidate_model_manifest_sha256": "b" * 64,
                    "evaluation_plan_content_sha256": canonical_json_digest(
                        evaluation_plan_content_identity(evaluation_payload)
                    ),
                    "model_plan_content_sha256": "d" * 64,
                    "public_data_root_manifest_sha256": public_manifest_sha256,
                }
            )
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        argv=_release_argv(arrow_root, public_data_root, component_members, output_dir),
        evaluation_plan=evaluation_plan,
        output_dir=output_dir,
    )


def test_subblocking_release_report_contains_metrics_and_artifact(release_inputs, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", release_inputs.argv)

    def python_subblocks(_args, dataset, signature_ids, _config):
        assert signature_ids == ["s1", "s2"]
        assert dataset.signatures["s1"].author_info_coauthor_blocks == ("a lovelace",)
        assert set(dataset.specter_embeddings) == {"p1", "p2"}
        return {"python-block": list(signature_ids)}, {"implementation": "python"}

    def rust_subblocks(_args, arrow_dataset, signature_ids, _config):
        assert isinstance(arrow_dataset, ArrowDataset)
        assert signature_ids == ["s1", "s2"]
        return {"rust-block": list(signature_ids)}, {"implementation": "rust"}

    monkeypatch.setattr(script, "_run_python_subblocking", python_subblocks)
    monkeypatch.setattr(script, "_run_rust_subblocking", rust_subblocks)

    script.main()

    output_dir = release_inputs.output_dir
    report = json.loads((output_dir / "subblocking_evaluation_report.json").read_text(encoding="utf-8"))
    assert set(report) == {
        "run_binding_sha256",
        "inputs",
        "graph_config",
        "counts",
        "python",
        "rust",
        "artifacts",
        "baseline_deltas",
    }
    assert len(report["run_binding_sha256"]) == 64
    assert report["counts"]["signature_count"] == 2
    assert report["inputs"]["comparison_mode"] == "python-vs-rust"
    assert report["inputs"]["python_source"] == "arrow"
    assert report["inputs"]["raw_root"] is None
    assert report["inputs"]["specter_pickle"] is None
    assert report["inputs"]["limit"] is None
    assert report["inputs"]["allow_full"] is True
    assert report["baseline_deltas"] == {}
    assert report["python"]["hook_telemetry"] == {}
    for implementation in ("python", "rust"):
        expected_fields = {"seconds", "telemetry", "partition", "component_preservation"}
        if implementation == "python":
            expected_fields.add("hook_telemetry")
        assert set(report[implementation]) == expected_fields
        assert report[implementation]["telemetry"] == {"implementation": implementation}
        assert report[implementation]["partition"]["max_subblock_size"] == 2
        assert report[implementation]["component_preservation"]["component_pair_recall"] == 1.0
        subblocks_path = output_dir / f"{implementation}_subblocks.json"
        assert list(json.loads(subblocks_path.read_text(encoding="utf-8")).values()) == [["s1", "s2"]]
        assert report["artifacts"][f"{implementation}_subblocks"] == str(subblocks_path)
        diff_path = output_dir / f"diff_heavy_{implementation}_subblocks.csv"
        assert diff_path.is_file()
        assert report["artifacts"][f"{implementation}_diff_csv"] == str(diff_path)

    with pytest.raises(FileExistsError, match="Report output already exists"):
        script.main()


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--comparison-mode", "rust-only"],
        ["--python-source", "raw", "--raw-root", "raw", "--specter-pickle", "specter.pkl"],
        ["--limit", "1"],
        ["--maximum-size", "1"],
    ],
)
def test_subblocking_release_rejects_unbound_workloads(release_inputs, monkeypatch, extra_args) -> None:
    monkeypatch.setattr(sys, "argv", [*release_inputs.argv, *extra_args])

    with pytest.raises(ValueError, match="Release subblocking does not accept|workload does not match"):
        script.main()

    assert not list(release_inputs.output_dir.iterdir())


@pytest.mark.parametrize("change", [{"comparison_mode": "rust-only"}, {"python_source": "raw"}])
def test_subblocking_release_plan_rejects_research_modes(release_inputs, monkeypatch, change) -> None:
    evaluation_plan = release_inputs.evaluation_plan
    payload = json.loads(evaluation_plan.read_text(encoding="utf-8"))
    payload["subblocking"]["workload"].update(change)
    evaluation_plan.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", release_inputs.argv)

    with pytest.raises(ValueError, match="release subblocking must compare Python-vs-Rust"):
        script.main()

    assert not list(release_inputs.output_dir.iterdir())
