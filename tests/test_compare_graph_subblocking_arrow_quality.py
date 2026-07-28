import hashlib
import json
import sys
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
from s2and.incremental_linking.feature_block import write_arrow_batch_lookup_index
from scripts.production.model.run_binding import (
    build_run_binding_payload,
    evaluation_plan_content_identity,
)
from scripts.verification import compare_graph_subblocking_arrow_quality as script
from tests.helpers import write_minimal_arrow_prediction_bundle


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
        "rust-only",
        "--component-members-parquet",
        str(component_members),
        "--maximum-size",
        "2",
        "--limit",
        "2",
    ]


def test_compare_graph_subblocking_parser_requires_bounded_or_explicit_full_run(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", _base_argv())

    with pytest.raises(SystemExit):
        script.parse_args()


def _write_table(path, table) -> None:
    pa = pytest.importorskip("pyarrow")
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


def test_load_lightweight_dataset_from_arrow_builds_python_subblocking_view(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    arrow_root = tmp_path / "arrow"
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


def test_subblocking_release_report_contains_metrics_and_artifact(tmp_path, monkeypatch) -> None:
    public_data_root = tmp_path / "public"
    arrow_root = public_data_root / "dummy"
    write_minimal_arrow_prediction_bundle(arrow_root, include_specter=True)
    public_data_root.mkdir(exist_ok=True)
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
    component_members.write_text("reviewed\n", encoding="utf-8")
    public_manifest_sha256 = hashlib.sha256((arrow_root / "manifest.json").read_bytes()).hexdigest()
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
        "baselines": {},
        "cluster": {"dummy": {"blocks": file_spec}},
        "gates": {},
        "pairwise": {"dummy": {"pairs": file_spec}},
        "parity": {
            "block": "dummy",
            "dataset": "dummy",
            "files": {"signatures": file_spec},
            "fixture_dir": str(tmp_path.resolve()),
            "workload": {"fixture": True},
        },
        "performance": {
            "arrow_root": str(arrow_root.resolve()),
            "arrow_root_manifest_sha256": public_manifest_sha256,
            "workload": {"dataset": "dummy"},
        },
        "subblocking": {
            "component_members": component_spec,
            "dataset": "dummy",
            "workload": {"release": True},
        },
    }
    evaluation_plan = tmp_path / "evaluation_plan.json"
    evaluation_plan.write_text(json.dumps(evaluation_payload), encoding="utf-8")
    public_root_sha256 = hashlib.sha256((public_data_root / "manifest.json").read_bytes()).hexdigest()
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
                    "public_data_root_manifest_sha256": public_root_sha256,
                }
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _release_argv(arrow_root, public_data_root, component_members, output_dir),
    )
    args = script.parse_args()
    workload = {
        "allow_full": bool(args.allow_full),
        "comparison_mode": str(args.comparison_mode),
        "graph_config": script._graph_config(args).__dict__,
        "limit": args.limit,
        "maximum_size": int(args.maximum_size),
        "orcid_subblocking": bool(args.orcid_subblocking),
        "python_source": str(args.python_source),
        "sample_mode": str(args.sample_mode),
        "seed": int(args.seed),
        "top_diff_subblocks": int(args.top_diff_subblocks),
    }
    monkeypatch.setattr(
        script,
        "_load_evaluation_plan",
        lambda _path: SimpleNamespace(
            subblocking={
                "component_members": (
                    component_members.resolve(),
                    component_spec["sha256"],
                ),
                "dataset": "dummy",
                "workload": workload,
            }
        ),
    )
    monkeypatch.setattr(script, "validate_release_root", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(script, "load_signature_ids_from_arrow", lambda *_args, **_kwargs: ["s1", "s2"])
    monkeypatch.setattr(script, "_load_component_labels", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        script,
        "_run_rust_subblocking",
        lambda *_args, **_kwargs: ({"a": ["s1"], "b": ["s2"]}, {"ok": 1}),
    )

    script.main()

    report = json.loads((output_dir / "subblocking_evaluation_report.json").read_text(encoding="utf-8"))
    assert len(report["run_binding_sha256"]) == 64
    assert report["counts"]["signature_count"] == 2
    assert json.loads((output_dir / "rust_subblocks.json").read_text(encoding="utf-8")) == {
        "a": ["s1"],
        "b": ["s2"],
    }
