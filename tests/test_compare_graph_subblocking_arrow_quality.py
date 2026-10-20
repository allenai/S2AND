import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from s2and.arrow_inputs import (
    ArrowDataset,
    build_arrow_artifact_manifest,
    write_arrow_artifact_manifest,
)
from s2and.incremental_linking.feature_block import write_arrow_batch_lookup_index, write_name_counts_index
from scripts.verification import compare_graph_subblocking_arrow_quality as script
from tests.helpers import tiny_name_counts_tuple, write_minimal_arrow_prediction_bundle


def _base_argv() -> list[str]:
    return [
        "compare_graph_subblocking_arrow_quality.py",
        "--arrow-root",
        "arrow",
        "--component-members-parquet",
        "components.parquet",
        "--output-dir",
        "out",
        "--maximum-size",
        "2500",
    ]


def _run_argv(arrow_root: Path, component_members: Path, output_dir: Path) -> list[str]:
    return [
        "compare_graph_subblocking_arrow_quality.py",
        "--arrow-root",
        str(arrow_root),
        "--output-dir",
        str(output_dir),
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


def test_arrow_signature_ids_respect_the_limit(tmp_path: Path) -> None:
    arrow_root = tmp_path / "arrow"
    _write_two_signature_arrow_bundle(arrow_root)
    with ArrowDataset.open(arrow_root) as arrow_dataset:
        assert script.load_signature_ids_from_arrow(arrow_dataset, limit=1, sample_mode="first", seed=7) == ["s1"]


@pytest.fixture
def run_inputs(tmp_path: Path) -> SimpleNamespace:
    import pyarrow as pa
    import pyarrow.parquet as pq

    arrow_root = tmp_path / "arrow"
    paths = _write_two_signature_arrow_bundle(arrow_root)
    paths["name_counts_index"], _metrics = write_name_counts_index(tmp_path, tiny_name_counts_tuple())
    write_arrow_artifact_manifest(build_arrow_artifact_manifest(paths, arrow_root), arrow_root)
    component_members = tmp_path / "components.parquet"
    pq.write_table(
        pa.table({"signature_id": ["s1", "s2"], "candidate_component_key": ["same", "same"]}),
        component_members,
    )
    output_dir = tmp_path / "report"
    return SimpleNamespace(
        argv=_run_argv(arrow_root, component_members, output_dir),
        output_dir=output_dir,
    )


def _patch_subblocking(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_subblocking_report_contains_metrics_and_artifacts(run_inputs, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", run_inputs.argv)
    _patch_subblocking(monkeypatch)

    script.main()

    output_dir = run_inputs.output_dir
    report = json.loads((output_dir / "subblocking_evaluation_report.json").read_text(encoding="utf-8"))
    assert set(report) == {
        "inputs",
        "graph_config",
        "counts",
        "python",
        "rust",
        "artifacts",
        "baseline_deltas",
    }
    assert report["counts"]["signature_count"] == 2
    assert report["inputs"]["comparison_mode"] == "python-vs-rust"
    assert report["inputs"]["python_source"] == "arrow"
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


def test_subblocking_report_compares_with_a_baseline_summary(run_inputs, monkeypatch) -> None:
    _patch_subblocking(monkeypatch)
    monkeypatch.setattr(sys, "argv", run_inputs.argv)
    script.main()
    baseline = run_inputs.output_dir / "subblocking_evaluation_report.json"
    second = run_inputs.output_dir.parent / "second"
    argv = [*run_inputs.argv, "--baseline-summary", str(baseline)]
    argv[argv.index(str(run_inputs.output_dir))] = str(second)
    monkeypatch.setattr(sys, "argv", argv)

    script.main()

    report = json.loads((second / "subblocking_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["baseline_deltas"]["rust.component_preservation.component_pair_recall"] == 0.0
