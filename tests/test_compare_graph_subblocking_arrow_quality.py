import sys

import numpy as np
import pytest

from s2and.incremental_linking.feature_block import write_arrow_batch_lookup_index
from scripts.verification import compare_graph_subblocking_arrow_quality as script


def _base_argv() -> list[str]:
    return [
        "compare_graph_subblocking_arrow_quality.py",
        "--raw-root",
        "raw",
        "--specter-pickle",
        "specter.pkl",
        "--arrow-root",
        "arrow",
        "--output-dir",
        "out",
        "--maximum-size",
        "2500",
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


def _write_indexed_table(path, table, *, key_column: str, table_name: str) -> None:
    _write_table(path, table)
    index_key = f"{table_name}_batch_index"
    write_arrow_batch_lookup_index(
        path,
        path.parent / f"{table_name}.{index_key}.bin",
        key_column=key_column,
        table_name=table_name,
    )


def test_load_lightweight_dataset_from_arrow_builds_python_subblocking_view(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    arrow_root = tmp_path / "arrow"
    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01]], dtype=np.float32)
    _write_indexed_table(
        arrow_root / "signatures.arrow",
        pa.table(
            {
                "signature_id": pa.array(["s1", "s2"], type=pa.string()),
                "paper_id": pa.array(["p1", "p2"], type=pa.string()),
                "author_first": pa.array(["Hui", "Hui"], type=pa.string()),
                "author_middle": pa.array(["", ""], type=pa.string()),
                "author_affiliations": pa.array([["AI Lab"], ["AI Lab"]], type=pa.list_(pa.string())),
                "author_orcid": pa.array([None, None], type=pa.string()),
                "author_position": pa.array([0, 0], type=pa.int64()),
            }
        ),
        key_column="signature_id",
        table_name="signatures",
    )
    _write_indexed_table(
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
    _write_indexed_table(
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

    dataset, signature_ids = script.load_lightweight_dataset_from_arrow(
        arrow_root,
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
