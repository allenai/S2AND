from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from s2and._sha256 import sha256_file
from s2and.arrow_inputs import RAW_PLANNER_ARROW_KEY_COLUMNS, ArrowDataset
from s2and.feature_port import build_rust_featurizer_from_arrow_dataset
from s2and.feature_schema import FEATURE_SCHEMA
from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from s2and.incremental_linking.gate_buckets import first_name_bucket_from_token_view
from s2and.incremental_linking.retrieval import RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS
from s2and.runtime import load_s2and_rust_extension
from scripts.production.data._incremental_query_views import append_initial_query_views, mask_initial_query_rows
from tests.helpers import write_arrow_tables_bundle, write_test_arrow_artifact_manifest
from tests.raw_arrow_helpers import base_arrow_paths


def _read(path: str | Path) -> pa.Table:
    with pa.memory_map(str(path), "r") as source:
        return pa.ipc.open_file(source).read_all()


def _paths(root: Path) -> dict[str, Path]:
    return {key: root / value for key, value in json.loads((root / "manifest.json").read_text())["paths"].items()}


def _selected(paths: dict[str, Path], table_name: str, values: set[str]) -> pa.Table:
    table = _read(paths[table_name])
    key = RAW_PLANNER_ARROW_KEY_COLUMNS[table_name]
    options = pc.SetLookupOptions(value_set=pa.array(sorted(values), type=pa.string()))
    return table.filter(pc.call_function("is_in", [table[key]], options=options))


def _initial_only_fixture(paths: dict[str, Path], output: Path) -> dict[str, Path]:
    signatures = _selected(paths, "signatures", {"q1", "s1", "s2"})
    assert signatures.num_rows == 3
    assert set(signatures["signature_id"].to_pylist()) == {"q1", "s1", "s2"}
    records = signatures.to_pylist()
    query = next(row for row in records if row["signature_id"] == "q1")
    query.update(author_first="a", author_middle="")
    roster = _selected(paths, "paper_authors", set(signatures["paper_id"].to_pylist()))
    roster_rows = roster.to_pylist()
    own_author = next(row for row in roster_rows if row["paper_id"] == "p_q" and row["position"] == 0)
    own_author["author_name"] = "a wang"
    masked = write_arrow_tables_bundle(
        output,
        {
            "signatures": pa.Table.from_pylist(records, schema=signatures.schema),
            "paper_authors": pa.Table.from_pylist(roster_rows, schema=roster.schema),
        },
    )
    return {**paths, **{key: Path(value) for key, value in masked.items()}}


def _fixture(root: Path) -> dict[str, str]:
    source = base_arrow_paths(root / "raw")
    signatures = _read(source["signatures"])
    middle = signatures.schema.field("author_middle")
    signatures = signatures.set_column(
        signatures.schema.get_field_index("author_middle"), middle, pa.array(["Marie", "", ""], type=middle.type)
    )
    signatures = signatures.append_column("author_block", pa.array(["a wang", "a wang", "b jones"]))
    signatures = signatures.append_column("source_author_ids", pa.array([["person_q"], ["person_a"], ["person_b"]]))
    signatures = signatures.append_column("source_extra", pa.array(["q proof", "a proof", "b proof"]))
    records = signatures.to_pylist()
    records.append(
        {
            **records[0],
            "signature_id": "q2",
            "author_first": "Ann",
            "author_middle": "",
            "author_last": "Smith",
            "author_position": 1,
            "author_block": "a smith",
        }
    )
    tables = {
        "signatures": pa.Table.from_pylist(records, schema=signatures.schema),
        "papers": _read(source["papers"]),
        "paper_authors": _read(source["paper_authors"]),
        "specter": pa.table(
            {
                "paper_id": pa.array(["p_q", "p1", "p2"]),
                "embedding": pa.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], type=pa.list_(pa.float32(), 2)),
            }
        ),
    }
    paths = write_arrow_tables_bundle(root / "source", tables)
    counts, _metrics = write_name_counts_index(
        root / "source" / "counts",
        (
            {"alice": 10.0, "ann": 11.0, "bob": 30.0},
            {"wang": 20.0, "smith": 21.0, "jones": 40.0},
            {"alice wang": 5.0, "ann smith": 7.0, "bob jones": 6.0},
            {"wang a": 8.0, "smith a": 9.0, "jones b": 9.0},
        ),
    )
    paths["name_counts_index"] = counts
    paths["cluster_seeds"] = str(shutil.copyfile(source["cluster_seeds"], root / "source" / "cluster_seeds.arrow"))
    write_test_arrow_artifact_manifest(root / "source", paths)
    return paths


def _matrix(dataset: ArrowDataset, query: str) -> np.ndarray:
    featurizer = build_rust_featurizer_from_arrow_dataset(
        dataset, signature_ids=[query, "s1"], name_tuples=frozenset(), num_threads=1
    )
    positions = {value: index for index, value in enumerate(featurizer.signature_ids())}
    return np.asarray(featurizer.featurize_pairs_matrix_indexed([(positions[query], positions["s1"])], None, 1, np.nan))


def test_derived_source_matches_public_overlay_in_native_training_and_pair_features(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    source_root = Path(paths["signatures"]).parent
    derived_root = tmp_path / "derived"
    shutil.copytree(source_root, derived_root)
    source_hashes = {path.name: sha256_file(path) for path in source_root.iterdir() if path.is_file()}
    original_tables = {key: _read(paths[key]) for key in ("signatures", "papers", "paper_authors", "specter")}
    result = append_initial_query_views(derived_root, ["q1"])
    clone = result.source_to_initial["q1"]
    assert clone.startswith("visibleq:")
    assert result.query_authors == {"q1": "a wang"}
    assert result.query_first_tokens == {"q1": "a"}
    assert result.source_paper_ids == {"q1": "p_q"}
    runtime = load_s2and_rust_extension()
    with ArrowDataset.open(source_root, require_specter=True, require_name_counts_index=True) as source:
        source_paths = _paths(source_root)
        overlay = _initial_only_fixture(source_paths, tmp_path / "overlay")
        original_features = _matrix(source, "q1")
        with ArrowDataset.open_trusted(overlay, require_specter=True, require_name_counts_index=True) as public:
            public_plan = runtime.RawBlockQueryCandidatePlanner.from_auto_queries(
                public.native, paths["cluster_seeds"], 2, orcid_enabled=False, num_threads=1
            ).plan(["q1"])
            public_features = _matrix(public, "q1")
    with ArrowDataset.open(derived_root, require_specter=True, require_name_counts_index=True) as derived:
        derived_paths = _paths(derived_root)
        training_plan = runtime.raw_arrow_labeled_candidate_plan(
            derived.native,
            [clone, clone],
            ["initial_only", "initial_only"],
            ["context", "context"],
            public_plan["row_component_keys"],
            public_plan["retrieval_ranks"].tolist(),
            {"c_match": ["s1"], "c_other": ["s2"]},
            num_threads=1,
        )
        derived_features = _matrix(derived, clone)
        for key, before in original_tables.items():
            after = _read(derived_paths[key])
            assert after.schema.equals(before.schema, check_metadata=True)
            assert after.slice(0, before.num_rows).equals(before)
        clone_row = _selected(derived_paths, "signatures", {clone}).to_pylist()[0]
        assert clone_row["paper_id"].startswith("visiblep:")
        assert result.initial_paper_ids == {"q1": clone_row["paper_id"]}
        source_row = next(row for row in original_tables["signatures"].to_pylist() if row["signature_id"] == "q1")
        for field in source_row.keys() - {"signature_id", "paper_id", "author_first", "author_middle"}:
            assert clone_row[field] == source_row[field]
        paper = _selected(derived_paths, "papers", {clone_row["paper_id"]}).to_pylist()[0]
        assert paper == {**original_tables["papers"].to_pylist()[0], "paper_id": clone_row["paper_id"]}
        roster = _selected(derived_paths, "paper_authors", {clone_row["paper_id"]}).to_pylist()
        assert [row["author_name"] for row in roster] == ["a wang", "Ann Smith"]
        embedding = _selected(derived_paths, "specter", {clone_row["paper_id"]}).to_pylist()[0]
        assert embedding["embedding"] == original_tables["specter"].to_pylist()[0]["embedding"]
    assert training_plan["query_views"] == public_plan["query_views"] == ["initial_only"]
    assert training_plan["query_authors"] == public_plan["query_authors"] == ["a wang"]
    np.testing.assert_allclose(training_plan["retrieval_scores"], public_plan["retrieval_scores"], rtol=1e-6)
    for key in ("retrieval_ranks", "row_query_first_tokens"):
        np.testing.assert_equal(training_plan[key], public_plan[key], err_msg=key)
    for key, _signal, dtype in RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS:
        if dtype == np.float32:
            np.testing.assert_allclose(training_plan[key], public_plan[key], rtol=1e-6, err_msg=key)
        else:
            np.testing.assert_equal(training_plan[key], public_plan[key], err_msg=key)
    assert (
        first_name_bucket_from_token_view(training_plan["row_query_first_tokens"][0], "initial_only")
        == (first_name_bucket_from_token_view(public_plan["row_query_first_tokens"][0], "initial_only"))
        == "single_letter_first"
    )
    np.testing.assert_equal(derived_features, public_features)
    columns = {feature.name: index for index, feature in enumerate(FEATURE_SCHEMA)}
    assert original_features[0, columns["single_char_first"]] == 0.0
    assert derived_features[0, columns["single_char_first"]] == 1.0
    assert original_features[0, columns["first_name_count_max"]] == 10.0
    assert np.isnan(derived_features[0, columns["first_name_count_max"]])
    assert original_features[0, columns["last_first_name_count_max"]] == 5.0
    assert np.isnan(derived_features[0, columns["last_first_name_count_max"]])
    assert source_hashes == {path.name: sha256_file(path) for path in source_root.iterdir() if path.is_file()}


def test_queries_on_one_paper_get_distinct_complete_rosters_and_stable_ids(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    source_root = Path(paths["signatures"]).parent
    first = tmp_path / "first"
    second = tmp_path / "second"
    shutil.copytree(source_root, first)
    shutil.copytree(source_root, second)
    first_result = append_initial_query_views(first, ["q2", "q1"])
    assert first_result == append_initial_query_views(second, ["q1", "q2"])
    assert first_result.source_paper_ids == {"q1": "p_q", "q2": "p_q"}
    assert len(set(first_result.initial_paper_ids.values())) == 2
    with ArrowDataset.open(first):
        bound = _paths(first)
        rows = _selected(bound, "signatures", set(first_result.source_to_initial.values())).to_pylist()
        clone_papers = {row["paper_id"] for row in rows}
        assert len(clone_papers) == 2
        rosters = _selected(bound, "paper_authors", clone_papers).to_pylist()
        for row in rows:
            names = [author["author_name"] for author in rosters if author["paper_id"] == row["paper_id"]]
            assert names == (["a wang", "Ann Smith"] if row["author_last"] == "Wang" else ["Alice Wang", "a smith"])
    before = {path.name: sha256_file(path) for path in first.iterdir() if path.is_file()}
    with pytest.raises(ValueError, match="already occur"):
        append_initial_query_views(first, ["q1"])
    assert before == {path.name: sha256_file(path) for path in first.iterdir() if path.is_file()}


@pytest.mark.parametrize("queries", [["missing"], ["q1", "q1"], [""]])
def test_invalid_queries_fail_before_dataset_changes(tmp_path: Path, queries: list[str]) -> None:
    paths = _fixture(tmp_path)
    root = Path(paths["signatures"]).parent
    before = {path.name: sha256_file(path) for path in root.iterdir() if path.is_file()}
    with pytest.raises(ValueError):
        append_initial_query_views(root, queries)
    assert before == {path.name: sha256_file(path) for path in root.iterdir() if path.is_file()}
    empty = append_initial_query_views(root, [])
    assert empty.source_to_initial == empty.source_paper_ids == empty.initial_paper_ids == {}
    assert json.loads((root / "manifest.json").read_text())["files"]["specter"]["sha256"] == before["specter.arrow"]


def test_masking_rejects_a_roster_without_the_query_position() -> None:
    signature = {
        "signature_id": "q",
        "paper_id": "p",
        "author_first": "Jonathan",
        "author_middle": "Q",
        "author_last": "Smith",
        "author_suffix": None,
        "author_position": 3,
    }
    roster = [
        {"paper_id": "p", "position": 2, "author_name": "Jonathan Q Smith"},
        {"paper_id": "p", "position": 0, "author_name": "Ann Lee"},
    ]
    with pytest.raises(ValueError, match="no row at author_position"):
        mask_initial_query_rows([signature], roster, "q")
    assert roster[0]["author_name"] == "Jonathan Q Smith"
    # A paper without roster rows has no full name to hide.
    mask_initial_query_rows([signature], [], "q")
    assert (signature["author_first"], signature["author_middle"]) == ("j", "")
