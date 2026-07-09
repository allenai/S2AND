from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

import s2and.subblocking as subblocking
from s2and.data import Signature
from s2and.incremental_linking.feature_block import write_arrow_batch_lookup_index


def _write_ipc(path, table: pa.Table) -> str:
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return str(path)


def _signature(signature_id: str, *, first: str, middle: str | None = None, orcid: str | None = None) -> Signature:
    return Signature(
        author_info_first=first,
        author_info_first_normalized_without_apostrophe=first,
        author_info_middle=middle,
        author_info_middle_normalized_without_apostrophe=middle or "",
        author_info_last_normalized="wang",
        author_info_last="Wang",
        author_info_suffix_normalized=None,
        author_info_suffix=None,
        author_info_coauthors=None,
        author_info_coauthor_blocks=None,
        author_info_full_name=None,
        author_info_affiliations=[],
        author_info_affiliations_n_grams=None,
        author_info_coauthor_n_grams=None,
        author_info_email=None,
        author_info_orcid=orcid,
        author_info_name_counts=None,
        author_info_position=0,
        author_info_block="h wang",
        author_info_given_block=None,
        author_info_estimated_gender=None,
        author_info_estimated_ethnicity=None,
        paper_id=int(signature_id[1:]) if signature_id[1:].isdigit() else 0,
        sourced_author_source=None,
        sourced_author_ids=[],
        author_id=None,
        signature_id=signature_id,
    )


def test_make_subblocks_uses_specter_for_oversized_single_letter_block(monkeypatch):
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="h", middle=""),
            "s2": _signature("s2", first="h", middle=""),
            "s3": _signature("s3", first="h", middle=""),
            "s4": _signature("s4", first="h", middle=""),
        },
        random_seed=0,
    )

    monkeypatch.setattr(
        subblocking,
        "cluster_with_specter",
        lambda signature_ids, anddata, target_subblock_size, compute_block_fn=None: {
            "0": list(signature_ids[:2]),
            "1": list(signature_ids[2:]),
        },
    )

    subblocks, _telemetry = subblocking.make_subblocks_with_telemetry(
        ["s1", "s2", "s3", "s4"],
        dataset,
        maximum_size=2,
        first_k_letter_counts_sorted={},
    )

    assert sorted(sorted(signature_ids) for signature_ids in subblocks.values()) == [["s1", "s2"], ["s3", "s4"]]


def test_make_subblocks_skips_specter_when_single_letter_block_is_in_budget(monkeypatch):
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="h", middle=""),
            "s2": _signature("s2", first="h", middle=""),
        },
        random_seed=0,
    )

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("cluster_with_specter should not be called when the dead-end block is already in budget")

    monkeypatch.setattr(subblocking, "cluster_with_specter", _fail_if_called)

    subblocks, _telemetry = subblocking.make_subblocks_with_telemetry(
        ["s1", "s2"],
        dataset,
        maximum_size=2,
        first_k_letter_counts_sorted={},
    )

    assert sorted(sorted(signature_ids) for signature_ids in subblocks.values()) == [["s1", "s2"]]


def test_make_subblocks_with_telemetry_uses_python_implementation(monkeypatch):
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="anna", middle=""),
            "s2": _signature("s2", first="anna", middle=""),
        },
        random_seed=0,
    )
    observed = {"python_subdivide_called": False}

    def fake_subdivide_helper(names, sig_ids, maximum_size, starting_k=2):
        del names, maximum_size, starting_k
        observed["python_subdivide_called"] = True
        return {"an": np.array(list(sig_ids))}, {}

    monkeypatch.setattr(subblocking, "subdivide_helper", fake_subdivide_helper)

    subblocks, telemetry = subblocking.make_subblocks_with_telemetry(
        ["s1", "s2"],
        dataset,
        maximum_size=3,
        first_k_letter_counts_sorted={},
    )

    assert observed["python_subdivide_called"] is True
    assert subblocks == {"an": ["s1", "s2"]}
    assert telemetry["input_signature_count"] == 2


def test_subdivide_helper_accepts_prefix_exactly_at_capacity() -> None:
    names = np.array(["anna", "anna", "bill"])
    signature_ids = np.array(["s1", "s2", "s3"])

    output, dead_ends = subblocking.subdivide_helper(names, signature_ids, maximum_size=2, starting_k=2)

    assert dead_ends == {}
    assert {key: sorted(values.tolist()) for key, values in output.items()} == {
        "an": ["s1", "s2"],
        "bi": ["s3"],
    }


def test_union_find_find_compresses_long_parent_chain_without_recursion() -> None:
    union_find = subblocking._UnionFind(1500)  # noqa: SLF001
    union_find.parent = list(range(1, 1500)) + [1499]

    assert union_find.find(0) == 1499
    assert union_find.parent[0] == 1499


def test_normalize_orcid_for_subblocking_matches_rust_arrow_canonical_form() -> None:
    assert (
        subblocking.normalize_orcid_for_subblocking(" https://orcid.org/0000-0002-1825-0097 ") == "0000-0002-1825-0097"
    )
    assert subblocking.normalize_orcid_for_subblocking("0000\u20110002\u20111825\u20110097") == "0000-0002-1825-0097"
    assert subblocking.normalize_orcid_for_subblocking("ORCID: 000000021825009x") == "0000-0002-1825-009X"
    assert subblocking.normalize_orcid_for_subblocking("https://example.org/0000-0002-1825-0097") == (
        "0000-0002-1825-0097"
    )
    assert subblocking.normalize_orcid_for_subblocking("0000-0002-1825-009 https://example.org") is None
    assert subblocking.normalize_orcid_for_subblocking("0000-0002-1825") is None
    assert subblocking.normalize_orcid_for_subblocking("   ") is None


def test_signature_name_parts_for_subblocking_recomputes_deferred_normalized_fields() -> None:
    # canonical_v2 (D4): dash-bound compounds stay together regardless of the
    # dash code point; the legacy non-ASCII-dash spill repair is retired.
    signature = SimpleNamespace(
        author_info_first="Arif\u2010ullah",
        author_info_middle=None,
        author_info_first_normalized_without_apostrophe=None,
        author_info_middle_normalized_without_apostrophe=None,
    )

    assert subblocking.signature_name_parts_for_subblocking(signature) == ("arif ullah", "")


def test_signature_name_parts_for_subblocking_treats_all_dashes_uniformly() -> None:
    unicode_dash = SimpleNamespace(
        author_info_first="Sang\u2010Min",
        author_info_middle=None,
        author_info_first_normalized_without_apostrophe="sang min",
        author_info_middle_normalized_without_apostrophe="",
    )
    ascii_dash = SimpleNamespace(
        author_info_first="Sang-Min",
        author_info_middle=None,
        author_info_first_normalized_without_apostrophe="sang min",
        author_info_middle_normalized_without_apostrophe="",
    )

    assert subblocking.signature_name_parts_for_subblocking(unicode_dash) == ("sang min", "")
    assert subblocking.signature_name_parts_for_subblocking(ascii_dash) == ("sang min", "")


def test_coauthor_blocks_from_rowwise_arrow_normalizes_author_names(tmp_path) -> None:
    paper_authors = pa.table(
        {
            "paper_id": pa.array(["p1", "p1"], type=pa.string()),
            "position": pa.array([0, 1], type=pa.int64()),
            "author_name": pa.array(["O'Connor", "Maciej Górski"], type=pa.string()),
        }
    )

    paper_authors_path = _write_ipc(tmp_path / "paper_authors.arrow", paper_authors)
    paper_authors_index_path = tmp_path / "paper_authors.paper_authors_batch_index.bin"
    write_arrow_batch_lookup_index(
        paper_authors_path,
        paper_authors_index_path,
        key_column="paper_id",
        table_name="paper_authors",
    )

    out = subblocking._coauthor_blocks_by_paper_from_arrow(  # noqa: SLF001
        {"paper_authors": paper_authors_path, "paper_authors_batch_index": str(paper_authors_index_path)},
        ["p1"],
        load_metrics={},
    )

    assert out == {"p1": [(0, "o connor"), (1, "m gorski")]}


def test_subblock_merge_candidate_metadata_preserves_middle_values_with_equals() -> None:
    assert subblocking._subblock_merge_candidate_metadata("a|middle=b=c", 2) == (2, "a", "b=c", "b=c", "b=c")


def test_make_subblocks_merges_normalized_orcid_components_when_enabled(monkeypatch):
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="aa", middle="", orcid="https://orcid.org/0000-0002-1825-0097"),
            "s2": _signature("s2", first="bb", middle="", orcid="ORCID: 0000000218250097"),
            "s3": _signature("s3", first="aa", middle="", orcid="   "),
            "s4": _signature("s4", first="cc", middle="", orcid="   "),
        },
        random_seed=0,
    )

    call_count = {"value": 0}

    def fake_subdivide_helper(names, sig_ids, maximum_size, starting_k=2):
        del names, sig_ids, maximum_size, starting_k
        call_count["value"] += 1
        if call_count["value"] == 1:
            return {
                "a": np.array(["s1", "s3"]),
                "b": np.array(["s2"]),
                "c": np.array(["s4"]),
            }, {}
        raise AssertionError("Unexpected extra call to subdivide_helper")

    monkeypatch.setattr(subblocking, "subdivide_helper", fake_subdivide_helper)

    subblocks, telemetry = subblocking.make_subblocks_with_telemetry(
        ["s1", "s2", "s3", "s4"],
        dataset,
        maximum_size=3,
        first_k_letter_counts_sorted={},
    )

    assert telemetry["orcid_subblocking_enabled"] is True
    assert sorted(sorted(signature_ids) for signature_ids in subblocks.values()) == [["s1", "s2", "s3"], ["s4"]]


def test_make_subblocks_orcid_repair_merges_whole_subblocks(monkeypatch):
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="bb", middle="", orcid="0000-0000-0000-0001"),
            "s2": _signature("s2", first="bb", middle="", orcid="0000-0000-0000-0001"),
            "s3": _signature("s3", first="aa", middle="", orcid="0000-0000-0000-0001"),
            "s4": _signature("s4", first="aa", middle=""),
            "s5": _signature("s5", first="aa", middle=""),
            "s6": _signature("s6", first="bb", middle=""),
        },
        random_seed=0,
    )

    call_count = {"value": 0}

    def fake_subdivide_helper(names, sig_ids, maximum_size, starting_k=2):
        del names, sig_ids, maximum_size, starting_k
        call_count["value"] += 1
        if call_count["value"] == 1:
            return {
                "a": np.array(["s3", "s4", "s5"]),
                "b": np.array(["s1", "s2", "s6"]),
            }, {}
        raise AssertionError("Unexpected extra call to subdivide_helper")

    monkeypatch.setattr(subblocking, "subdivide_helper", fake_subdivide_helper)

    subblocks, _telemetry = subblocking.make_subblocks_with_telemetry(
        ["s1", "s2", "s3", "s4", "s5", "s6"],
        dataset,
        maximum_size=6,
        first_k_letter_counts_sorted={},
    )

    assert sorted(sorted(signature_ids) for signature_ids in subblocks.values()) == [
        ["s1", "s2", "s3", "s4", "s5", "s6"],
    ]


def test_make_subblocks_orcid_repair_does_not_extract_from_oversized_whole_merge(monkeypatch):
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="aa", middle="", orcid="0000-0000-0000-0001"),
            "s2": _signature("s2", first="aa", middle="", orcid="0000-0000-0000-0001"),
            "s3": _signature("s3", first="aa", middle=""),
            "s4": _signature("s4", first="bb", middle="", orcid="0000-0000-0000-0001"),
            "s5": _signature("s5", first="bb", middle=""),
        },
        random_seed=0,
    )

    call_count = {"value": 0}

    def fake_subdivide_helper(names, sig_ids, maximum_size, starting_k=2):
        del names, sig_ids, maximum_size, starting_k
        call_count["value"] += 1
        if call_count["value"] == 1:
            return {
                "a": np.array(["s1", "s2", "s3"]),
                "b": np.array(["s4", "s5"]),
            }, {}
        raise AssertionError("Unexpected extra call to subdivide_helper")

    monkeypatch.setattr(subblocking, "subdivide_helper", fake_subdivide_helper)

    subblocks, telemetry = subblocking.make_subblocks_with_telemetry(
        ["s1", "s2", "s3", "s4", "s5"],
        dataset,
        maximum_size=4,
        first_k_letter_counts_sorted={},
    )

    assert sorted(sorted(signature_ids) for signature_ids in subblocks.values()) == [
        ["s1", "s2", "s3"],
        ["s4", "s5"],
    ]
    assert telemetry["orcid_merge_skipped_due_to_capacity_count"] == 1
    assert telemetry["orcid_merge_skipped_due_to_capacity_signature_count"] == 3


def test_make_subblocks_leaves_orcid_components_split_when_disabled(monkeypatch):
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="aa", middle="", orcid="https://orcid.org/0000-0002-1825-0097"),
            "s2": _signature("s2", first="bb", middle="", orcid="ORCID: 0000000218250097"),
            "s3": _signature("s3", first="cc", middle=""),
        },
        random_seed=0,
    )

    call_count = {"value": 0}

    def fake_subdivide_helper(names, sig_ids, maximum_size, starting_k=2):
        del names, sig_ids, maximum_size, starting_k
        call_count["value"] += 1
        if call_count["value"] == 1:
            return {
                "a": np.array(["s1"]),
                "b": np.array(["s2"]),
                "c": np.array(["s3"]),
            }, {}
        raise AssertionError("Unexpected extra call to subdivide_helper")

    monkeypatch.setattr(subblocking, "subdivide_helper", fake_subdivide_helper)

    subblocks, telemetry = subblocking.make_subblocks_with_telemetry(
        ["s1", "s2", "s3"],
        dataset,
        maximum_size=3,
        first_k_letter_counts_sorted={},
        use_orcid_subblocking=False,
    )

    assert telemetry["orcid_subblocking_enabled"] is False
    assert sorted(sorted(signature_ids) for signature_ids in subblocks.values()) == [["s1"], ["s2"], ["s3"]]


def test_make_subblocks_does_not_merge_orcid_components_past_capacity(monkeypatch):
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="aa", middle="", orcid="0000-0000-0000-0001"),
            "s2": _signature("s2", first="aa", middle="", orcid="0000-0000-0000-0002"),
            "s3": _signature("s3", first="bb", middle="", orcid="0000-0000-0000-0001"),
            "s4": _signature("s4", first="bb", middle=""),
            "s5": _signature("s5", first="cc", middle="", orcid="0000-0000-0000-0002"),
            "s6": _signature("s6", first="cc", middle=""),
        },
        random_seed=0,
    )

    call_count = {"value": 0}

    def fake_subdivide_helper(names, sig_ids, maximum_size, starting_k=2):
        del names, sig_ids, maximum_size, starting_k
        call_count["value"] += 1
        if call_count["value"] == 1:
            return {
                "a": np.array(["s1", "s2"]),
                "b": np.array(["s3", "s4"]),
                "c": np.array(["s5", "s6"]),
            }, {}
        raise AssertionError("Unexpected extra call to subdivide_helper")

    def fail_if_specter_called(*_args, **_kwargs):
        raise AssertionError("cluster_with_specter should not be called in this regression test")

    monkeypatch.setattr(subblocking, "subdivide_helper", fake_subdivide_helper)
    monkeypatch.setattr(subblocking, "cluster_with_specter", fail_if_specter_called)

    subblocks, telemetry = subblocking.make_subblocks_with_telemetry(
        ["s1", "s2", "s3", "s4", "s5", "s6"],
        dataset,
        maximum_size=2,
        first_k_letter_counts_sorted={},
    )

    assert sorted(sorted(signature_ids) for signature_ids in subblocks.values()) == [
        ["s1", "s2"],
        ["s3", "s4"],
        ["s5", "s6"],
    ]
    assert telemetry["orcid_merge_skipped_due_to_capacity_count"] == 2
    assert telemetry["orcid_merge_skipped_due_to_capacity_signature_count"] == 4


def test_make_subblocks_orcid_repair_skips_oversized_connected_component_without_partial_merge(monkeypatch):
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="aa", middle="", orcid="0000-0000-0000-0001"),
            "s2": _signature("s2", first="bb", middle="", orcid="0000-0000-0000-0001"),
            "s3": _signature("s3", first="bb", middle="", orcid="0000-0000-0000-0002"),
            "s4": _signature("s4", first="cc", middle="", orcid="0000-0000-0000-0002"),
        },
        random_seed=0,
    )

    def fake_subdivide_helper(names, sig_ids, maximum_size, starting_k=2):
        del names, sig_ids, maximum_size, starting_k
        return {
            "a": np.array(["s1"]),
            "b": np.array(["s2", "s3"]),
            "c": np.array(["s4"]),
        }, {}

    monkeypatch.setattr(subblocking, "subdivide_helper", fake_subdivide_helper)

    subblocks, telemetry = subblocking.make_subblocks_with_telemetry(
        ["s1", "s2", "s3", "s4"],
        dataset,
        maximum_size=3,
        first_k_letter_counts_sorted={},
    )

    assert subblocks == {
        "a": ["s1"],
        "b": ["s2", "s3"],
        "c": ["s4"],
    }
    assert telemetry["orcid_merge_skipped_due_to_capacity_count"] == 2
    assert telemetry["orcid_merge_skipped_due_to_capacity_signature_count"] == 4


def _require_rust_arrow_subblocking():
    rust_module = pytest.importorskip("s2and_rust")
    if not hasattr(rust_module, "make_subblocks_with_telemetry_arrow_native_graph"):
        raise pytest.skip.Exception("s2and_rust.make_subblocks_with_telemetry_arrow_native_graph is unavailable")
    return rust_module


def _write_signatures_arrow(
    path,
    rows: list[tuple[str, str, str, str | None]],
    *,
    author_positions: list[int | None] | None = None,
) -> None:
    pa = pytest.importorskip("pyarrow")
    ipc = pytest.importorskip("pyarrow.ipc")
    positions = [0] * len(rows) if author_positions is None else author_positions
    table = pa.table(
        {
            "signature_id": pa.array([row[0] for row in rows], type=pa.string()),
            "paper_id": pa.array([f"p_{row[0]}" for row in rows], type=pa.string()),
            "author_first": pa.array([row[1] for row in rows], type=pa.string()),
            "author_middle": pa.array([row[2] for row in rows], type=pa.string()),
            "author_last": pa.array(["wang"] * len(rows), type=pa.string()),
            "author_suffix": pa.array([""] * len(rows), type=pa.string()),
            "author_affiliations": pa.array([[] for _row in rows], type=pa.list_(pa.string())),
            "author_orcid": pa.array([row[3] for row in rows], type=pa.string()),
            "author_position": pa.array(positions, type=pa.int64()),
        }
    )
    with path.open("wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def _add_batch_index(path, index_path, *, key_column: str, table_name: str) -> str:
    write_arrow_batch_lookup_index(path, index_path, key_column=key_column, table_name=table_name)
    return str(index_path)


def test_rust_arrow_make_subblocks_matches_python_orcid_repair(tmp_path):
    _require_rust_arrow_subblocking()
    signatures_path = tmp_path / "signatures.arrow"
    _write_signatures_arrow(
        signatures_path,
        [
            ("s1", "aa", "", "https://orcid.org/0000-0002-1825-0097"),
            ("s2", "bb", "", "ORCID: 0000000218250097"),
            ("s3", "aa", "", "   "),
            ("s4", "cc", "", "   "),
        ],
    )
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="aa", middle="", orcid="https://orcid.org/0000-0002-1825-0097"),
            "s2": _signature("s2", first="bb", middle="", orcid="ORCID: 0000000218250097"),
            "s3": _signature("s3", first="aa", middle="", orcid="   "),
            "s4": _signature("s4", first="cc", middle="", orcid="   "),
        },
        random_seed=0,
    )

    python_subblocks, python_telemetry = subblocking.make_subblocks_with_telemetry(
        ["s1", "s2", "s3", "s4"],
        dataset,
        maximum_size=3,
        first_k_letter_counts_sorted={},
    )
    rust_subblocks, rust_telemetry = subblocking._make_subblocks_with_telemetry_arrow_rust(
        {
            "signatures": str(signatures_path),
            "signatures_batch_index": _add_batch_index(
                signatures_path,
                tmp_path / "signatures.signatures_batch_index.bin",
                key_column="signature_id",
                table_name="signatures",
            ),
        },
        ["s1", "s2", "s3", "s4"],
        maximum_size=3,
        first_k_letter_counts_sorted={},
        graph_subblocking_config=subblocking.GraphSubblockingConfig(),
    )

    assert sorted(sorted(signature_ids) for signature_ids in rust_subblocks.values()) == sorted(
        sorted(signature_ids) for signature_ids in python_subblocks.values()
    )
    for key, value in python_telemetry.items():
        assert rust_telemetry[key] == value
    assert rust_telemetry["graph_fallback_native"] is True


def test_rust_arrow_orcid_repair_merges_whole_subblocks(tmp_path):
    _require_rust_arrow_subblocking()
    signatures_path = tmp_path / "signatures.arrow"
    _write_signatures_arrow(
        signatures_path,
        [
            ("s1", "bb", "", "0000-0000-0000-0001"),
            ("s2", "bb", "", "0000-0000-0000-0001"),
            ("s3", "aa", "", "0000-0000-0000-0001"),
            ("s4", "aa", "", None),
            ("s5", "aa", "", None),
            ("s6", "bb", "", None),
        ],
    )

    rust_subblocks, telemetry = subblocking._make_subblocks_with_telemetry_arrow_rust(
        {
            "signatures": str(signatures_path),
            "signatures_batch_index": _add_batch_index(
                signatures_path,
                tmp_path / "signatures.signatures_batch_index.bin",
                key_column="signature_id",
                table_name="signatures",
            ),
        },
        ["s1", "s2", "s3", "s4", "s5", "s6"],
        maximum_size=6,
        first_k_letter_counts_sorted={},
        graph_subblocking_config=subblocking.GraphSubblockingConfig(),
    )

    assert sorted(sorted(signature_ids) for signature_ids in rust_subblocks.values()) == [
        ["s1", "s2", "s3", "s4", "s5", "s6"],
    ]
    assert telemetry["orcid_subblocking_enabled"] is True


def test_rust_arrow_orcid_repair_does_not_extract_from_oversized_whole_merge(tmp_path):
    _require_rust_arrow_subblocking()
    signatures_path = tmp_path / "signatures.arrow"
    _write_signatures_arrow(
        signatures_path,
        [
            ("s1", "aa", "", "0000-0000-0000-0001"),
            ("s2", "aa", "", "0000-0000-0000-0001"),
            ("s3", "aa", "", None),
            ("s4", "bb", "", "0000-0000-0000-0001"),
            ("s5", "bb", "", None),
        ],
    )

    rust_subblocks, telemetry = subblocking._make_subblocks_with_telemetry_arrow_rust(
        {
            "signatures": str(signatures_path),
            "signatures_batch_index": _add_batch_index(
                signatures_path,
                tmp_path / "signatures.signatures_batch_index.bin",
                key_column="signature_id",
                table_name="signatures",
            ),
        },
        ["s1", "s2", "s3", "s4", "s5"],
        maximum_size=4,
        first_k_letter_counts_sorted={},
        graph_subblocking_config=subblocking.GraphSubblockingConfig(),
    )

    assert sorted(sorted(signature_ids) for signature_ids in rust_subblocks.values()) == [
        ["s1", "s2", "s3"],
        ["s4", "s5"],
    ]
    assert telemetry["orcid_merge_skipped_due_to_capacity_count"] == 1
    assert telemetry["orcid_merge_skipped_due_to_capacity_signature_count"] == 3


def test_rust_arrow_native_graph_subblocking_uses_arrow_evidence_without_python_callback(tmp_path):
    _require_rust_arrow_subblocking()
    signatures_path = tmp_path / "signatures.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    specter_path = tmp_path / "specter.arrow"
    _write_signatures_arrow(
        signatures_path,
        [
            ("s1", "hui", "", None),
            ("s2", "hui", "", None),
            ("s3", "hui", "", None),
            ("s4", "hui", "", None),
        ],
    )
    _write_ipc(
        paper_authors_path,
        pa.table(
            {
                "paper_id": pa.array(["p_s1", "p_s1", "p_s2", "p_s2", "p_s3", "p_s3", "p_s4", "p_s4"]),
                "position": pa.array([0, 1, 0, 1, 0, 1, 0, 1], type=pa.int64()),
                "author_name": pa.array(
                    [
                        "Hui Wang",
                        "Ada Lovelace",
                        "Hui Wang",
                        "Ada Lovelace",
                        "Hui Wang",
                        "Grace Hopper",
                        "Hui Wang",
                        "Grace Hopper",
                    ],
                    type=pa.string(),
                ),
            }
        ),
    )
    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]], dtype=np.float32)
    _write_ipc(
        specter_path,
        pa.table(
            {
                "paper_id": pa.array(["p_s1", "p_s2", "p_s3", "p_s4"], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(pa.array(np.ravel(embeddings), type=pa.float32()), 2),
            }
        ),
    )
    paths = {
        "signatures": str(signatures_path),
        "signatures_batch_index": _add_batch_index(
            signatures_path,
            tmp_path / "signatures.signatures_batch_index.bin",
            key_column="signature_id",
            table_name="signatures",
        ),
        "paper_authors": str(paper_authors_path),
        "paper_authors_batch_index": _add_batch_index(
            paper_authors_path,
            tmp_path / "paper_authors.paper_authors_batch_index.bin",
            key_column="paper_id",
            table_name="paper_authors",
        ),
        "specter": str(specter_path),
        "specter_batch_index": _add_batch_index(
            specter_path,
            tmp_path / "specter.specter_batch_index.bin",
            key_column="paper_id",
            table_name="specter",
        ),
    }

    def fail_python_callback(*_args, **_kwargs):
        raise AssertionError("native graph Arrow subblocking should not call Python fallback")

    subblocks, telemetry = subblocking._make_subblocks_with_telemetry_arrow_rust(
        paths,
        ["s1", "s2", "s3", "s4"],
        maximum_size=2,
        first_k_letter_counts_sorted={},
        graph_subblocking_config=subblocking.GraphSubblockingConfig(
            neighbor_mode="exact",
            neighbors=1,
            min_edge_score=0.8,
        ),
        graph_subblocking_random_seed=7,
    )
    assert callable(fail_python_callback)

    assert {frozenset(values) for values in subblocks.values()} == {frozenset({"s1", "s2"}), frozenset({"s3", "s4"})}
    assert telemetry["specter_invocation_count"] == 1
    assert telemetry["graph_fallback_native"] is True
    assert telemetry["graph_fallback_invocation_count"] == 1
    assert telemetry["graph_fallback_load_metrics"]["paper_authors_rows_loaded"] == 8
    assert telemetry["graph_fallback_stats"][0]["packed_component_count"] == 2


def test_rust_arrow_native_graph_subblocking_rejects_null_author_position(tmp_path):
    _require_rust_arrow_subblocking()
    signatures_path = tmp_path / "signatures.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    specter_path = tmp_path / "specter.arrow"
    _write_signatures_arrow(
        signatures_path,
        [
            ("s1", "hui", "", None),
            ("s2", "hui", "", None),
            ("s3", "hui", "", None),
            ("s4", "hui", "", None),
        ],
        author_positions=[None, 0, 0, 0],
    )
    _write_ipc(
        paper_authors_path,
        pa.table(
            {
                "paper_id": pa.array(["p_s1", "p_s1", "p_s2", "p_s2", "p_s3", "p_s3", "p_s4", "p_s4"]),
                "position": pa.array([0, 1, 0, 1, 0, 1, 0, 1], type=pa.int64()),
                "author_name": pa.array(
                    [
                        "Hui Wang",
                        "Ada Lovelace",
                        "Hui Wang",
                        "Ada Lovelace",
                        "Hui Wang",
                        "Grace Hopper",
                        "Hui Wang",
                        "Grace Hopper",
                    ],
                    type=pa.string(),
                ),
            }
        ),
    )
    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]], dtype=np.float32)
    _write_ipc(
        specter_path,
        pa.table(
            {
                "paper_id": pa.array(["p_s1", "p_s2", "p_s3", "p_s4"], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(pa.array(np.ravel(embeddings), type=pa.float32()), 2),
            }
        ),
    )
    paths = {
        "signatures": str(signatures_path),
        "signatures_batch_index": _add_batch_index(
            signatures_path,
            tmp_path / "signatures.signatures_batch_index.bin",
            key_column="signature_id",
            table_name="signatures",
        ),
        "paper_authors": str(paper_authors_path),
        "paper_authors_batch_index": _add_batch_index(
            paper_authors_path,
            tmp_path / "paper_authors.paper_authors_batch_index.bin",
            key_column="paper_id",
            table_name="paper_authors",
        ),
        "specter": str(specter_path),
        "specter_batch_index": _add_batch_index(
            specter_path,
            tmp_path / "specter.specter_batch_index.bin",
            key_column="paper_id",
            table_name="specter",
        ),
    }

    with pytest.raises(ValueError, match="author_position is null"):
        subblocking._make_subblocks_with_telemetry_arrow_rust(
            paths,
            ["s1", "s2", "s3", "s4"],
            maximum_size=2,
            first_k_letter_counts_sorted={},
            graph_subblocking_config=subblocking.GraphSubblockingConfig(
                neighbor_mode="exact",
                neighbors=1,
                min_edge_score=0.8,
            ),
        )
