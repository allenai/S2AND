from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

import s2and.subblocking as subblocking
from s2and.arrow_inputs import ArrowDataset
from s2and.data import Signature
from s2and.incremental_linking.feature_block import write_arrow_batch_lookup_index
from s2and.runtime import load_s2and_rust_extension
from tests.helpers import write_test_arrow_artifact_manifest
from tests.raw_arrow_helpers import write_ipc as _write_ipc


class _SingleBatchArrowLease:
    def __init__(self, path: str) -> None:
        self.path = path
        self.native = self

    def batch_indices(self, _table_key: str, _values: list[str]) -> list[int]:
        return [0]

    def has(self, _table_key: str) -> bool:
        return True

    @contextmanager
    def open_file(self, _table_key: str):
        with Path(self.path).open("rb") as source:
            yield source


def _signature(signature_id: str, *, first: str, middle: str | None = None, orcid: str | None = None) -> Signature:
    return Signature(
        author_info_first=first,
        author_info_first_normalized=first,
        author_info_middle=middle,
        author_info_middle_normalized=middle or "",
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


def test_make_subblocks_rejects_duplicate_ids_after_string_coercion() -> None:
    dataset = SimpleNamespace(signatures={"1": _signature("1", first="anna")}, random_seed=0)

    with pytest.raises(ValueError, match="must be unique.*'1'"):
        subblocking.make_subblocks_with_telemetry(
            [1, "1"],
            dataset,
            maximum_size=2,
            first_k_letter_counts_sorted={},
        )


def test_make_subblocks_preserves_integer_signature_ids() -> None:
    dataset = SimpleNamespace(signatures={1: _signature("1", first="anna")}, random_seed=0)

    subblocks, _telemetry = subblocking.make_subblocks_with_telemetry(
        [1],
        dataset,
        maximum_size=2,
        first_k_letter_counts_sorted={},
    )

    output_signature_ids = [signature_id for subblock in subblocks.values() for signature_id in subblock]
    assert output_signature_ids == [1]
    assert type(output_signature_ids[0]) is int


def test_rust_arrow_make_subblocks_rejects_duplicate_ids_after_string_coercion(tmp_path) -> None:
    load_s2and_rust_extension()
    signatures_path = tmp_path / "signatures.arrow"
    _write_signatures_arrow(signatures_path, [("1", "anna", "", None)])
    paths = {
        "signatures": str(signatures_path),
        "signatures_batch_index": _add_batch_index(
            signatures_path,
            tmp_path / "signatures.signatures_batch_index.bin",
            key_column="signature_id",
            table_name="signatures",
        ),
    }
    arrow_dataset = _open_subblocking_arrow_dataset(paths, tmp_path)

    with pytest.raises(ValueError, match="must be unique.*'1'"):
        subblocking._make_subblocks_with_telemetry_arrow_rust(
            arrow_dataset,
            [1, "1"],
            maximum_size=2,
            first_k_letter_counts_sorted={},
            graph_subblocking_config=subblocking.GraphSubblockingConfig(),
        )

    rust_module = load_s2and_rust_extension()
    with pytest.raises(ValueError, match='must be unique.*"1"'):
        rust_module.make_subblocks_with_telemetry_arrow_native_graph(
            arrow_dataset.native,
            ["1", "1"],
            2,
            {},
            subblocking.GraphSubblockingConfig(),
            0,
            True,
        )
    arrow_dataset.close()


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


def test_subdivide_helper_accepts_prefix_exactly_at_capacity() -> None:
    names = np.array(["anna", "anna", "bill"])
    signature_ids = np.array(["s1", "s2", "s3"])

    output, dead_ends = subblocking.subdivide_helper(names, signature_ids, maximum_size=2, starting_k=2)

    assert dead_ends == {}
    assert {key: sorted(values.tolist()) for key, values in output.items()} == {
        "an": ["s1", "s2"],
        "bi": ["s3"],
    }


@pytest.mark.parametrize(
    ("names", "signature_ids", "expected_output", "expected_dead_ends"),
    [
        (["ann", "ann", "anna"], ["s1", "s2", "s3"], {"anna": ["s3"]}, {"ann": ["s1", "s2"]}),
        (
            ["anna", "anna", "anne", "anne"],
            ["s1", "s2", "s3", "s4"],
            {},
            {"anna": ["s1", "s2"], "anne": ["s3", "s4"]},
        ),
    ],
    ids=["terminal-prefix-collision", "distinct-full-names"],
)
def test_subdivide_helper_keeps_terminal_full_name_groups_separate(
    names: list[str],
    signature_ids: list[str],
    expected_output: dict[str, list[str]],
    expected_dead_ends: dict[str, list[str]],
) -> None:
    output, dead_ends = subblocking.subdivide_helper(
        np.array(names), np.array(signature_ids), maximum_size=1, starting_k=2
    )

    assert {key: values.tolist() for key, values in output.items()} == expected_output
    assert {key: values.tolist() for key, values in dead_ends.items()} == expected_dead_ends
    assert sorted(value for values in (*output.values(), *dead_ends.values()) for value in values) == signature_ids


def test_make_subblocks_does_not_fallback_when_longer_name_prefixes_fit(monkeypatch) -> None:
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="anna"),
            "s2": _signature("s2", first="anne"),
            "s3": _signature("s3", first="anny"),
        },
        random_seed=0,
    )

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("name-prefix subdivision should avoid fallback")

    monkeypatch.setattr(subblocking, "cluster_with_specter", _fail_if_called)

    subblocks, telemetry = subblocking.make_subblocks_with_telemetry(
        ["s1", "s2", "s3"],
        dataset,
        maximum_size=1,
        first_k_letter_counts_sorted={},
    )

    assert sorted(subblocks.values()) == [["s1"], ["s2"], ["s3"]]
    assert telemetry["first_name_dead_end_signature_count"] == 0
    assert telemetry["specter_invocation_count"] == 0


def test_cluster_with_specter_surfaces_unexpected_auxiliary_feature_failures(monkeypatch) -> None:
    first_signature = _signature("s1", first="anna")._replace(
        author_info_coauthor_blocks=["a smith"],
        author_info_affiliations=["alpha laboratory"],
    )
    second_signature = _signature("s2", first="anne")._replace(
        author_info_coauthor_blocks=["b smith"],
        author_info_affiliations=["beta laboratory"],
    )
    dataset = SimpleNamespace(
        signatures={
            "s1": first_signature,
            "s2": second_signature,
        },
        papers={},
        specter_embeddings={},
        random_seed=0,
    )

    def _raise_unexpected(_self, _values):
        raise RuntimeError("unexpected auxiliary failure")

    monkeypatch.setattr(subblocking.MultiLabelBinarizer, "fit_transform", _raise_unexpected)

    with pytest.raises(RuntimeError, match="unexpected auxiliary failure"):
        subblocking.cluster_with_specter(["s1", "s2"], dataset, target_subblock_size=1)


def test_cluster_with_specter_handles_degenerate_auxiliary_fallback() -> None:
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="anna"),
            "s2": _signature("s2", first="anne"),
        },
        papers={},
        specter_embeddings={},
        random_seed=0,
    )

    output = subblocking.cluster_with_specter(["s1", "s2"], dataset, target_subblock_size=1)

    assert sorted(output.values()) == [["s1"], ["s2"]]


def test_cluster_with_specter_returns_capacity_sized_input_without_clustering(monkeypatch) -> None:
    dataset = SimpleNamespace(signatures={}, papers={}, specter_embeddings={}, random_seed=0)

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("Genie should not run when the input already fits")

    monkeypatch.setattr("genieclust.Genie", _raise_if_called)

    output = subblocking.cluster_with_specter(["s1"], dataset, target_subblock_size=1)

    assert output == {"0": ["s1"]}


def test_cluster_with_specter_clusters_nonzero_embeddings() -> None:
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="anna"),
            "s2": _signature("s2", first="anne"),
        },
        papers={},
        specter_embeddings={
            "1": np.ones(768, dtype=np.float64),
            "2": np.concatenate((np.zeros(767, dtype=np.float64), np.ones(1, dtype=np.float64))),
        },
        random_seed=0,
    )

    output = subblocking.cluster_with_specter(["s1", "s2"], dataset, target_subblock_size=1)

    assert sorted(output.values()) == [["s1"], ["s2"]]


def test_cluster_with_specter_caps_requested_clusters_below_sample_count(monkeypatch) -> None:
    requested_clusters: list[int] = []

    class StrictGenie:
        def __init__(self, *, n_clusters: int, gini_threshold: float) -> None:
            requested_clusters.append(n_clusters)
            assert gini_threshold == 0.01

        def fit_predict(self, values: np.ndarray) -> np.ndarray:
            return np.zeros(values.shape[0], dtype=int)

    monkeypatch.setattr("genieclust.Genie", StrictGenie)
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="anna"),
            "s2": _signature("s2", first="anne"),
        },
        papers={},
        specter_embeddings={
            "1": np.ones(768, dtype=np.float64),
            "2": np.ones(768, dtype=np.float64),
        },
        random_seed=0,
    )

    output = subblocking.cluster_with_specter(["s1", "s2"], dataset, target_subblock_size=1)

    assert requested_clusters == [1]
    assert sorted(output.values()) == [["s1"], ["s2"]]


def test_union_find_find_compresses_long_parent_chain_without_recursion() -> None:
    union_find = subblocking._UnionFind(1500)  # noqa: SLF001
    union_find.parent = list(range(1, 1500)) + [1499]

    assert union_find.find(0) == 1499
    assert union_find.parent[0] == 1499


@pytest.mark.parametrize("bridge", [(0, 2), (2, 0)])
def test_union_find_capacity_counts_existing_component_in_either_argument_order(bridge) -> None:
    union_find = subblocking._UnionFind(4)
    assert union_find.union_if_capacity(0, 1, maximum_size=3)
    assert union_find.union_if_capacity(*bridge, maximum_size=3)
    assert not union_find.union_if_capacity(2, 3, maximum_size=3)
    assert union_find.find(0) == union_find.find(1) == union_find.find(2)
    assert union_find.find(2) != union_find.find(3)


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
    # dash code point.
    signature = SimpleNamespace(
        author_info_first="Arif\u2010ullah",
        author_info_middle=None,
        author_info_last="Khan",
        author_info_first_normalized=None,
        author_info_middle_normalized=None,
    )

    assert subblocking.signature_name_parts_for_subblocking(signature) == ("arif ullah", "")
    # D9 needs the raw last name to read joined capitals as initials.
    for last, expected in (("Wells", ("h", "g")), ("WELLS", ("hg", ""))):
        signature.author_info_first, signature.author_info_last = "HG", last
        assert subblocking.signature_name_parts_for_subblocking(signature) == expected


def test_signature_name_parts_for_subblocking_treats_all_dashes_uniformly() -> None:
    unicode_dash = SimpleNamespace(
        author_info_first="Sang\u2010Min",
        author_info_middle=None,
        author_info_first_normalized="sang min",
        author_info_middle_normalized="",
    )
    ascii_dash = SimpleNamespace(
        author_info_first="Sang-Min",
        author_info_middle=None,
        author_info_first_normalized="sang min",
        author_info_middle_normalized="",
    )

    assert subblocking.signature_name_parts_for_subblocking(unicode_dash) == ("sang min", "")
    assert subblocking.signature_name_parts_for_subblocking(ascii_dash) == ("sang min", "")


def _key_names(targets, context, maximum_size: int) -> list[tuple[str, str]]:
    """Key names of the target rows after the rule runs on targets plus repeated context rows."""
    names = list(targets) + [(first, middle) for first, middle, count in context for _ in range(count)]
    return subblocking.subblocking_name_parts(names, maximum_size)[: len(targets)]


def test_subblocking_name_parts_key_all_initial_compounds_as_initials() -> None:
    # The Rust twin is all_initial_compounds_key_as_initials_even_with_joined_evidence.
    assert _key_names([("j d", ""), ("j d", "t"), ("j d t", "")], [("jd", "", 50)], 100) == [
        ("j", "d"),
        ("j", "d t"),
        ("j", "d t"),
    ]


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        ([("huimin", "", 7)], ("hui", "min x")),
        ([("huimin", "", 8)], ("huimin", "x")),
        # Split-form records ("hui" with a middle that starts with "min") raise the bar.
        ([("huimin", "", 15), ("hui", "min jie", 1)], ("hui", "min x")),
        # Neither a longer middle token nor another compound counts as split form.
        ([("huimin", "", 16), ("hui", "minh", 1), ("hui min", "", 3)], ("huimin", "x")),
    ],
)
def test_subblocking_name_parts_join_only_above_the_ratio(context, expected) -> None:
    # (n_join + 1) / (n_split + 1) must be above 8. The Rust twin is compound_joins_only_above_the_ratio.
    assert _key_names([("hui min", "x")], context, 100) == [expected]


@pytest.mark.parametrize(("joined_count", "expected"), [(20, ("qian", "")), (21, ("qi", "an"))])
def test_subblocking_name_parts_do_not_join_a_first_name_above_the_cap(joined_count, expected) -> None:
    assert _key_names([("qi an", "")], [("qian", "", joined_count)], 20) == [expected]


def test_subblocking_name_parts_join_all_tokens_or_split_after_the_first() -> None:
    targets = [("xiao yu ting", "m"), ("xiao yu ting", "")]
    assert _key_names(targets, [("xiaoyuting", "", 9)], 100) == [("xiaoyuting", "m"), ("xiaoyuting", "")]
    assert _key_names(targets[:1], [("xiaoyuting", "", 9), ("xiao", "yu", 1)], 100) == [("xiao", "yu ting m")]


@pytest.mark.parametrize(
    ("rows", "maximum_size", "expected"),
    [
        pytest.param(
            [("s0", "J.-D.", "T."), ("s1", "J.", ""), ("s2", "Jian", ""), ("s3", "Jian", ""), ("s4", "Jing", "")],
            2,
            {"j": ["s0", "s1"], "jia": ["s2", "s3"], "jin": ["s4"]},
            id="all-initial-compound-joins-initials",
        ),
        pytest.param(
            [(f"s{i}", "Huimin", "") for i in range(8)] + [("s8", "Hui-Min", ""), ("s9", "Hui", "Ling")],
            9,
            {"huim": [f"s{i}" for i in range(9)], "hui": ["s9"]},
            id="joined-majority",
        ),
        pytest.param(
            [(f"s{i}", "Huimin", "") for i in range(8)] + [("s8", "Hui-Min", ""), ("s9", "Hui", "Min")],
            9,
            {"huim": [f"s{i}" for i in range(8)], "hui": ["s8", "s9"]},
            id="split-form-evidence",
        ),
        pytest.param(
            [(f"s{i}", "Xiaoyuting", "") for i in range(8)] + [("s8", "Xiao-Yu-Ting", ""), ("s9", "Xiao", "")],
            9,
            {"xiaoy": [f"s{i}" for i in range(9)], "xiao": ["s9"]},
            id="three-token-join",
        ),
        pytest.param(
            [(f"s{i}", "Xiaoyuting", "") for i in range(8)]
            + [("s8", "Xiao-Yu-Ting", ""), ("s9", "Xiao", ""), ("s10", "Xiao", "Yu")],
            9,
            {"xiaoy": [f"s{i}" for i in range(8)], "xiao": ["s10", "s8", "s9"]},
            id="three-token-split",
        ),
        pytest.param(
            [(f"s{i}", "Huimin", chr(ord("A") + i)) for i in range(10)] + [("s10", "Hui-Min", "")],
            9,
            None,
            id="joined-form-above-cap",
        ),
    ],
)
def test_python_and_arrow_subblock_compound_first_names_identically(tmp_path, rows, maximum_size, expected) -> None:
    """Both planners key compounds the same way from the same raw names."""
    load_s2and_rust_extension()
    signatures_path = tmp_path / "signatures.arrow"
    _write_signatures_arrow(signatures_path, [(*row, None) for row in rows])
    signatures = {}
    for sid, first, middle in rows:
        # _write_signatures_arrow stores the last name "wang", and canonicalization reads it.
        parts = subblocking.canonicalize_name_parts(first, middle, "wang")
        signatures[sid] = _signature(sid, first=parts.first, middle=parts.middle)
    dataset = SimpleNamespace(signatures=signatures, random_seed=0)
    python_subblocks, _ = subblocking.make_subblocks_with_telemetry(
        list(signatures), dataset, maximum_size, first_k_letter_counts_sorted={}, use_orcid_subblocking=False
    )
    with _open_subblocking_arrow_dataset({"signatures": str(signatures_path)}, tmp_path) as arrow_dataset:
        native_subblocks, _, _ = subblocking._make_subblocks_with_telemetry_arrow_rust(
            arrow_dataset,
            list(signatures),
            maximum_size,
            first_k_letter_counts_sorted={},
            graph_subblocking_config=subblocking.GraphSubblockingConfig(),
            use_orcid_subblocking=False,
        )

    python_plan = {key: sorted(values) for key, values in python_subblocks.items()}
    assert python_plan == {key: sorted(values) for key, values in native_subblocks.items()}
    if expected is not None:
        assert python_plan == expected
    else:
        # Ten "huimin" records exceed the cap of nine, so the compound keys as "hui" and not as "huimin".
        compound_key = next(key for key, values in python_plan.items() if "s10" in values)
        assert compound_key.split(", ")[0] == "hui"


def _rows(*groups: tuple[str, str, int]) -> list[tuple[str, str, str]]:
    names = [(first, middle) for first, middle, count in groups for _ in range(count)]
    return [(f"s{index}", first, middle) for index, (first, middle) in enumerate(names)]


@pytest.mark.parametrize(
    ("rows", "maximum_size", "expected"),
    [
        pytest.param(
            _rows(("Wei", "", 3), ("Wei", "Min", 1), ("Wen", "", 3), ("W.", "", 2), ("W.", "A", 2), ("Wei-Lan", "", 1)),
            3,
            {"wei": ["s0", "s1", "s2"]},
            # "wen" is at the cap, "w" is an initial over the cap, and "Wei-Lan" keys as "wei" with middle "lan".
            id="dead-end-at-and-over-cap",
        ),
        pytest.param(
            _rows(("Weimin", "", 5), ("Weimin", "A", 1), ("Weimin", "B", 1), ("Weimin", "C", 1), ("Weimin", "D", 1))
            + [("s9", "Wei-Min", "")],
            9,
            {"weimin": ["s0", "s1", "s2", "s3", "s4", "s9"]},
            id="joined-compound-in-dead-end",
        ),
        pytest.param(
            _rows(("Wei", "", 3), ("Wei", "Min", 1), ("Hao", "", 2), ("Hao", "Xu", 2)),
            3,
            {"wei": ["s0", "s1", "s2"], "hao": ["s4", "s5"]},
            id="two-dead-ends",
        ),
        pytest.param(
            _rows(("J.-D.", "", 1), ("J", "", 2), ("JD", "", 1), ("Jian", "", 1)),
            2,
            {},
            id="initials-and-joined-capitals-are-not-dead-ends",
        ),
    ],
)
def test_python_and_arrow_report_the_same_no_middle_dead_ends(tmp_path, rows, maximum_size, expected) -> None:
    """Both planners report the no-middle bucket of each first-name dead end, from the same raw names."""
    load_s2and_rust_extension()
    signatures_path = tmp_path / "signatures.arrow"
    _write_signatures_arrow(signatures_path, [(*row, None) for row in rows])
    names = {sid: subblocking.canonicalize_name_parts(first, middle, "wang") for sid, first, middle in rows}
    signature_ids = list(names)
    python_dead_ends = subblocking.no_middle_dead_end_buckets(
        signature_ids, [(parts.first, parts.middle) for parts in names.values()], maximum_size
    )
    dataset = SimpleNamespace(
        signatures={sid: _signature(sid, first=parts.first, middle=parts.middle) for sid, parts in names.items()},
        random_seed=0,
    )
    python_subblocks, _ = subblocking.make_subblocks_with_telemetry(
        signature_ids, dataset, maximum_size, first_k_letter_counts_sorted={}, use_orcid_subblocking=False
    )
    with _open_subblocking_arrow_dataset({"signatures": str(signatures_path)}, tmp_path) as arrow_dataset:
        native_subblocks, _, native_dead_ends = subblocking._make_subblocks_with_telemetry_arrow_rust(
            arrow_dataset,
            signature_ids,
            maximum_size,
            first_k_letter_counts_sorted={},
            graph_subblocking_config=subblocking.GraphSubblockingConfig(),
            use_orcid_subblocking=False,
        )

    assert python_dead_ends == expected
    assert {first: sorted(bucket) for first, bucket in native_dead_ends.items()} == expected
    # Each reported record sits in the no-middle bucket of its dead end in both plans.
    for plan in (python_subblocks, native_subblocks):
        bucket_of = {sid: key for key, members in plan.items() for sid in members}
        for first, bucket in expected.items():
            assert all(f"{first}|middle=" in bucket_of[sid].split(", ") for sid in bucket)


def test_coauthor_blocks_from_rowwise_arrow_normalizes_author_names(tmp_path) -> None:
    paper_authors = pa.table(
        {
            "paper_id": pa.array(["p1", "p1", "p1", "p1"], type=pa.string()),
            "position": pa.array([0, 1, 2, 3], type=pa.int64()),
            "author_name": pa.array(["O'Connor", "", "   ", "Maciej Górski"], type=pa.string()),
        }
    )

    paper_authors_path = _write_ipc(tmp_path / "paper_authors.arrow", paper_authors)
    load_metrics: dict[str, int] = {}
    out = subblocking._coauthor_blocks_by_paper_from_arrow(  # noqa: SLF001
        _SingleBatchArrowLease(paper_authors_path),
        ["p1"],
        load_metrics=load_metrics,
    )

    assert out == {"p1": [(0, "o connor"), (3, "m gorski")]}
    assert load_metrics["paper_authors_rows_loaded"] == 4


def test_coauthor_blocks_from_rowwise_arrow_validates_blank_row_positions(tmp_path) -> None:
    paper_authors = pa.table(
        {
            "paper_id": pa.array(["p1", "p1"], type=pa.string()),
            "position": pa.array([0, 0], type=pa.int64()),
            "author_name": pa.array(["", "   "], type=pa.string()),
        }
    )
    paper_authors_path = _write_ipc(tmp_path / "paper_authors.arrow", paper_authors)
    with pytest.raises(ValueError, match=r"duplicate \(paper_id, position\)"):
        subblocking._coauthor_blocks_by_paper_from_arrow(  # noqa: SLF001
            _SingleBatchArrowLease(paper_authors_path),
            ["p1"],
            load_metrics={},
        )


def test_subblock_merge_candidate_metadata_preserves_middle_values_with_equals() -> None:
    assert subblocking._subblock_merge_candidate_metadata("a|middle=b=c", 2) == (2, "a", "b=c", "b=c", "b=c")


def test_terminal_full_name_is_used_for_merge_scoring() -> None:
    candidates = subblocking._sorted_subblock_merge_candidates(
        {"chen|middle=w": ["s1"], "cheng": ["s2"]},
        maximum_size=2,
        first_k_letter_counts_sorted={},
    )

    assert candidates == [(("chen|middle=w", "cheng"), 100004.0)]


@pytest.mark.parametrize(
    ("names", "maximum_size", "enabled", "expected", "skipped"),
    [
        (
            [
                ("aa", "https://orcid.org/0000-0002-1825-0097"),
                ("bb", "ORCID: 0000000218250097"),
                ("aa", "   "),
                ("cc", "   "),
            ],
            3,
            True,
            [["s1", "s2", "s3"], ["s4"]],
            (0, 0),
        ),
        (
            [
                ("bb", "0000-0000-0000-0001"),
                ("bb", "0000-0000-0000-0001"),
                ("aa", "0000-0000-0000-0001"),
                ("aa", None),
                ("aa", None),
                ("bb", None),
            ],
            6,
            True,
            [["s1", "s2", "s3", "s4", "s5", "s6"]],
            (0, 0),
        ),
        (
            [
                ("aa", "0000-0000-0000-0001"),
                ("aa", "0000-0000-0000-0001"),
                ("aa", None),
                ("bb", "0000-0000-0000-0001"),
                ("bb", None),
            ],
            4,
            True,
            [["s1", "s2", "s3"], ["s4", "s5"]],
            (1, 3),
        ),
        (
            [("aa", "https://orcid.org/0000-0002-1825-0097"), ("bb", "ORCID: 0000000218250097"), ("cc", None)],
            3,
            False,
            [["s1"], ["s2"], ["s3"]],
            (0, 0),
        ),
        (
            [
                ("aa", "0000-0000-0000-0001"),
                ("bb", "0000-0000-0000-0001"),
                ("bb", "0000-0000-0000-0002"),
                ("cc", "0000-0000-0000-0002"),
            ],
            3,
            True,
            [["s1"], ["s2", "s3"], ["s4"]],
            (2, 4),
        ),
    ],
    ids=["normalized-orcid", "whole-subblocks", "over-capacity", "disabled", "transitive-over-capacity"],
)
def test_python_and_arrow_orcid_repair_preserve_whole_subblocks(
    tmp_path: Path, names, maximum_size: int, enabled: bool, expected, skipped
) -> None:
    """Both real implementations obey the same ORCID and capacity contract."""
    rows = [(f"s{index}", first, "", orcid) for index, (first, orcid) in enumerate(names, start=1)]
    signatures_path = tmp_path / "signatures.arrow"
    _write_signatures_arrow(signatures_path, rows)
    dataset = SimpleNamespace(
        signatures={
            sid: _signature(sid, first=first, middle=middle, orcid=orcid) for sid, first, middle, orcid in rows
        },
        random_seed=0,
    )
    kwargs = dict(maximum_size=maximum_size, first_k_letter_counts_sorted={}, use_orcid_subblocking=enabled)
    python_result = subblocking.make_subblocks_with_telemetry(list(dataset.signatures), dataset, **kwargs)
    with _open_subblocking_arrow_dataset({"signatures": str(signatures_path)}, tmp_path) as arrow_dataset:
        native_result = subblocking._make_subblocks_with_telemetry_arrow_rust(
            arrow_dataset,
            list(dataset.signatures),
            **kwargs,
            graph_subblocking_config=subblocking.GraphSubblockingConfig(),
        )[:2]
    for subblocks, telemetry in (python_result, native_result):
        assert sorted(sorted(members) for members in subblocks.values()) == expected
        assert telemetry["orcid_subblocking_enabled"] is enabled
        assert telemetry["orcid_merge_skipped_due_to_capacity_count"] == skipped[0]
        assert telemetry["orcid_merge_skipped_due_to_capacity_signature_count"] == skipped[1]


def _write_signatures_arrow(
    path,
    rows: list[tuple[str, str, str, str | None]],
) -> None:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    table = pa.table(
        {
            "signature_id": pa.array([row[0] for row in rows], type=pa.string()),
            "paper_id": pa.array([f"p_{row[0]}" for row in rows], type=pa.string()),
            "author_first": pa.array([row[1] for row in rows], type=pa.string()),
            "author_middle": pa.array([row[2] for row in rows], type=pa.string()),
            "author_last": pa.array(["wang"] * len(rows), type=pa.string()),
            "author_suffix": pa.array([""] * len(rows), type=pa.string()),
            "author_block": pa.array([f"{row[1][:1]} wang" for row in rows], type=pa.string()),
            "author_affiliations": pa.array([[] for _row in rows], type=pa.list_(pa.string())),
            "author_orcid": pa.array([row[3] for row in rows], type=pa.string()),
            "author_position": pa.array([0] * len(rows), type=pa.int64()),
        }
    )
    with path.open("wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def _add_batch_index(path, index_path, *, key_column: str, table_name: str) -> str:
    write_arrow_batch_lookup_index(path, index_path, key_column=key_column, table_name=table_name)
    return str(index_path)


def _open_subblocking_arrow_dataset(paths: dict[str, str], tmp_path: Path) -> ArrowDataset:
    paths = dict(paths)
    with pa.memory_map(paths["signatures"], "r") as source:
        paper_ids = [str(value) for value in pa.ipc.open_file(source).read_all()["paper_id"].to_pylist()]
    papers_path = _write_ipc(
        tmp_path / "papers.arrow",
        pa.table(
            {
                "paper_id": pa.array(paper_ids, type=pa.string()),
                "title": pa.array([""] * len(paper_ids), type=pa.string()),
                "venue": pa.array([""] * len(paper_ids), type=pa.string()),
                "journal_name": pa.array([""] * len(paper_ids), type=pa.string()),
            }
        ),
    )
    paths["papers"] = papers_path
    if "paper_authors" not in paths:
        paths["paper_authors"] = _write_ipc(
            tmp_path / "paper_authors.arrow",
            pa.table(
                {
                    "paper_id": pa.array(paper_ids, type=pa.string()),
                    "position": pa.array([0] * len(paper_ids), type=pa.int64()),
                    "author_name": pa.array([""] * len(paper_ids), type=pa.string()),
                }
            ),
        )
    for table_name, key_column in (
        ("signatures", "signature_id"),
        ("papers", "paper_id"),
        ("paper_authors", "paper_id"),
        ("specter", "paper_id"),
    ):
        if table_name not in paths:
            continue
        index_key = f"{table_name}_batch_index"
        if index_key not in paths:
            paths[index_key] = _add_batch_index(
                paths[table_name],
                tmp_path / f"{table_name}.{index_key}.bin",
                key_column=key_column,
                table_name=table_name,
            )
    write_test_arrow_artifact_manifest(tmp_path, paths)
    return ArrowDataset.open(tmp_path, require_specter="specter" in paths)


def test_rust_arrow_prefix_subdivision_matches_python(tmp_path) -> None:
    load_s2and_rust_extension()
    cases = (
        (
            "first-prefix",
            [("s1", "anna", ""), ("s2", "anne", ""), ("s3", "anny", "")],
            {"anna": ["s1"], "anne": ["s2"], "anny": ["s3"]},
        ),
        (
            "middle-prefix",
            [("s1", "h", "will"), ("s2", "h", "william"), ("s3", "h", "wim")],
            {
                "h|middle=will": ["s1"],
                "h|middle=willi": ["s2"],
                "h|middle=wim": ["s3"],
            },
        ),
        (
            "first-and-middle-prefix",
            [("s1", "anna", "w"), ("s2", "anna", "x"), ("s3", "anne", "y"), ("s4", "anne", "z")],
            {
                "anna|middle=w": ["s1"],
                "anna|middle=x": ["s2"],
                "anne|middle=y": ["s3"],
                "anne|middle=z": ["s4"],
            },
        ),
    )
    for case_id, rows, expected in cases:
        case_root = tmp_path / case_id
        case_root.mkdir()
        signatures_path = case_root / "signatures.arrow"
        _write_signatures_arrow(signatures_path, [(*row, None) for row in rows])
        signature_ids = [row[0] for row in rows]
        dataset = SimpleNamespace(
            signatures={
                signature_id: _signature(signature_id, first=first, middle=middle)
                for signature_id, first, middle in rows
            },
            random_seed=0,
        )

        python_subblocks, _python_telemetry = subblocking.make_subblocks_with_telemetry(
            signature_ids,
            dataset,
            maximum_size=1,
            first_k_letter_counts_sorted={},
        )
        arrow_dataset = _open_subblocking_arrow_dataset({"signatures": str(signatures_path)}, case_root)
        rust_subblocks, _rust_telemetry, _ = subblocking._make_subblocks_with_telemetry_arrow_rust(
            arrow_dataset,
            signature_ids,
            maximum_size=1,
            first_k_letter_counts_sorted={},
            graph_subblocking_config=subblocking.GraphSubblockingConfig(),
        )

        assert {key: sorted(values) for key, values in python_subblocks.items()} == expected, case_id
        assert {key: sorted(values) for key, values in rust_subblocks.items()} == expected, case_id
        arrow_dataset.close()


def test_rust_arrow_native_graph_subblocking_uses_arrow_evidence_and_records_telemetry(tmp_path):
    load_s2and_rust_extension()
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
                "paper_id": pa.array(
                    ["p_s1", "p_s1", "p_s1", "p_s2", "p_s2", "p_s2", "p_s3", "p_s3", "p_s3", "p_s4", "p_s4", "p_s4"]
                ),
                "position": pa.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], type=pa.int64()),
                "author_name": pa.array(
                    [
                        "Hui Wang",
                        "Ada Lovelace",
                        "",
                        "Hui Wang",
                        "Ada Lovelace",
                        "   ",
                        "Hui Wang",
                        "Grace Hopper",
                        "",
                        "Hui Wang",
                        "Grace Hopper",
                        "\t",
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

    arrow_dataset = _open_subblocking_arrow_dataset(paths, tmp_path)
    subblocks, telemetry, no_middle_dead_ends = subblocking._make_subblocks_with_telemetry_arrow_rust(
        arrow_dataset,
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
    assert {frozenset(values) for values in subblocks.values()} == {frozenset({"s1", "s2"}), frozenset({"s3", "s4"})}
    assert telemetry["specter_invocation_count"] == 1
    # Four "hui" records without a middle name exceed the cap of two.
    assert no_middle_dead_ends == {"hui": ["s1", "s2", "s3", "s4"]}
    assert telemetry["graph_fallback_native"] is True
    assert telemetry["graph_fallback_invocation_count"] == 1
    assert telemetry["graph_fallback_load_metrics"]["paper_authors_rows_loaded"] == 12
    assert telemetry["graph_fallback_stats"][0]["packed_component_count"] == 2
    arrow_dataset.close()
