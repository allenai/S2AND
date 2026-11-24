from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

import s2and.subblocking as subblocking
from s2and.arrow_inputs import ArrowDataset
from s2and.incremental_linking.feature_block import write_arrow_batch_lookup_index
from s2and.production_bundle_contract import CALIBRATED_EPS_CALIBRATION
from s2and.production_model import _load_pairwise_staging_model, canonical_artifact_hashes
from s2and.runtime import load_s2and_rust_extension
from tests.helpers import write_test_arrow_artifact_manifest
from tests.promoted_linking_helpers import write_synthetic_pairwise_bundle
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


def _key_names(targets, context, maximum_size: int) -> list[tuple[str, str]]:
    """Key names of the target rows after the rule runs on targets plus repeated context rows."""
    names = list(targets) + [(first, middle) for first, middle, count in context for _ in range(count)]
    return subblocking.subblocking_name_parts(names, maximum_size)[: len(targets)]


def test_subblocking_name_parts_respect_join_ratio_and_capacity() -> None:
    targets = [("hui min", "x"), ("qi an", ""), ("xiao yu ting", "m"), ("j d", "t")]
    # Include split evidence that puts Huimin exactly at the required ratio.
    at_ratio = [("huimin", "", 15), ("hui", "min jie", 1), ("qian", "", 20), ("xiaoyuting", "", 9), ("jd", "", 10)]
    assert _key_names(targets, at_ratio, 20) == [("hui", "min x"), ("qian", ""), ("xiaoyuting", "m"), ("j", "d t")]

    # Ignore longer middle tokens and compounds when counting the split form.
    above_ratio = [("huimin", "", 16), ("hui", "minh", 1), ("hui min", "", 2)]
    above_ratio += [("qian", "", 21), ("xiaoyuting", "", 9), ("xiao", "yu", 1)]
    assert _key_names(targets[:3], above_ratio, 20) == [("huimin", "x"), ("qi", "an"), ("xiao", "yu ting m")]


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
            [(f"s{i}", "Huimin", chr(ord("A") + i)) for i in range(10)] + [("s10", "Hui-Min", "")],
            9,
            None,
            id="joined-form-above-cap",
        ),
    ],
)
def test_native_subblock_compound_first_names(tmp_path, rows, maximum_size, expected) -> None:
    """Group compound first names using the raw Arrow names."""
    load_s2and_rust_extension()
    signatures_path = tmp_path / "signatures.arrow"
    _write_signatures_arrow(signatures_path, [(*row, None) for row in rows])
    signature_ids = [row[0] for row in rows]
    with _open_subblocking_arrow_dataset({"signatures": str(signatures_path)}, tmp_path) as arrow_dataset:
        native_subblocks, _, _ = subblocking._make_subblocks_with_telemetry_arrow_rust(
            arrow_dataset,
            signature_ids,
            maximum_size,
            first_k_letter_counts_sorted={},
            graph_subblocking_config=subblocking.GraphSubblockingConfig(),
            use_orcid_subblocking=False,
        )

    native_plan = {key: sorted(values) for key, values in native_subblocks.items()}
    if expected is not None:
        assert native_plan == expected
    else:
        # Ten "huimin" records exceed the cap of nine, so the compound keys as "hui" and not as "huimin".
        compound_key = next(key for key, values in native_plan.items() if "s10" in values)
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
    ],
)
def test_native_reports_no_middle_dead_ends(tmp_path, rows, maximum_size, expected) -> None:
    """Report the no-middle bucket of each first-name dead end."""
    load_s2and_rust_extension()
    signatures_path = tmp_path / "signatures.arrow"
    _write_signatures_arrow(signatures_path, [(*row, None) for row in rows])
    signature_ids = [row[0] for row in rows]
    with _open_subblocking_arrow_dataset({"signatures": str(signatures_path)}, tmp_path) as arrow_dataset:
        native_subblocks, _, native_dead_ends = subblocking._make_subblocks_with_telemetry_arrow_rust(
            arrow_dataset,
            signature_ids,
            maximum_size,
            first_k_letter_counts_sorted={},
            graph_subblocking_config=subblocking.GraphSubblockingConfig(),
            use_orcid_subblocking=False,
        )

    assert {first: sorted(bucket) for first, bucket in native_dead_ends.items()} == expected
    bucket_of = {sid: key for key, members in native_subblocks.items() for sid in members}
    for first, bucket in expected.items():
        assert all(f"{first}|middle=" in bucket_of[sid].split(", ") for sid in bucket)


@pytest.mark.parametrize(
    ("names", "maximum_size", "enabled", "expected", "skipped"),
    [
        (
            [
                ("bb", "https://orcid.org/0000-0000-0000-0001"),
                ("bb", "0000-0000-0000-0001"),
                ("aa", "ORCID: 0000000000000001"),
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
    ids=["whole-subblocks", "disabled", "transitive-over-capacity"],
)
def test_native_orcid_repair_preserves_whole_subblocks(
    tmp_path: Path, names, maximum_size: int, enabled: bool, expected, skipped
) -> None:
    """Preserve whole subblocks during same-ORCID repair."""
    rows = [(f"s{index}", first, "", orcid) for index, (first, orcid) in enumerate(names, start=1)]
    signatures_path = tmp_path / "signatures.arrow"
    _write_signatures_arrow(signatures_path, rows)
    with _open_subblocking_arrow_dataset({"signatures": str(signatures_path)}, tmp_path) as arrow_dataset:
        subblocks, telemetry, _ = subblocking._make_subblocks_with_telemetry_arrow_rust(
            arrow_dataset,
            [row[0] for row in rows],
            maximum_size=maximum_size,
            first_k_letter_counts_sorted={},
            use_orcid_subblocking=enabled,
            graph_subblocking_config=subblocking.GraphSubblockingConfig(),
        )
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


def test_native_subblocking_uses_loaded_models_orcid_prefix_counts(tmp_path: Path) -> None:
    refreshed_hashes = canonical_artifact_hashes()
    original_hashes = {
        **refreshed_hashes,
        "orcid_prefix_counts_data_sha256": "fcb4cb7bdf30f07366b946b9fdb1bb0a807bdb00f7dda521575daea6b2b9d8ce",
    }
    models = []
    for version, hashes in (("1.4", refreshed_hashes), ("1.3.2", original_hashes)):
        bundle = tmp_path / f"production_model_v{version}"
        write_synthetic_pairwise_bundle(bundle, artifact_hashes=hashes, release_version=version)
        models.append(_load_pairwise_staging_model(bundle, expected_eps_calibration=CALIBRATED_EPS_CALIBRATION))

    rows = [("s1", "ch", ""), ("s2", "cy", ""), ("s3", "cq", "")]
    signatures_path = tmp_path / "signatures.arrow"
    _write_signatures_arrow(signatures_path, [(*row, None) for row in rows])
    block = {"c wang": [row[0] for row in rows]}
    with _open_subblocking_arrow_dataset({"signatures": str(signatures_path)}, tmp_path) as arrow:
        for model, count, expected in zip(
            models, (16, None), ([["s1", "s2"], ["s3"]], [["s1"], ["s2"], ["s3"]]), strict=True
        ):
            counts = model._subblocking_prefix_counts()
            assert counts.get("ch", {}).get("cy") == count
            native_subblocks, _ = model._build_arrow_subblocked_block_dict(
                block, arrow, batching_threshold=2, cluster_seeds_require={}
            )
            assert sorted(map(sorted, native_subblocks.values())) == expected


def test_unexported_model_uses_the_orcid_prefix_counts_its_contract_records() -> None:
    from s2and.model import Clusterer

    original = "fcb4cb7bdf30f07366b946b9fdb1bb0a807bdb00f7dda521575daea6b2b9d8ce"
    # Export records the contract hash, so prediction before export must already use it.
    bound = SimpleNamespace(feature_contract={"orcid_prefix_counts_data_sha256": original})
    unbound = SimpleNamespace(feature_contract={})

    assert Clusterer._subblocking_prefix_counts(bound).get("ch", {}).get("cy") is None
    assert Clusterer._subblocking_prefix_counts(unbound)["ch"]["cy"] == 16


def test_native_subblock_prefix_subdivision(tmp_path) -> None:
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
        arrow_dataset = _open_subblocking_arrow_dataset({"signatures": str(signatures_path)}, case_root)
        rust_subblocks, _rust_telemetry, _ = subblocking._make_subblocks_with_telemetry_arrow_rust(
            arrow_dataset,
            signature_ids,
            maximum_size=1,
            first_k_letter_counts_sorted={},
            graph_subblocking_config=subblocking.GraphSubblockingConfig(),
        )

        assert {key: sorted(values) for key, values in rust_subblocks.items()} == expected, case_id
        arrow_dataset.close()


def _hui_wang_graph_arrow_dataset(
    tmp_path: Path,
    *,
    first_names: Sequence[str] = ("hui",) * 4,
    orcids: Sequence[str | None] = (None,) * 4,
) -> ArrowDataset:
    """Open four "h wang" records whose embeddings pair s1 with s2 and s3 with s4."""
    signatures_path = tmp_path / "signatures.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    specter_path = tmp_path / "specter.arrow"
    _write_signatures_arrow(
        signatures_path,
        [
            (f"s{index}", first, "", orcid)
            for index, (first, orcid) in enumerate(zip(first_names, orcids, strict=True), start=1)
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
    return _open_subblocking_arrow_dataset(paths, tmp_path)


def _eps_sweep_and_production_subblocks(
    arrow_dataset: ArrowDataset, monkeypatch, *, batching_threshold: int = 2
) -> tuple[dict[str, list[str]], dict[bool, dict[str, list[str]]], dict[bool, list[dict[str, Any]]]]:
    """Subblock one block through production and through the sweep with ORCID subblocking off and on."""
    from s2and.featurizer import FeaturizationInfo
    from s2and.model import Clusterer
    from scripts.eps_sweep.sweep_eps_on_linking_gold import _build_arrow_subblocked_block_dict

    monkeypatch.setattr(subblocking, "_resolved_orcid_prefix_counts", lambda counts: {})
    clusterer = Clusterer(FeaturizationInfo(["year_diff"]), None, n_jobs=1, random_state=7)
    clusterer.subblocking_graph_config = subblocking.GraphSubblockingConfig(
        neighbor_mode="exact",
        neighbors=1,
        min_edge_score=0.8,
    )
    blocks = {"hui wang": ["s1", "s2", "s3", "s4"]}
    production, _ = clusterer._build_arrow_subblocked_block_dict(
        blocks,
        arrow_dataset,
        batching_threshold=batching_threshold,
        cluster_seeds_require={},
    )
    swept: dict[bool, dict[str, list[str]]] = {}
    telemetry: dict[bool, list[dict[str, Any]]] = {}
    for orcid in (False, True):
        swept[orcid], telemetry[orcid] = _build_arrow_subblocked_block_dict(
            args=SimpleNamespace(batching_threshold=batching_threshold, use_orcid_subblocking=orcid),
            clusterer=clusterer,
            block_dict=blocks,
            target_signature_ids={"s1"},
            arrow_dataset=arrow_dataset,
        )
    return production, swept, telemetry


def test_rust_arrow_native_graph_subblocking_uses_arrow_evidence_and_records_telemetry(tmp_path, monkeypatch):
    load_s2and_rust_extension()
    arrow_dataset = _hui_wang_graph_arrow_dataset(tmp_path)
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
    production, swept, sweep_telemetry = _eps_sweep_and_production_subblocks(arrow_dataset, monkeypatch)
    arrow_dataset.close()
    # Without ORCIDs the flag cannot matter, so both sweep settings match production.
    for orcid in (False, True):
        assert swept[orcid] == production
        assert sweep_telemetry[orcid][0]["subblocking_telemetry"]["graph_fallback_native"] is True
        assert sweep_telemetry[orcid][0]["subblocking_telemetry"]["graph_fallback_invocation_count"] == 1


def test_eps_sweep_orcid_subblocking_flag_matches_production_only_when_enabled(tmp_path, monkeypatch):
    """A shared ORCID changes the partition, so the sweep's ORCID flag must reach the native planner."""
    load_s2and_rust_extension()
    shared_orcid = "0000-0002-1825-0097"
    # Name splitting leaves s1 apart from s3 and s4. The ORCID repair merges whole subblocks only within the cap.
    arrow_dataset = _hui_wang_graph_arrow_dataset(
        tmp_path,
        first_names=("hui", "hao", "hong", "hong"),
        orcids=(shared_orcid, None, shared_orcid, None),
    )
    production, swept, _telemetry = _eps_sweep_and_production_subblocks(
        arrow_dataset, monkeypatch, batching_threshold=3
    )
    arrow_dataset.close()

    partitions = {orcid: {frozenset(members) for members in subblocks.values()} for orcid, subblocks in swept.items()}
    assert not any({"s1", "s3"} <= members for members in partitions[False]), partitions
    assert frozenset({"s1", "s3", "s4"}) in partitions[True], partitions
    assert swept[True] == production
    assert swept[False] != production


def test_eps_sweep_orcid_free_copy_plans_like_hidden_orcid_sweep(tmp_path, monkeypatch):
    """Production always repairs by ORCID; on the ORCID-free copy that equals the sweep's hidden-ORCID plan."""
    from scripts.eps_sweep.sweep_eps_on_linking_gold import _write_orcid_free_dataset

    load_s2and_rust_extension()
    shared_orcid = "0000-0002-1825-0097"
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = _hui_wang_graph_arrow_dataset(
        source_root,
        first_names=("hui", "hao", "hong", "hong"),
        orcids=(shared_orcid, None, shared_orcid, None),
    )
    _production, swept, _telemetry = _eps_sweep_and_production_subblocks(source, monkeypatch, batching_threshold=3)
    source.close()

    copy_root = _write_orcid_free_dataset(source_root, tmp_path / "copy")
    with (
        pa.memory_map(str(copy_root / "signatures.arrow"), "r") as copied,
        pa.memory_map(str(source_root / "signatures.arrow"), "r") as original,
    ):
        copied_table = pa.ipc.open_file(copied).read_all()
        original_table = pa.ipc.open_file(original).read_all()
    assert copied_table["author_orcid"].null_count == copied_table.num_rows == 4
    assert copied_table.drop_columns(["author_orcid"]).equals(original_table.drop_columns(["author_orcid"]))
    assert (copy_root / "specter.arrow").samefile(source_root / "specter.arrow")
    copy = ArrowDataset.open(copy_root, require_specter=True)
    production_on_copy, _, _ = _eps_sweep_and_production_subblocks(copy, monkeypatch, batching_threshold=3)
    copy.close()
    assert production_on_copy == swept[False]
    # A retry reuses the copy of the same source and refuses a copy of another source.
    assert _write_orcid_free_dataset(source_root, tmp_path / "copy") == copy_root
    other_root = tmp_path / "other"
    other_root.mkdir()
    _hui_wang_graph_arrow_dataset(
        other_root, first_names=("hui", "hao", "hong", "hung"), orcids=(shared_orcid, None, shared_orcid, None)
    ).close()
    with pytest.raises(ValueError, match="another source"):
        _write_orcid_free_dataset(other_root, tmp_path / "copy")
    assert not (tmp_path / "copy.partial").exists()
