from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

from s2and.consts import NORMALIZATION_VERSION, PROJECT_ROOT_PATH
from s2and.subblocking import _LazyCanonicalOrcidPrefixCounts

ORCID_1 = "0000-0000-0000-0001"
ORCID_2 = "0000-0000-0000-0002"
ORCID_X = "0000-0000-0000-000X"


def _load_module() -> ModuleType:
    module_path = Path(PROJECT_ROOT_PATH) / "scripts" / "production" / "counts" / "generate_orcid_name_prefix_counts.py"
    spec = importlib.util.spec_from_file_location("generate_orcid_name_prefix_counts", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_is_side_effect_free_without_internal_pys2() -> None:
    module = _load_module()

    assert callable(module.main)


def test_warehouse_query_emits_and_orders_by_the_canonical_orcid() -> None:
    module = _load_module()

    query = module._warehouse_query(None)

    assert "regexp_substr(" in query
    assert "pae.orcid raw_orcid" in query
    assert "end orcid" in query
    assert "order by orcid nulls last" in query
    assert "regexp_replace(upper(pae.orcid), '[^0-9X]', '')" not in query


@pytest.mark.parametrize(
    ("raw_orcid", "expected"),
    [
        ("leading junk 0000-0000-0000-0001 trailing junk", ORCID_1),
        ("leading-x 0000-0000-0000-000x trailing-x", ORCID_X),
        ("x0000-0000-0000-0001", None),
        ("not-an-orcid", None),
        (None, None),
    ],
)
def test_source_orcid_canonicalization_matches_the_warehouse_key(
    raw_orcid: str | None,
    expected: str | None,
) -> None:
    module = _load_module()
    match = re.search(
        module._CANONICAL_SOURCE_ORCID_SQL_PATTERN,
        raw_orcid or "",
        flags=re.IGNORECASE,
    )
    if match is None:
        warehouse_key = None
    else:
        compact = re.sub(module._ORCID_DASH_SQL_PATTERN, "", match.group(0)).upper()
        warehouse_key = f"{compact[:4]}-{compact[4:8]}-{compact[8:12]}-{compact[12:]}"

    assert module._canonical_source_orcid(raw_orcid) == expected
    assert warehouse_key == expected


def test_streaming_builder_keeps_repeated_canonical_groups_contiguous() -> None:
    module = _load_module()
    rows = [
        {
            "raw_orcid": "leading-x 0000-0000-0000-0001 trailing-x",
            "orcid": ORCID_1,
            "first_name": "Alice",
            "middle": None,
        },
        {
            "raw_orcid": "ORCID: 0000000000000001",
            "orcid": ORCID_1,
            "first_name": "Amy",
            "middle": None,
        },
        {
            "raw_orcid": "ORCID: 0000-0000-0000-000x",
            "orcid": ORCID_X,
            "first_name": "Axel",
            "middle": None,
        },
        {
            "raw_orcid": "000000000000000x trailing",
            "orcid": ORCID_X,
            "first_name": "Ava",
            "middle": None,
        },
        {
            "raw_orcid": "not-an-orcid",
            "orcid": None,
            "first_name": "Invalid",
            "middle": None,
        },
        {
            "raw_orcid": None,
            "orcid": None,
            "first_name": "Missing",
            "middle": None,
        },
    ]

    counts, metrics, _digest = module.build_prefix_counts_from_sorted_rows(
        rows,
        [],
        min_orcid_count=1,
    )

    assert counts
    assert metrics["accepted_rows"] == 4
    assert metrics["orcid_groups"] == 2
    assert metrics["unique_orcid_names"] == 4
    assert metrics["rejected_invalid_orcid"] == 1
    assert metrics["rejected_missing_orcid"] == 1


def test_streaming_builder_retains_the_canonical_orcid_monotonicity_check() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="sorted by canonical orcid"):
        module.build_prefix_counts_from_sorted_rows(
            [
                {"orcid": ORCID_2, "first_name": "Amy", "middle": None},
                {"orcid": ORCID_1, "first_name": "Alice", "middle": None},
            ],
            [],
            min_orcid_count=1,
        )


def test_legacy_orcid_counts_remain_excluded_until_regenerated() -> None:
    with (Path(PROJECT_ROOT_PATH) / "pyproject.toml").open("rb") as stream:
        setuptools = tomllib.load(stream)["tool"]["setuptools"]
    package_data = setuptools["package-data"]["s2and"]
    excluded_package_data = setuptools["exclude-package-data"]["s2and"]

    assert "data/first_k_letter_counts_from_orcid.json" not in package_data
    assert "data/first_k_letter_counts_from_orcid.meta.json" not in package_data
    assert "data/first_k_letter_counts_from_orcid.json" in excluded_package_data
    assert "data/first_k_letter_counts_from_orcid.meta.json" in excluded_package_data
    assert "data/first_k_letter_counts_from_orcid.manifest.json" not in package_data
    assert "data/orcid-prefix-counts-*/*.json" not in package_data


def test_empty_canonical_names_are_rejected_with_metrics() -> None:
    module = _load_module()
    counts, metrics, _digest = module.build_prefix_counts_from_sorted_rows(
        [
            {"orcid": ORCID_1, "first_name": "Alice", "middle": None},
            {"orcid": ORCID_1, "first_name": "", "middle": None},
            {"orcid": ORCID_1, "first_name": "...", "middle": None},
            {"orcid": "", "first_name": "Amy", "middle": None},
            {"orcid": "not-an-orcid", "first_name": "Ava", "middle": None},
        ],
        [],
        min_orcid_count=1,
    )

    assert counts == {}
    assert metrics["accepted_rows"] == 1
    assert metrics["rejected_empty_canonical_first"] == 2
    assert metrics["rejected_missing_orcid"] == 1
    assert metrics["rejected_invalid_orcid"] == 1


def test_prefix_counts_are_unordered_and_deterministic() -> None:
    module = _load_module()
    forward, _, _ = module.build_prefix_counts_from_sorted_rows(
        [
            {"orcid": ORCID_1, "first_name": "Alice", "middle": None},
            {"orcid": ORCID_1, "first_name": "Amy", "middle": None},
        ],
        {("alicia", "amanda")},
        min_orcid_count=1,
        min_alias_count=1,
    )
    reverse, _, _ = module.build_prefix_counts_from_sorted_rows(
        [
            {"orcid": ORCID_1, "first_name": "Amy", "middle": None},
            {"orcid": ORCID_1, "first_name": "Alice", "middle": None},
        ],
        {("amanda", "alicia")},
        min_orcid_count=1,
        min_alias_count=1,
    )

    assert forward == reverse
    assert all(left <= right for left, nested in forward.items() for right in nested)


def test_fixture_cli_writes_direct_data_and_metadata(tmp_path: Path) -> None:
    module = _load_module()
    fixture_path = tmp_path / "rows.json"
    fixture_path.write_text(
        json.dumps(
            [
                {"orcid": ORCID_1, "first_name": "Alice", "middle": None},
                {"orcid": ORCID_1, "first_name": "Amy", "middle": None},
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"

    assert (
        module.main(
            [
                "--input-json",
                str(fixture_path),
                "--output-dir",
                str(output_dir),
                "--source-snapshot-id",
                "fixture-2026-07-09",
            ]
        )
        == 0
    )

    data_path = output_dir / "first_k_letter_counts_from_orcid.json"
    metadata_path = output_dir / "first_k_letter_counts_from_orcid.meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata == {
        "schema_version": 1,
        "normalization_version": NORMALIZATION_VERSION,
        "pair_key_semantics": "unordered_lexicographic",
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }

    with pytest.raises(FileExistsError, match="artifact already exists"):
        module.main(
            [
                "--input-json",
                str(fixture_path),
                "--output-dir",
                str(output_dir),
                "--source-snapshot-id",
                "fixture-2026-07-09",
            ]
        )


def test_runtime_loader_is_lazy_and_verifies_the_direct_artifact(tmp_path: Path) -> None:
    module = _load_module()
    lazy_counts = _LazyCanonicalOrcidPrefixCounts(tmp_path)

    with pytest.raises(FileNotFoundError, match="Missing canonical ORCID prefix-count metadata"):
        len(lazy_counts)

    module.write_artifact(
        {"al": {"am": 7}},
        output_dir=tmp_path,
        overwrite=False,
    )
    assert _LazyCanonicalOrcidPrefixCounts(tmp_path).load() == {"al": {"am": 7}}
    assert lazy_counts.load() is lazy_counts.load()
    assert dict(lazy_counts) == {"al": {"am": 7}}

    metadata = json.loads((tmp_path / "first_k_letter_counts_from_orcid.meta.json").read_text(encoding="utf-8"))
    assert lazy_counts.data_sha256() == metadata["data_sha256"]
    data_path = tmp_path / "first_k_letter_counts_from_orcid.json"
    data_path.write_text('{"al":{"az":9}}', encoding="utf-8")
    with pytest.raises(ValueError, match="data SHA-256"):
        _LazyCanonicalOrcidPrefixCounts(tmp_path).load()


def test_runtime_loader_does_not_fall_back_to_unversioned_data(tmp_path: Path) -> None:
    (tmp_path / "first_k_letter_counts_from_orcid.json").write_text('{"al":{"am":7}}', encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Missing canonical ORCID prefix-count metadata"):
        _LazyCanonicalOrcidPrefixCounts(tmp_path).load()


@pytest.mark.parametrize(
    ("metadata_change", "expected_error"),
    [
        ({"normalization_version": "legacy"}, "normalization_version"),
        ({"unexpected": True}, "fields do not match"),
    ],
)
def test_runtime_loader_enforces_the_small_metadata_contract(
    tmp_path: Path,
    metadata_change: dict[str, object],
    expected_error: str,
) -> None:
    module = _load_module()
    module.write_artifact({"al": {"am": 7}}, output_dir=tmp_path, overwrite=False)
    metadata_path = tmp_path / "first_k_letter_counts_from_orcid.meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(metadata_change)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        _LazyCanonicalOrcidPrefixCounts(tmp_path).load()


@pytest.mark.parametrize("counts", [{"Al": {"Am": 7}}, {"ál": {"ám": 7}}])
def test_runtime_loader_rejects_noncanonical_prefix_tokens(
    tmp_path: Path,
    counts: dict[str, dict[str, int]],
) -> None:
    module = _load_module()
    module.write_artifact(
        {"al": {"am": 7}},
        output_dir=tmp_path,
        overwrite=False,
    )
    data_path = tmp_path / "first_k_letter_counts_from_orcid.json"
    metadata_path = tmp_path / "first_k_letter_counts_from_orcid.meta.json"
    data_bytes = json.dumps(counts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    data_path.write_bytes(data_bytes)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["data_sha256"] = hashlib.sha256(data_bytes).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="lowercase printable ASCII prefixes"):
        _LazyCanonicalOrcidPrefixCounts(tmp_path).load()


def test_streaming_source_digest_covers_selected_row_content() -> None:
    module = _load_module()
    rows = [
        {"orcid": ORCID_1, "first_name": "Alice", "middle": None},
        {"orcid": ORCID_1, "first_name": "Amy", "middle": None},
    ]
    counts, metrics, digest = module.build_prefix_counts_from_sorted_rows(
        rows,
        [],
        min_orcid_count=1,
    )
    changed_counts, _, changed_digest = module.build_prefix_counts_from_sorted_rows(
        [*rows[:1], {"orcid": ORCID_1, "first_name": "Ava", "middle": None}],
        [],
        min_orcid_count=1,
    )

    assert counts != changed_counts
    assert digest != changed_digest
    assert metrics["source_rows"] == 2
    assert metrics["max_unique_names_per_orcid"] == 2


def test_compact_json_writer_matches_canonical_encoding(tmp_path: Path) -> None:
    module = _load_module()
    payload = {"zo": {"gian ": 2, "amy": 4}, "al": {"bob": 3}}
    path = tmp_path / "counts.json"

    digest = module._write_compact_json(path, payload)
    expected = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    assert path.read_bytes() == expected
    assert digest == hashlib.sha256(expected).hexdigest()


def test_source_digest_covers_deduplicated_rows_and_name_tuple_content() -> None:
    module = _load_module()
    rows = [
        {"orcid": ORCID_1, "first_name": "Alice", "middle": None},
        {"orcid": ORCID_1, "first_name": "Amy", "middle": None},
    ]
    counts, _, digest = module.build_prefix_counts_from_sorted_rows(rows, [("alicia", "amanda")], min_orcid_count=1)
    duplicate_counts, _, duplicate_digest = module.build_prefix_counts_from_sorted_rows(
        [*rows, rows[-1]],
        [("alicia", "amanda")],
        min_orcid_count=1,
    )
    alias_counts, _, alias_digest = module.build_prefix_counts_from_sorted_rows(
        rows,
        [("alicia", "ava")],
        min_orcid_count=1,
        min_alias_count=1,
    )

    assert duplicate_counts == counts
    assert duplicate_digest == digest
    assert alias_counts != counts
    assert alias_digest != digest


def test_writer_rejects_noncanonical_count_pairs_before_writing(tmp_path: Path) -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="lexicographically ordered"):
        module.write_artifact(
            {"am": {"al": 7}},
            output_dir=tmp_path,
            overwrite=False,
        )

    assert not list(tmp_path.iterdir())


def test_writer_rejects_non_ascii_prefixes_before_writing(tmp_path: Path) -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="lowercase printable ASCII prefixes"):
        module.write_artifact(
            {"ál": {"amy": 7}},
            output_dir=tmp_path,
            overwrite=False,
        )

    assert not list(tmp_path.iterdir())


def test_name_pair_expansion_has_an_explicit_per_orcid_bound() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="max_names_per_orcid=2"):
        module.build_prefix_counts_from_sorted_rows(
            [
                {"orcid": ORCID_1, "first_name": "Alice", "middle": None},
                {"orcid": ORCID_1, "first_name": "Amy", "middle": None},
                {"orcid": ORCID_1, "first_name": "Ava", "middle": None},
            ],
            [],
            min_orcid_count=1,
            max_names_per_orcid=2,
        )


def test_cli_refuses_implicit_warehouse_access(tmp_path: Path) -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="explicitly authorize warehouse access"):
        module.main(
            [
                "--output-dir",
                str(tmp_path),
                "--source-snapshot-id",
                "fixture",
            ]
        )
