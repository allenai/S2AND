from __future__ import annotations

import json
from pathlib import Path

import pytest

from s2and.arrow_schema import validate_arrow_schema

CONTRACT_PATH = Path("s2and/arrow_schema_contract.json")


EXPECTED_REQUIRED_COLUMNS = {
    "altered_cluster_signatures": {"signature_id"},
    "cluster_seed_disallows": {"signature_id_1", "signature_id_2"},
    "cluster_seeds": {"signature_id", "cluster_id"},
    "incremental_query_signatures": {"signature_id", "query_view", "query_author"},
    "paper_authors": {"paper_id", "position", "author_name"},
    "papers": {"paper_id", "title", "venue", "journal_name"},
    "signatures": {
        "signature_id",
        "paper_id",
        "author_first",
        "author_middle",
        "author_last",
        "author_suffix",
        "author_affiliations",
        "author_position",
    },
    "specter": {"paper_id", "embedding"},
}


def test_arrow_schema_contract_required_columns_are_pinned() -> None:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "s2and_arrow_schema_contract_v1"
    required_by_table = {
        table_name: {column["name"] for column in columns if column["required"]}
        for table_name, columns in payload["tables"].items()
    }

    assert required_by_table == EXPECTED_REQUIRED_COLUMNS


def test_arrow_schema_contract_has_no_duplicate_columns() -> None:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    for table_name, columns in payload["tables"].items():
        column_names = [column["name"] for column in columns]
        assert len(column_names) == len(set(column_names)), table_name


def test_full_schema_validation_allows_missing_optional_columns_and_checks_present_ones() -> None:
    pa = pytest.importorskip("pyarrow")
    required_only = pa.schema(
        [
            pa.field("paper_id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("venue", pa.string()),
            pa.field("journal_name", pa.string()),
        ]
    )

    validate_arrow_schema(required_only, table_name="papers")

    missing_required = required_only.remove(required_only.get_field_index("title"))
    with pytest.raises(ValueError, match="papers table is missing required column 'title'"):
        validate_arrow_schema(missing_required, table_name="papers")

    malformed_optional = required_only.append(pa.field("year", pa.int32()))
    with pytest.raises(ValueError, match="papers column 'year' expected int64"):
        validate_arrow_schema(malformed_optional, table_name="papers")


def test_subset_schema_validation_uses_exact_contract_physical_types() -> None:
    pa = pytest.importorskip("pyarrow")
    canonical_subset = pa.schema([pa.field("paper_id", pa.string())])
    validate_arrow_schema(canonical_subset, table_name="papers", columns={"paper_id"})

    with pytest.raises(ValueError, match="papers table is missing required column 'year'"):
        validate_arrow_schema(canonical_subset, table_name="papers", columns={"year"})

    noncanonical_subset = pa.schema([pa.field("paper_id", pa.large_string())])
    with pytest.raises(ValueError, match="papers column 'paper_id' expected string, got large_string"):
        validate_arrow_schema(noncanonical_subset, table_name="papers", columns={"paper_id"})
