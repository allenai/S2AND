from __future__ import annotations

import pytest

import s2and.arrow_schema as arrow_schema
from s2and.arrow_schema import validate_arrow_schema


def test_full_schema_validation_allows_missing_optional_columns_and_checks_present_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyarrow as pa

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

    duplicate = {"name": "paper_id", "datatype": "string", "required": True}
    monkeypatch.setattr(arrow_schema, "_contract_tables", lambda: {"papers": [duplicate, duplicate]})
    with pytest.raises(ValueError, match="repeats column 'paper_id'"):
        validate_arrow_schema(required_only, table_name="papers")


def test_subset_schema_validation_uses_exact_contract_physical_types() -> None:
    import pyarrow as pa

    canonical_subset = pa.schema([pa.field("paper_id", pa.string())])
    validate_arrow_schema(canonical_subset, table_name="papers", columns={"paper_id"})

    with pytest.raises(ValueError, match="papers table is missing required column 'year'"):
        validate_arrow_schema(canonical_subset, table_name="papers", columns={"year"})

    noncanonical_subset = pa.schema([pa.field("paper_id", pa.large_string())])
    with pytest.raises(ValueError, match="papers column 'paper_id' expected string, got large_string"):
        validate_arrow_schema(noncanonical_subset, table_name="papers", columns={"paper_id"})
