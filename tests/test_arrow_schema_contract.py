from __future__ import annotations

import json
from pathlib import Path

CONTRACT_PATH = Path("s2and/arrow_schema_contract.json")


EXPECTED_REQUIRED_COLUMNS = {
    "altered_cluster_signatures": {"signature_id"},
    "cluster_seed_disallows": {"signature_id_1", "signature_id_2"},
    "cluster_seeds": {"signature_id", "cluster_id"},
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
