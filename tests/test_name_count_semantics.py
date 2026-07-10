"""Canonical ``last_first_initial`` count-key behavior."""

from typing import Any

from s2and.data import ANDData
from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from tests.helpers import tiny_name_counts_provenance


def _name_count_tables() -> dict[str, Any]:
    return {
        "first_dict": {"abdul": 7, "alexander": 8},
        "last_dict": {"sattar": 9, "konovalov": 10},
        "first_last_dict": {"abdul sattar": 11, "alexander konovalov": 12},
        "last_first_initial_dict": {
            "sattar a": 13,
            "sattar abdul": 41,
            "konovalov a": 14,
            "konovalov alexander": 42,
        },
        "provenance": tiny_name_counts_provenance(),
    }


def _name_count_index(tmp_path: Any) -> str:
    tables = _name_count_tables()
    path, _metrics = write_name_counts_index(
        tmp_path,
        (
            tables["first_dict"],
            tables["last_dict"],
            tables["first_last_dict"],
            tables["last_first_initial_dict"],
        ),
        tables["provenance"],
    )
    return path


def _dummy_dataset(name: str, *, name_counts_index: str) -> ANDData:
    return ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        name=name,
        mode="inference",
        name_counts_index=name_counts_index,
        preprocess=True,
        n_jobs=1,
    )


def test_last_first_initial_uses_first_character(tmp_path):
    dataset = _dummy_dataset(
        "dummy_name_count_semantics_default",
        name_counts_index=_name_count_index(tmp_path),
    )
    baseline = dataset.signatures["1"].author_info_name_counts
    assert baseline is not None
    assert baseline.last_first_initial == 13
