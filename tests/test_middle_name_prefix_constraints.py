"""Verify middle-name prefix policy through the Python and native boundaries."""

from pathlib import Path
from typing import Any

import pytest

from s2and.consts import CLUSTER_SEEDS_LOOKUP, LARGE_DISTANCE
from s2and.data import ANDData
from s2and.feature_port import _get_rust_featurizer
from tests.helpers import build_arrow_training_dataset
from tests.incremental_test_helpers import _dataset

Name = tuple[str, str, str]

CASES: list[tuple[Name, Name, float | None]] = [
    (("C.", "Meg", "Urry"), ("C.", "Megan", "Urry"), None),
    (("Jean", "Ch", "Smith"), ("Jean", "Charles", "Smith"), None),
    (("C.", "Yi", "Tang"), ("C.", "Yik", "Tang"), None),
    (("Ho", "Yun", "Lee"), ("Ho", "Yung", "Lee"), None),
    (("C.", "Ann Marie", "Smith"), ("C.", "Anna M", "Smith"), None),
    (("C.", "Meg", "Smith"), ("C.", "Megan Claire", "Smith"), None),
    (("C.", "Meg Jr", "Smith"), ("C.", "Megan Jr", "Smith"), None),
    (("C.", "Meg Ann", "Smith"), ("C.", "Megan Alice", "Smith"), LARGE_DISTANCE),
    (("C.", "Meg Paul", "Smith"), ("C.", "Paul Megan", "Smith"), None),
    (("C.", "M Paul", "Smith"), ("C.", "Meg Peter", "Smith"), LARGE_DISTANCE),
    (("C.", "M", "Smith"), ("C.", "Megan", "Smith"), None),
    (("C.", "", "Smith"), ("C.", "Megan", "Smith"), None),
    (("C.", "Ann", "Smith"), ("C.", "Nan", "Smith"), LARGE_DISTANCE),
    (("C.", "Maria", "Smith"), ("C.", "Marai", "Smith"), None),
    (("C.", "Meg Ann", "Smith"), ("C.", "Ann Meg", "Smith"), None),
    (("C.", "Meg", "Smith"), ("C.", "Michael", "Smith"), LARGE_DISTANCE),
    (("C.", "Meg", "Smith"), ("C.", "Charles", "Smith"), LARGE_DISTANCE),
    (("C.", "Meg", "Smith"), ("D.", "Megan", "Smith"), LARGE_DISTANCE),
    (("C.", "Meg", "Smith"), ("C.", "Megan", "Jones"), LARGE_DISTANCE),
    (("Carl", "Meg", "Smith"), ("Chris", "Megan", "Smith"), LARGE_DISTANCE),
    # A prefix-matched dash-bound first name adds its later tokens to the middle.
    (("J.-D.", "T.", "Smith"), ("J.", "D.", "Smith"), None),
    (("J.-D.", "", "Smith"), ("J.", "T.", "Smith"), LARGE_DISTANCE),
    (("Chi-Hung", "", "Tam"), ("C.", "M.", "Tam"), LARGE_DISTANCE),
    (("Jean-Daniel", "", "Smith"), ("Jean", "Daniel", "Smith"), None),
    (("Jean-Daniel", "", "Smith"), ("Jean", "Paul", "Smith"), LARGE_DISTANCE),
    (("Jean-Daniel", "T.", "Smith"), ("Jean-D.", "", "Smith"), None),
    # Tokens that the other first name covers do not join the middle.
    (("Jean-Pierre", "A.", "Smith"), ("Jean-Pierre", "B.", "Smith"), LARGE_DISTANCE),
    (("Xiao-Qing", "", "Liu"), ("Xiaoqing", "Frank", "Liu"), None),
    # An alias covers the full first name, so its later tokens stay out.
    (("Abol-Ghasem", "", "Smith"), ("Ali", "R.", "Smith"), None),
    (("Abol-Ghasem", "M.", "Smith"), ("Ali", "R.", "Smith"), LARGE_DISTANCE),
]
ALIASES = {("abol ghasem", "ali")}


@pytest.fixture(scope="module")
def prefix_dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[ANDData, Any, dict[str, int]]:
    """Use real ingestion and Arrow conversion for both implementations."""
    names = [name for left, right, _ in CASES for name in (left, right)]
    dataset = _dataset(names, {})
    dataset.name_tuples = set(ALIASES)
    arrow_dataset = build_arrow_training_dataset(dataset, tmp_path_factory.mktemp("middle-prefix"))
    native = _get_rust_featurizer(arrow_dataset)
    indices = {signature_id: index for index, signature_id in enumerate(native.signature_ids())}
    return dataset, native, indices


@pytest.mark.parametrize("case_index", range(len(CASES)))
def test_middle_prefix_constraint_parity(prefix_dataset: tuple[ANDData, Any, dict[str, int]], case_index: int) -> None:
    """Accept aligned prefixes and preserve each prior independent veto."""
    dataset, native, indices = prefix_dataset
    expected = CASES[case_index][2]
    signature_ids = (str(2 * case_index), str(2 * case_index + 1))
    for left, right in (signature_ids, signature_ids[::-1]):
        assert dataset.get_constraint(left, right) == expected
        assert native.get_constraints_block_upper_triangle_indexed([indices[left], indices[right]], num_threads=1)[
            2
        ] == [expected]


@pytest.mark.parametrize(
    ("require", "disallow", "expected"),
    [
        ({}, {("0", "1")}, CLUSTER_SEEDS_LOOKUP["disallow"]),
        ({"0": "a", "1": "a"}, set(), CLUSTER_SEEDS_LOOKUP["require"]),
        ({"0": "a", "1": "b"}, set(), CLUSTER_SEEDS_LOOKUP["disallow"]),
    ],
)
def test_middle_prefix_preserves_seed_precedence(
    prefix_dataset: tuple[ANDData, Any, dict[str, int]],
    require: dict[str, str],
    disallow: set[tuple[str, str]],
    expected: float,
) -> None:
    """Keep explicit seed rules above compatible middle names."""
    dataset, native, indices = prefix_dataset
    overlay = native.with_cluster_seeds(require, disallow)
    for left, right in (("0", "1"), ("1", "0")):
        assert (
            dataset._get_constraint(left, right, cluster_seeds_require=require, cluster_seeds_disallow=disallow)
            == expected
        )
        assert overlay.get_constraints_block_upper_triangle_indexed([indices[left], indices[right]], num_threads=1)[
            2
        ] == [expected]


def test_middle_prefix_preserves_orcid_precedence(tmp_path: Path) -> None:
    """Keep the ORCID override and its suppression above name constraints."""
    dataset = _dataset([("C.", "Meg", "Smith"), ("D.", "Megan", "Jones")], {})
    for signature_id in ("0", "1"):
        dataset.signatures[signature_id] = dataset.signatures[signature_id]._replace(
            author_info_orcid="0000000218250097"
        )
    arrow_dataset = build_arrow_training_dataset(dataset, tmp_path / "arrow")
    native = _get_rust_featurizer(arrow_dataset)
    indices = {signature_id: index for index, signature_id in enumerate(native.signature_ids())}
    for suppress_orcid, expected in ((False, -2.0), (True, 23.0)):
        for left, right in (("0", "1"), ("1", "0")):
            assert (
                dataset.get_constraint(left, right, low_value=-2, high_value=23, suppress_orcid=suppress_orcid)
                == expected
            )
            assert native.get_constraints_block_upper_triangle_indexed(
                [indices[left], indices[right]],
                low_value=-2,
                high_value=23,
                suppress_orcid=suppress_orcid,
                num_threads=1,
            )[2] == [expected]
