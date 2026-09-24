"""Public prediction regressions for block-local, atomic claimed profiles."""

import numpy as np
import pytest

from tests.incremental_test_helpers import _dataset, _predict
from tests.incremental_test_helpers import empty_orcid_priors as empty_orcid_priors


def _assert_partition(clusters, blocks):
    ids = [sid for members in clusters.values() for sid in members]
    assert sorted(ids) == sorted(sid for members in blocks.values() for sid in members)
    assert all(any(set(members) <= set(block) for block in blocks.values()) for members in clusters.values())


@pytest.mark.parametrize("route", ["python", "arrow"])
@pytest.mark.parametrize("claimed", [False, True])
@pytest.mark.parametrize(("count", "threshold"), [(20, 45), (2, 3)])
def test_mixed_initial_claim_keeps_compatible_full_name_neighbors(tmp_path, route, claimed, count, threshold):
    names = [("A", "", "Smith")] * count + [("Ada", "", "Smith")] * count + [("Bob", "", "Smith")] * count
    dataset = _dataset(names, {"0": "claimed", str(count): "claimed"} if claimed else {})
    blocks = {"one": list(dataset.signatures)}

    clusters = _predict(tmp_path, dataset, blocks, route, threshold=threshold)

    _assert_partition(clusters, blocks)
    assert {frozenset(members) for members in clusters.values()} == {
        frozenset(str(index) for index in range(count * 2)),
        frozenset(str(index) for index in range(count * 2, count * 3)),
    }


@pytest.mark.parametrize("route", ["python", "arrow"])
def test_mixed_initial_claim_at_capacity_preserves_membership_and_name_constraints(tmp_path, route):
    names = [("A", "", "Smith")] * 2 + [("Ada", "", "Smith")] * 2 + [("Bob", "", "Smith")] * 2
    dataset = _dataset(names, {"0": "claimed", "2": "claimed"})
    blocks = {"one": list(dataset.signatures)}

    clusters = _predict(tmp_path, dataset, blocks, route, threshold=2)

    _assert_partition(clusters, blocks)
    assert any({"0", "2"} <= set(members) for members in clusters.values())
    assert all(not (set(members) & {"0", "1", "2", "3"} and set(members) & {"4", "5"}) for members in clusters.values())


@pytest.mark.parametrize("initials", [False, True])
def test_arrow_subblocking_normalizes_integer_signature_ids(tmp_path, initials):
    names = [("Ada", "", "Smith")] * 4
    if initials:
        names = [("A", "", "Smith")] * 2 + [("Ada", "", "Smith")] * 2 + [("Bob", "", "Smith")] * 2
    dataset = _dataset(names, {})
    string_blocks = {"one": list(dataset.signatures)}
    integer_blocks = {"one": [int(signature_id) for signature_id in dataset.signatures]}

    expected = _predict(tmp_path / "strings", dataset, string_blocks, "arrow", threshold=3)
    actual = _predict(tmp_path / "integers", dataset, integer_blocks, "arrow", threshold=3)

    assert actual == expected
    _assert_partition(actual, string_blocks)


@pytest.mark.parametrize("route", ["python", "arrow"])
@pytest.mark.parametrize("same_last", [False, True])
def test_claimed_profile_and_initial_attachment_stay_in_original_block(tmp_path, route, same_last):
    names = [("Ada", "", "Smith")] * 3 + [("Ada", "", "Smith" if same_last else "Jones")] * 3
    names.append(("A", "", "Smith" if same_last else "Jones"))
    dataset = _dataset(names, {"0": "claimed", "3": "claimed"})
    blocks = {"left": ["0", "1", "2"], "right": ["3", "4", "5", "6"]}
    clusters = _predict(tmp_path, dataset, blocks, route)
    _assert_partition(clusters, blocks)
    assert any({"0", "1", "2"} <= set(members) for members in clusters.values())
    assert any({"3", "4", "5"} <= set(members) for members in clusters.values())


@pytest.mark.parametrize("route", ["python", "arrow"])
def test_claimed_component_is_atomic_across_first_name_subblocks(tmp_path, route):
    dataset = _dataset([("Abdul", "", "Smith"), ("Ahmed", "", "Smith")] * 2, {"0": "claimed", "1": "claimed"})
    clusters = _predict(tmp_path, dataset, {"one": list(dataset.signatures)}, route, threshold=2, distance=0.9)
    assert {frozenset(members) for members in clusters.values()} == {
        frozenset({"0", "1"}),
        frozenset({"2"}),
        frozenset({"3"}),
    }


@pytest.mark.parametrize("route", ["python", "arrow"])
def test_claimed_component_does_not_absorb_incompatible_names(tmp_path, route):
    dataset = _dataset(
        [("Ada", "", "Smith")] * 3 + [("Amy", "", "Smith")] * 3 + [("A", "", "Smith")], {"0": "claimed", "3": "claimed"}
    )
    blocks = {"one": list(dataset.signatures)}
    clusters = _predict(tmp_path, dataset, blocks, route)
    _assert_partition(clusters, blocks)
    assert any({"0", "3"} <= set(members) for members in clusters.values())
    assert not any({"1", "4"} <= set(members) for members in clusters.values())


@pytest.mark.parametrize("route", ["python", "arrow"])
def test_initial_only_block_uses_bulk_even_beside_full_name_block(tmp_path, route):
    dataset = _dataset([("Ada", "", "Smith")] + [("A", "", "Smith")] * 4, {})
    blocks = {"full": ["0"], "initial": ["1", "2", "3", "4"]}
    clusters = _predict(tmp_path, dataset, blocks, route, threshold=2)
    _assert_partition(clusters, blocks)


@pytest.mark.parametrize("route", ["python", "arrow"])
def test_profile_size_limit_applies_to_each_original_block(tmp_path, route):
    dataset = _dataset(
        [("Ada", "", "Smith"), ("Ada", "", "Smith"), ("A", "", "Smith")] * 2,
        dict.fromkeys(["0", "1", "3", "4"], "claimed"),
    )
    blocks = {"left": ["0", "1", "2"], "right": ["3", "4", "5"]}
    clusters = _predict(tmp_path, dataset, blocks, route, threshold=2)
    _assert_partition(clusters, blocks)
    assert any({"0", "1"} <= set(members) for members in clusters.values())
    assert any({"3", "4"} <= set(members) for members in clusters.values())


@pytest.mark.parametrize("route", ["python", "arrow"])
def test_residual_cluster_ids_from_independent_blocks_preserve_all_outputs(tmp_path, route):
    dataset = _dataset([("Ada", "", "Smith"), ("Ada", "", "Smith"), ("A", "", "Smith")] * 2, {})
    dataset.cluster_seeds_disallow = {("0", "2"), ("1", "2"), ("3", "5"), ("4", "5")}
    blocks = {"left": ["0", "1", "2"], "right": ["3", "4", "5"]}
    clusters = _predict(tmp_path, dataset, blocks, route, threshold=2)
    _assert_partition(clusters, blocks)
    assert {frozenset(members) for members in clusters.values()} == {
        frozenset({"0", "1"}),
        frozenset({"2"}),
        frozenset({"3", "4"}),
        frozenset({"5"}),
    }


@pytest.mark.parametrize("route", ["python", "arrow"])
@pytest.mark.parametrize("constraint_source", ["explicit_disallow", "partial_supervision", "ignored_seeds"])
@pytest.mark.parametrize("disallow", [("1", "3"), ("1", "2")])
def test_query_disallows_survive_subblocked_attachment(tmp_path, route, constraint_source, disallow):
    dataset = _dataset(
        [
            ("Ada", "", "Smith"),
            ("A", "", "Smith"),
            ("A", "", "Smith"),
            ("A", "Alpha", "Smith"),
            ("A", "Alpha", "Smith"),
        ],
        {},
    )
    # A missing middle name is compatible with Alpha, so name rules cannot
    # accidentally mask a dropped explicit negative across initial-only groups.
    assert dataset.get_constraint(*disallow) is None
    kwargs = {}
    if constraint_source == "partial_supervision":
        kwargs["partial_supervision"] = {disallow: 10000.0}
    else:
        dataset.cluster_seeds_disallow = {disallow}
        if constraint_source == "ignored_seeds":
            dataset.cluster_seeds_require = dict.fromkeys(disallow, "claimed")
            dataset.max_seed_cluster_id = 1
            kwargs["incremental_dont_use_cluster_seeds"] = True
    blocks = {"one": list(dataset.signatures)}
    clusters = _predict(tmp_path, dataset, blocks, route, threshold=2, **kwargs)
    _assert_partition(clusters, blocks)
    assert not any(set(disallow) <= set(members) for members in clusters.values())


@pytest.mark.parametrize("route", ["python", "arrow"])
def test_query_must_link_survives_subblocked_attachment(tmp_path, route):
    """A must-link whose partner is attached in a later initial-only group."""
    dataset = _dataset(
        [
            ("Ada", "", "Smith"),
            ("A", "", "Smith"),
            ("A", "", "Smith"),
            ("A", "Alpha", "Smith"),
            ("A", "Alpha", "Smith"),
        ],
        {},
    )
    blocks = {"one": list(dataset.signatures)}
    clusters = _predict(
        tmp_path, dataset, blocks, route, threshold=2, distance=0.9, partial_supervision={("1", "3"): 0.0}
    )
    _assert_partition(clusters, blocks)
    assert any({"1", "3"} <= set(members) for members in clusters.values())


@pytest.mark.parametrize("route", ["python", "arrow"])
@pytest.mark.parametrize(
    "flags", [{}, {"incremental_dont_use_cluster_seeds": True}, {"use_default_constraints_as_supervision": False}]
)
def test_oversized_required_component_only_rejected_when_active(tmp_path, route, flags):
    dataset = _dataset([("Ada", "", "Smith")] * 3, dict.fromkeys(["0", "1", "2"], "claimed"))
    if not flags:
        with pytest.raises(ValueError, match="component exceeds batching_threshold"):
            _predict(tmp_path, dataset, {"one": list(dataset.signatures)}, route, threshold=2, **flags)
    else:
        clusters = _predict(
            tmp_path, dataset, {"one": list(dataset.signatures)}, route, threshold=2, distance=0.9, **flags
        )
        assert all(len(members) == 1 for members in clusters.values())


def test_legacy_author_id_prediction_does_not_pack_unused_seeds(tmp_path):
    dataset = _dataset([("Ada", "", "Smith")] * 3, dict.fromkeys(["0", "1", "2"], "claimed"))
    for sid, signature in dataset.signatures.items():
        dataset.signatures[sid] = signature._replace(author_id=f"legacy-{sid}")
    result = _predict(tmp_path, dataset, {"one": list(dataset.signatures)}, "python", threshold=2, use_s2_clusters=True)
    assert result == {f"legacy-{sid}": [sid] for sid in dataset.signatures}


class _YearDistanceClassifier:
    """Make two tight groups whose average distance dilutes one negative."""

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        distances = np.where(features[:, 0] == 0, 0.0, 0.1)
        return np.column_stack([distances, 1.0 - distances])


@pytest.mark.parametrize("route", ["python", "arrow"])
@pytest.mark.parametrize("source", ["dataset", "caller"])
@pytest.mark.parametrize("exclude_query", [False, True])
def test_generated_bulk_seed_negatives_do_not_reject_initial_attachment(tmp_path, route, source, exclude_query):
    """Real bulk linkage can contain a negative without being a caller seed claim."""
    dataset = _dataset([("Ada", "", "Smith")] * 3 + [("A", "", "Smith")], {})
    negatives = {("0", "2")}
    if exclude_query:
        negatives.add(("0", "3"))
    dataset.cluster_seeds_disallow = negatives.copy() if source == "dataset" else set()
    partial_supervision = dict.fromkeys(negatives, 10000.0) if source == "caller" else {}
    original_disallows = dataset.cluster_seeds_disallow.copy()
    original_supervision = partial_supervision.copy()
    blocks = {"one": list(dataset.signatures)}

    clusters = _predict(
        tmp_path,
        dataset,
        blocks,
        route,
        threshold=3,
        linkage="single",
        partial_supervision=partial_supervision,
    )

    _assert_partition(clusters, blocks)
    # Single linkage joins the three full names around their internal negative;
    # that generated group must not become an inconsistent caller seed claim.
    expected = {frozenset({"0", "1", "2"}), frozenset({"3"})}
    if not exclude_query:
        expected = {frozenset({"0", "1", "2", "3"})}
    assert {frozenset(members) for members in clusters.values()} == expected
    assert dataset.cluster_seeds_require == {}
    assert dataset.cluster_seeds_disallow == original_disallows
    assert partial_supervision == original_supervision


def test_generated_bulk_seed_negatives_preserve_average_linkage_arrow_partition(tmp_path):
    """Keep the original average-linkage dilution and excluded-query mechanism."""
    dataset = _dataset([("Ada", "", "Smith")] * 400 + [("A", "", "Smith")], {})
    for index in range(200, 400):
        paper_id = str(100 + index)
        dataset.papers[paper_id] = dataset.papers[paper_id]._replace(year=2001)
    dataset.cluster_seeds_disallow = {("0", "200"), ("0", "400")}
    partial_supervision = {}
    original_disallows = dataset.cluster_seeds_disallow.copy()
    original_supervision = partial_supervision.copy()
    blocks = {"one": list(dataset.signatures)}

    clusters = _predict(
        tmp_path,
        dataset,
        blocks,
        "arrow",
        threshold=400,
        classifier=_YearDistanceClassifier(),
        partial_supervision=partial_supervision,
    )

    _assert_partition(clusters, blocks)
    # The 200-by-200 linkage mean is about .35, below eps=.5. Preserve that
    # computed bulk group, while a negative involving the new query stays active.
    expected = {frozenset(str(index) for index in range(400)), frozenset({"400"})}
    assert {frozenset(members) for members in clusters.values()} == expected
    assert dataset.cluster_seeds_require == {}
    assert dataset.cluster_seeds_disallow == original_disallows
    assert partial_supervision == original_supervision
