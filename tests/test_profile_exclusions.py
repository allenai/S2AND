"""Profile exclusions evict excluded mentions instead of vetoing the profile's merge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import squareform

from s2and.consts import LARGE_DISTANCE
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.feature_block import read_profile_exclusions_arrow, write_profile_exclusions_arrow
from s2and.model import Clusterer
from s2and.model_pairwise import FastCluster
from s2and.profile_exclusions import (
    enforce_profile_exclusions_in_block,
    evict_profile_exclusion_violations,
    normalize_profile_exclusions,
    validate_profile_exclusions,
)
from tests.helpers import build_arrow_training_dataset, build_dummy_dataset
from tests.model_helpers import ConstantDistanceClassifier

# Mentions: p0, p1 placed on the profile; w0..w3 the person's other work;
# x sits inside that work but the user excluded it from the profile.
SIGNATURES = ["p0", "p1", "w0", "w1", "w2", "w3", "x"]


def _square() -> np.ndarray:
    square = np.full((7, 7), 0.9)
    np.fill_diagonal(square, 0.0)
    work = [2, 3, 4, 5, 6]
    for i in work:
        for j in work:
            if i != j:
                square[i, j] = 0.05
    square[0, 1] = square[1, 0] = 0.0
    for placed in (0, 1):
        for i in work:
            square[placed, i] = square[i, placed] = 0.3
    return square


def _labels(condensed: np.ndarray) -> list[int]:
    labels = FastCluster(eps=0.5).fit(condensed).labels_
    assert labels is not None
    return [int(label) for label in labels]


def test_disallow_veto_isolates_the_profile() -> None:
    # The production failure: one hard negative from x to the anchor keeps
    # the placed pair away from the whole body of work.
    square = _square()
    square[0, 6] = square[6, 0] = LARGE_DISTANCE
    labels = _labels(squareform(square))
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]


def test_eviction_keeps_profile_and_removes_excluded_mention() -> None:
    condensed = squareform(_square())
    exclusions = normalize_profile_exclusions([("p0", "x")])
    first = _labels(condensed.copy())
    assert first[0] == first[6]

    labels = enforce_profile_exclusions_in_block(SIGNATURES, first, condensed, exclusions, _labels)

    profile = {s for s, label in zip(SIGNATURES, labels, strict=True) if label == labels[0]}
    assert profile == {"p0", "p1", "w0", "w1", "w2", "w3"}
    assert labels[6] != labels[0]


def test_majority_excluded_cluster_is_still_evicted() -> None:
    # Most of the cluster is excluded. The unexcluded rest stays with the profile:
    # no share cutoff turns the exclusions back into a veto.
    condensed = squareform(_square())
    exclusions = normalize_profile_exclusions([("p0", s) for s in ("w0", "w1", "w2", "x")])
    first = _labels(condensed.copy())
    assert len(set(first)) == 1

    labels = enforce_profile_exclusions_in_block(SIGNATURES, first, condensed, exclusions, _labels)

    profile = {s for s, label in zip(SIGNATURES, labels, strict=True) if label == labels[0]}
    assert profile == {"p0", "p1", "w3"}


def test_enforcement_leaves_unviolated_labels_alone() -> None:
    square = _square()
    square[6, :] = square[:, 6] = 0.9
    square[6, 6] = 0.0
    condensed = squareform(square)
    first = _labels(condensed.copy())
    labels = enforce_profile_exclusions_in_block(
        SIGNATURES, first, condensed, normalize_profile_exclusions([("p0", "x")]), _labels
    )
    assert labels == first


def test_unsettled_block_returns_labels_for_the_final_pass() -> None:
    # A relabel that never separates the block stands in for a chain longer than the round bound.
    condensed = squareform(_square())
    exclusions = normalize_profile_exclusions([("p0", "x")])
    stuck = [0] * len(SIGNATURES)

    labels = enforce_profile_exclusions_in_block(SIGNATURES, stuck, condensed, exclusions, lambda _matrix: stuck)

    assert labels == stuck
    repaired, moved = evict_profile_exclusion_violations({"a": list(SIGNATURES)}, exclusions)
    assert moved == 1
    assert "x" not in repaired["a"]


def test_final_pass_moves_violations_to_singletons() -> None:
    clusters = {"a": ["p0", "w0", "x"], "b": ["y"]}
    repaired, moved = evict_profile_exclusion_violations(clusters, normalize_profile_exclusions([("p0", "x")]))
    assert moved == 1
    assert repaired["a"] == ["p0", "w0"]
    assert ["x"] in repaired.values()


def test_validation_rejects_exclusion_of_a_placed_mention() -> None:
    with pytest.raises(ValueError, match="contradicts a placement"):
        validate_profile_exclusions(normalize_profile_exclusions([("p0", "p1")]), {"p0": "0", "p1": "0"})
    with pytest.raises(ValueError, match="its own anchor"):
        normalize_profile_exclusions([("p0", "p0")])


def test_arrow_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "profile_exclusions.arrow"
    write_profile_exclusions_arrow(path, [("p0", "x"), ("p0", "y"), ("p0", "x")])
    assert read_profile_exclusions_arrow(path) == (("p0", "x"), ("p0", "y"))


@pytest.mark.parametrize("route", ["python", "arrow"])
def test_predict_routes_evict_excluded_mention(route: str, tmp_path: Path) -> None:
    # Every pair is closer than eps, so the whole block merges unless the exclusion is honored.
    dataset = build_dummy_dataset("profile_exclusions")
    model = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=ConstantDistanceClassifier(0.4),
        cluster_model=FastCluster(eps=0.5),
        n_jobs=1,
        use_default_constraints_as_supervision=False,
    )
    blocks = {"a": ["0", "1", "2", "3"]}
    arrow_dataset = (
        build_arrow_training_dataset(dataset, tmp_path / "arrow").arrow_dataset if route == "arrow" else None
    )

    def run(profile_exclusions: list[tuple[str, str]] | None) -> set[frozenset[str]]:
        if route == "python":
            clusters, _ = model.predict(blocks, dataset, profile_exclusions=profile_exclusions)
        else:
            assert arrow_dataset is not None
            clusters, _ = model.predict_from_arrow(
                blocks, arrow_dataset, name_tuples=frozenset(), profile_exclusions=profile_exclusions
            )
        return {frozenset(members) for members in clusters.values()}

    assert run(None) == {frozenset({"0", "1", "2", "3"})}
    assert run([("0", "1")]) == {frozenset({"0", "2", "3"}), frozenset({"1"})}
