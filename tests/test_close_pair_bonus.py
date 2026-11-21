"""Check the close-pair bonus of average linkage: heights, cuts, hard negatives, and input ownership."""

import numpy as np
import pytest
from fastcluster import linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform

from s2and.consts import LARGE_DISTANCE
from s2and.model_pairwise import FastCluster, constrained_linkage

BONUS, CLOSE = 8.0, 0.05


def _two_groups(cross: list[float]) -> np.ndarray:
    """Condensed distances of two tight groups of three, with the given nine cross distances."""
    square = np.full((6, 6), 0.01)
    square[:3, 3:] = np.asarray(cross).reshape(3, 3)
    square[3:, :3] = square[:3, 3:].T
    np.fill_diagonal(square, 0.0)
    return squareform(square)


def test_bonus_merges_clusters_that_few_close_pairs_link() -> None:
    # Two of nine cross pairs are near certain; the plain average 0.547 stays above EPS 0.53.
    distances = _two_groups([0.01, 0.01, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])

    plain = FastCluster(eps=0.53).fit(distances).labels_
    with_bonus = FastCluster(eps=0.41, close_pair_bonus=BONUS, close_pair_distance=CLOSE).fit(distances).labels_

    assert len(set(plain)) == 2
    assert len(set(with_bonus)) == 1


def test_lower_eps_with_bonus_keeps_clusters_without_close_pairs_apart() -> None:
    # No cross pair is close; the plain average 0.45 merges at 0.53 but not at the lower EPS.
    distances = _two_groups([0.45] * 9)

    plain = FastCluster(eps=0.53).fit(distances).labels_
    with_bonus = FastCluster(eps=0.41, close_pair_bonus=BONUS, close_pair_distance=CLOSE).fit(distances).labels_

    assert len(set(plain)) == 1
    assert len(set(with_bonus)) == 2


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tree_heights_are_average_minus_bonus_times_close_share(seed: int) -> None:
    rng = np.random.default_rng(seed)
    # A bimodal mix, as the pair model gives: some near-certain pairs, the rest spread out.
    distances = np.where(rng.random(40 * 39 // 2) < 0.2, rng.uniform(0, 0.04, 780), rng.uniform(0.2, 1.0, 780))

    tree = constrained_linkage(
        distances, "average", preserve_input=True, close_pair_bonus=BONUS, close_pair_distance=CLOSE
    )
    reference = linkage(distances + BONUS * (distances >= CLOSE), "average")

    assert np.array_equal(tree[:, :2], reference[:, :2])
    assert np.allclose(tree[:, 2], np.maximum(reference[:, 2] - BONUS, 0.0), atol=1e-12)
    for eps in (0.0, 0.2, 0.41, 0.53, 0.9):
        cut = fcluster(tree, t=eps, criterion="distance")
        expected = fcluster(reference, t=eps + BONUS, criterion="distance")
        assert np.array_equal(cut, expected)


def test_zero_bonus_is_bit_identical_to_plain_linkage() -> None:
    distances = np.random.default_rng(3).uniform(0, 1, 30 * 29 // 2)

    tree = constrained_linkage(
        distances, "average", preserve_input=True, close_pair_bonus=0.0, close_pair_distance=CLOSE
    )

    assert np.array_equal(tree, linkage(distances, "average"))


def test_hard_negative_still_separates_clusters_that_close_pairs_link() -> None:
    cross = [0.01] * 8 + [LARGE_DISTANCE]
    distances = _two_groups(cross)

    labels = FastCluster(eps=0.41, close_pair_bonus=BONUS, close_pair_distance=CLOSE).fit(distances).labels_

    assert len(set(labels)) == 2


def test_soft_sentinel_above_one_is_not_treated_as_close() -> None:
    # Points 0 and 1 are close; point 2 sits at a soft distance between 1 and the bonus.
    distances = np.asarray([0.01, 5.0, 5.0])

    tree = constrained_linkage(
        distances, "average", preserve_input=True, close_pair_bonus=BONUS, close_pair_distance=CLOSE
    )

    # Without the shift of sentinels, this height would fall to zero and merge at every EPS.
    assert tree[-1, 2] == pytest.approx(5.0)
    assert len(set(fcluster(tree, t=0.9, criterion="distance"))) == 2


@pytest.mark.parametrize("preserve_input", [True, False])
def test_preserved_input_is_unchanged(preserve_input: bool) -> None:
    distances = _two_groups([0.01, 0.01, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
    original = distances.copy()

    model = FastCluster(eps=0.41, preserve_input=preserve_input, close_pair_bonus=BONUS, close_pair_distance=CLOSE)
    labels = model.fit(distances).labels_

    assert len(set(labels)) == 1
    if preserve_input:
        assert np.array_equal(distances, original)


def test_read_only_input_is_copied() -> None:
    distances = _two_groups([0.01, 0.01, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
    distances.setflags(write=False)

    model = FastCluster(eps=0.41, preserve_input=False, close_pair_bonus=BONUS, close_pair_distance=CLOSE)

    assert len(set(model.fit(distances).labels_)) == 1


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"linkage": "single", "close_pair_bonus": 1.0, "close_pair_distance": 0.05}, "requires average linkage"),
        ({"close_pair_bonus": -1.0, "close_pair_distance": 0.05}, "must be within"),
        ({"close_pair_bonus": 1.0, "close_pair_distance": 1.5}, "must be within"),
        # A bonus near the hard-negative barrier would cancel it, so the bonus has a cap.
        ({"close_pair_bonus": 1e12, "close_pair_distance": 0.05}, "must be within"),
        ({"close_pair_bonus": 1.0, "close_pair_distance": 0.0}, "positive with a bonus"),
    ],
)
def test_invalid_settings_are_rejected(params: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FastCluster(**params).fit(_two_groups([0.5] * 9))
