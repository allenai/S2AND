import math
import random
import tracemalloc

import pytest

from s2and.consts import NUMPY_NAN
from s2and.data import ANDData, _sample_within_block_random_pairs


def _legacy_sample(
    blocks: dict[str, list[str]],
    signature_to_cluster_id: dict[str, str] | None,
    sample_size: int,
    random_seed: int,
) -> list[tuple[str, str, int | float]]:
    candidates: list[tuple[str, str, int | float]] = []
    for signatures in blocks.values():
        for first_index, first_signature in enumerate(signatures):
            for second_signature in signatures[first_index + 1 :]:
                if signature_to_cluster_id is None:
                    label: int | float = NUMPY_NAN
                else:
                    label = int(signature_to_cluster_id[first_signature] == signature_to_cluster_id[second_signature])
                candidates.append((first_signature, second_signature, label))
    return random.Random(random_seed).sample(candidates, min(len(candidates), sample_size))


def _assert_pairs_equal(
    actual: list[tuple[str, str, int | float]],
    expected: list[tuple[str, str, int | float]],
    *,
    case_id: str,
) -> None:
    assert [pair[:2] for pair in actual] == [pair[:2] for pair in expected], case_id
    for actual_pair, expected_pair in zip(actual, expected, strict=True):
        if math.isnan(expected_pair[2]):
            assert math.isnan(actual_pair[2]), case_id
        else:
            assert actual_pair[2] == expected_pair[2], case_id


def test_within_block_rank_sampling_matches_legacy_output() -> None:
    blocks = {
        "empty": [],
        "singleton": ["singleton"],
        "first": ["a", "b", "c", "d"],
        "second": ["e", "f"],
        "duplicates": ["g", "h", "g"],
    }
    cluster_ids = {
        "singleton": "unused",
        "a": "one",
        "b": "one",
        "c": "two",
        "d": "three",
        "e": "four",
        "f": "four",
        "g": "five",
        "h": "six",
    }
    cases = (
        ("empty-sample", 0, 0, False),
        ("single-pair", 0, 1, False),
        ("partial-sample", 0, 4, False),
        ("exact-population", 0, 10, False),
        ("sample-capped-to-population", 0, 11, False),
        ("seed-1111", 1111, 4, False),
        ("maximum-seed", 2**31 - 1, 4, False),
        ("cluster-labels", 7, 10, True),
    )
    for case_id, random_seed, sample_size, with_cluster_labels in cases:
        signature_to_cluster_id = cluster_ids if with_cluster_labels else None

        dataset = ANDData.__new__(ANDData)
        dataset.pair_sampling_mode = "within_block_random"
        dataset.signature_to_cluster_id = signature_to_cluster_id
        dataset.random_seed = random_seed

        actual = dataset.pair_sampling(sample_size, [], blocks)
        expected = _legacy_sample(blocks, signature_to_cluster_id, sample_size, random_seed)

        _assert_pairs_equal(actual, expected, case_id=case_id)


def test_within_block_rank_sampling_preserves_negative_sample_error() -> None:
    dataset = ANDData.__new__(ANDData)
    dataset.pair_sampling_mode = "within_block_random"
    dataset.signature_to_cluster_id = None
    dataset.random_seed = 7

    with pytest.raises(ValueError, match="Sample larger than population or is negative"):
        dataset.pair_sampling(-1, [], {"block": ["a", "b"]})


def test_within_block_rank_sampling_validates_all_cluster_labels_before_sampling() -> None:
    dataset = ANDData.__new__(ANDData)
    dataset.pair_sampling_mode = "within_block_random"
    dataset.signature_to_cluster_id = {"a": "cluster"}
    dataset.random_seed = 7

    with pytest.raises(KeyError, match="b"):
        dataset.pair_sampling(0, [], {"block": ["a", "b"]})


def test_within_block_rank_sampling_memory_is_independent_of_candidate_count() -> None:
    signatures = [str(index) for index in range(200_000)]
    total_candidates = len(signatures) * (len(signatures) - 1) // 2

    tracemalloc.start()
    try:
        pairs = _sample_within_block_random_pairs(
            {"large": signatures},
            signature_to_cluster_id=None,
            sample_size=25,
            random_seed=1111,
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert total_candidates == 19_999_900_000
    assert len(pairs) == 25
    assert len(set(pair[:2] for pair in pairs)) == 25
    assert peak_bytes < 1_000_000
