"""Augmented identities must be confined to their source training population."""

import pytest

from scripts.production.model.augmented_leakage import require_training_members, source_members
from tests.incremental_test_helpers import _dataset


@pytest.mark.parametrize("population", ["validation", "test", "omitted"])
def test_augmented_training_rejects_nontraining_source_identity(population: str) -> None:
    # Use the actual split implementation, with enough size strata for 80/10/10.
    dataset = _dataset([("Ada", "", "Smith")] * 300, {})
    dataset.num_clusters_for_block_size = 5
    blocks = {}
    offset = 0
    for index in range(100):
        size = index % 5 + 1
        blocks[f"b{index}"] = [str(sid) for sid in range(offset, offset + size)]
        offset += size
    train, val, test = dataset.split_blocks_helper(blocks)
    training = {sid for members in train.values() for sid in members}
    protected = (
        "outside-plan"
        if population == "omitted"
        else next(iter((val if population == "validation" else test).values()))[0]
    )
    safe = next(iter(training))
    require_training_members(source_members([(f"qian___{safe}", "medline___1")]), "qian", training)
    with pytest.raises(ValueError, match="outside its training split"):
        require_training_members(source_members([(f"qian___{safe}", f"qian___{protected}")]), "qian", training)


def test_augmented_source_ids_preserve_separators_and_dataset_scope() -> None:
    members = source_members([("qian___id___suffix", "kisti___id___suffix")])
    require_training_members(members, "qian", {"id___suffix"})
    with pytest.raises(ValueError, match="'kisti'"):
        require_training_members(members, "kisti", set())


@pytest.mark.parametrize("signature_id", ["unprefixed", "___id", "qian___"])
def test_augmented_source_ids_require_the_canonical_prefix(signature_id: str) -> None:
    with pytest.raises(ValueError, match="Invalid augmented source signature ID"):
        source_members([(signature_id, "qian___valid")])
