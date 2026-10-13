"""Augmented identities must be confined to their source training population."""

import json
from pathlib import Path

import pytest

from scripts.production.model.augmented_leakage import (
    require_declared_sources,
    require_training_members,
    source_members,
    training_members_from_sources,
)
from tests.incremental_test_helpers import _dataset
from tests.pairwise_release_helpers import write_pairwise_dataset


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


def test_augmented_sources_follow_the_declared_dataset_population() -> None:
    members = source_members([("patents___p1", "medline___m1")])
    require_declared_sources(members, ("patents", "augmented"), context="Test")
    with pytest.raises(ValueError, match="undeclared source datasets.*patents"):
        require_declared_sources(members, ("qian", "augmented"), context="Test")
    with pytest.raises(ValueError, match="undeclared source datasets.*aminer"):
        require_declared_sources({"aminer": {"a1"}}, ("patents", "augmented"), context="Test")


def test_augmented_copies_of_a_declared_fixed_pair_source_use_only_fit_members(tmp_path: Path) -> None:
    pair_path = tmp_path / "medline_train_pairs.csv"
    pair_path.write_text("signature_id_1,signature_id_2,label\nm1,m2,YES\n", encoding="utf-8")
    training = training_members_from_sources("medline", {"train_pairs": pair_path}, random_seed=1111)
    require_training_members(source_members([("medline___m1", "medline___m2")]), "medline", training)
    with pytest.raises(ValueError, match="outside its training split"):
        require_training_members(source_members([("medline___m1", "medline___validation")]), "medline", training)


def test_inventor_fit_members_preserve_signature_ids_above_int64(tmp_path: Path) -> None:
    files = write_pairwise_dataset(tmp_path, name="inventors_s2and")
    signatures = json.loads(files["signatures"].read_text(encoding="utf-8"))
    remap = {signature_id: str(10**24 + index) for index, signature_id in enumerate(signatures)}
    signatures = {
        remap[signature_id]: {**record, "signature_id": remap[signature_id]}
        for signature_id, record in signatures.items()
    }
    files["signatures"].write_text(json.dumps(signatures), encoding="utf-8")
    clusters = json.loads(files["clusters"].read_text(encoding="utf-8"))
    for cluster in clusters.values():
        cluster["signature_ids"] = [remap[signature_id] for signature_id in cluster["signature_ids"]]
    files["clusters"].write_text(json.dumps(clusters), encoding="utf-8")

    training = training_members_from_sources("inventors_s2and", files, random_seed=1111)
    assert training
    assert training <= signatures.keys()
    assert all(isinstance(signature_id, str) and int(signature_id) > 2**63 for signature_id in training)
    signature_id = next(iter(training))
    members = source_members([(f"inventors_s2and___{signature_id}", "medline___m1")])
    require_declared_sources(members, ("inventors_s2and", "augmented"), context="Test")
    require_training_members(members, "inventors_s2and", training)
