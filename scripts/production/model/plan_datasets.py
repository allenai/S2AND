"""Load model-plan datasets from the published Arrow data.

A random-block dataset names one complete benchmark Arrow dataset. Its frozen
held-out blocks and any excluded signatures never enter development, so the
trainer, EPS calibration, and outer-test evaluation all select from one source.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import pandas as pd

from s2and._sha256 import sha256_file
from s2and.arrow_inputs import ArrowDataset
from s2and.arrow_training import build_training_anddata_from_arrow
from s2and.data import ANDData, _map_fixed_pair_labels
from s2and.production_training_contract import EXCLUDED_SIGNATURES_ROLE, UNKNOWN_SIGNATURES_ROLE, ModelDataset
from scripts.production.model.unknown_signatures import UnknownSignatures, load_declared_unknown_signatures

# ANDData defaults; pair counts matter only where pairs are sampled for training.
_DEFAULT_TRAIN_PAIRS_SIZE = 30_000
_DEFAULT_VAL_PAIRS_SIZE = 5_000


def require_arrow_plan(datasets: Mapping[str, ModelDataset]) -> None:
    """Reject a plan whose datasets do not select pairs from published Arrow data."""

    json_sources = sorted(name for name, dataset in datasets.items() if not dataset.uses_arrow)
    if json_sources:
        raise ValueError(f"Model plan datasets must name published Arrow data: {json_sources}")


def heldout_blocks(dataset: ModelDataset) -> dict[str, list[str]]:
    """Read the frozen outer test blocks of one random-block dataset."""

    path = dataset.files["heldout_blocks"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Held-out blocks must form a nonempty object: {path}")
    blocks: dict[str, list[str]] = {}
    seen: set[str] = set()
    for block_id, members in payload.items():
        if not isinstance(members, list) or not members:
            raise ValueError(f"Held-out block {block_id!r} must list signatures: {path}")
        blocks[str(block_id)] = []
        for member in members:
            signature_id = str(member)
            if signature_id in seen:
                raise ValueError(f"Held-out blocks repeat signature {signature_id!r}: {path}")
            seen.add(signature_id)
            blocks[str(block_id)].append(signature_id)
    return blocks


def heldout_signature_ids(dataset: ModelDataset) -> frozenset[str]:
    """Return every signature in the frozen outer test blocks."""

    return frozenset(signature_id for members in heldout_blocks(dataset).values() for signature_id in members)


def excluded_signature_ids(dataset: ModelDataset) -> frozenset[str]:
    """Return signatures reserved outside both development and the outer test."""

    if EXCLUDED_SIGNATURES_ROLE not in dataset.files:
        return frozenset()
    path = dataset.files[EXCLUDED_SIGNATURES_ROLE]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(not isinstance(value, str) or not value for value in payload):
        raise ValueError(f"Excluded signatures must form a list of nonempty strings: {path}")
    return frozenset(str(value) for value in payload)


def plan_unknown_signatures(dataset: ModelDataset) -> UnknownSignatures:
    """Read the unknown-signature list that the plan declares beside the labels.

    A plan without the role declares that no list exists, so an added list fails.
    """

    return load_declared_unknown_signatures(
        dataset.files,
        dataset.files.get(UNKNOWN_SIGNATURES_ROLE),
        expected_sha256=dataset.sha256.get(UNKNOWN_SIGNATURES_ROLE),
    )


@contextmanager
def open_plan_arrow(dataset: ModelDataset) -> Iterator[ArrowDataset]:
    """Open the Arrow dataset that the plan binds by manifest digest.

    The manifest is checked again at open time, so a swap after plan verification
    fails here. ``ArrowDataset.open`` then checks every file against the manifest.
    """

    manifest = dataset.files["arrow"]
    if sha256_file(manifest) != dataset.sha256["arrow"]:
        raise ValueError(f"Arrow dataset changed after model-plan verification: {manifest}")
    with ArrowDataset.open(
        dataset.files["arrow"].parent,
        require_specter=True,
        require_name_counts_index=True,
    ) as arrow_dataset:
        yield arrow_dataset


def load_development_dataset(
    name: str,
    dataset: ModelDataset,
    arrow_dataset: ArrowDataset,
    *,
    name_tuples: frozenset[tuple[str, str]],
    random_seed: int,
    n_jobs: int,
    train_pairs_size: int = _DEFAULT_TRAIN_PAIRS_SIZE,
    val_pairs_size: int = _DEFAULT_VAL_PAIRS_SIZE,
) -> ANDData:
    """Build the population that selects training and validation pairs.

    Args:
        name: Dataset name from the model plan.
        dataset: Verified plan entry with an ``arrow`` role.
        arrow_dataset: The open dataset that ``open_plan_arrow`` returned.
        name_tuples: Canonical name tuples.
        random_seed: Seed for the block split and pair sampling.
        n_jobs: Worker count for dataset construction.
        train_pairs_size: Training pairs sampled per random-block dataset.
        val_pairs_size: Validation pairs sampled per random-block dataset.

    Returns:
        A train-mode dataset that excludes held-out and excluded signatures.
    """

    files = dataset.files
    if dataset.split_mode == "fixed_pairs":
        return build_training_anddata_from_arrow(
            arrow_dataset,
            name,
            train_pairs=str(files["train_pairs"]),
            val_pairs=str(files["val_pairs"]),
            train_pairs_size=train_pairs_size,
            val_pairs_size=val_pairs_size,
            random_seed=random_seed,
            n_jobs=n_jobs,
            name_tuples=name_tuples,
        )
    withheld = heldout_signature_ids(dataset) | excluded_signature_ids(dataset)
    return build_training_anddata_from_arrow(
        arrow_dataset,
        name,
        clusters=str(files["clusters"]),
        train_pairs_size=train_pairs_size,
        val_pairs_size=val_pairs_size,
        random_seed=random_seed,
        n_jobs=n_jobs,
        name_tuples=name_tuples,
        keep_signature=lambda signature_id: signature_id not in withheld,
    )


def load_heldout_dataset(
    name: str,
    dataset: ModelDataset,
    arrow_dataset: ArrowDataset,
    *,
    name_tuples: frozenset[tuple[str, str]],
    n_jobs: int,
) -> ANDData:
    """Build the outer test population of one random-block dataset."""

    heldout = heldout_signature_ids(dataset)
    return build_training_anddata_from_arrow(
        arrow_dataset,
        name,
        clusters=str(dataset.files["clusters"]),
        n_jobs=n_jobs,
        name_tuples=name_tuples,
        keep_signature=heldout.__contains__,
    )


def development_split_members(name: str, dataset: ModelDataset, *, random_seed: int) -> tuple[set[str], set[str]]:
    """Return the training and validation signatures of one development population.

    The trainer runs this check for every augmented source before any featurization,
    so a leaked copy fails before the expensive stages start.

    Args:
        name: Dataset name from the model plan.
        dataset: Verified plan entry with an ``arrow`` role.
        random_seed: Seed that the trainer uses for the block split.

    Returns:
        Signature IDs in the training split and in the validation split.
    """

    if dataset.split_mode == "fixed_pairs":
        members: list[set[str]] = []
        for role in ("train_pairs", "val_pairs"):
            pairs = _map_fixed_pair_labels(pd.read_csv(dataset.files[role]), f"{name} {role}")
            members.append(
                {str(value) for pair in pairs.iloc[:, :2].itertuples(index=False, name=None) for value in pair}
            )
        return members[0], members[1]
    with open_plan_arrow(dataset) as arrow_dataset:
        # Block splits do not depend on name tuples, so the check runs before artifact loading.
        development = load_development_dataset(
            name, dataset, arrow_dataset, name_tuples=frozenset(), random_seed=random_seed, n_jobs=1
        )
        train, validation, _ = development.split_cluster_signatures()
    return (
        {signature_id for members in train.values() for signature_id in members},
        {signature_id for members in validation.values() for signature_id in members},
    )
