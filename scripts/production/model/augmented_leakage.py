"""Match augmented pair members to their original benchmark identities."""

from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path

PRODUCTION_AUGMENTED_SOURCES = frozenset(
    {"aminer", "arnetminer", "inspire", "kisti", "medline", "pubmed", "qian", "zbmath"}
)


def split_members_from_sources(name: str, files: Mapping[str, Path], *, random_seed: int) -> tuple[set[str], set[str]]:
    """Resolve the actual fit and validation populations without access to held-out inputs.

    Args:
        name: Benchmark dataset name.
        files: Model files with clusters or fixed train and validation pairs.
        random_seed: Seed used by the production pairwise trainer.

    Returns:
        Signature IDs in the fit population and in the validation population.
    """

    import pandas as pd

    from s2and.data import ANDData, _map_fixed_pair_labels

    if "train_pairs" in files:
        members: list[set[str]] = []
        for role in ("train_pairs", "val_pairs"):
            if role not in files:
                members.append(set())
                continue
            pairs = _map_fixed_pair_labels(pd.read_csv(files[role]), f"{name} {role}")
            members.append(
                {str(value) for pair in pairs.iloc[:, :2].itertuples(index=False, name=None) for value in pair}
            )
        return members[0], members[1]

    dataset = ANDData(
        signatures=str(files["signatures"]),
        papers=str(files["papers"]),
        clusters=str(files["clusters"]),
        name=name,
        mode="train",
        preprocess=False,
        random_seed=random_seed,
        name_counts_index=None,
        name_tuples=frozenset(),
        n_jobs=1,
    )
    train, validation, _ = dataset.split_cluster_signatures()
    return (
        {signature_id for members in train.values() for signature_id in members},
        {signature_id for members in validation.values() for signature_id in members},
    )


def training_members_from_sources(name: str, files: Mapping[str, Path], *, random_seed: int) -> set[str]:
    """Resolve the actual fit population without access to held-out inputs."""

    return split_members_from_sources(name, files, random_seed=random_seed)[0]


def source_members(pairs: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    """Return source signature IDs grouped by the augmentation's dataset prefix."""
    members: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        for signature_id in pair:
            dataset, separator, original_id = signature_id.partition("___")
            if not separator or not dataset or not original_id:
                raise ValueError(f"Invalid augmented source signature ID: {signature_id!r}")
            members[dataset].add(original_id)
    return dict(members)


def require_declared_sources(
    augmented: Mapping[str, set[str]],
    declared_dataset_names: Collection[str],
    *,
    context: str,
) -> None:
    """Reject copies from datasets absent from the declared model population."""
    # MEDLINE remains an augmentation-only source in existing production inputs.
    undeclared = set(augmented) - (set(declared_dataset_names) - {"augmented"}) - {"medline"}
    if undeclared:
        raise ValueError(f"{context} has undeclared source datasets: {sorted(undeclared)}")


def reject_heldout_members(
    augmented: Mapping[str, set[str]],
    heldout: Mapping[str, set[str]],
    *,
    context: str,
    population: str = "training",
) -> None:
    """Reject augmented training or validation members present in a held-out population."""
    for name, signature_ids in heldout.items():
        overlap = augmented.get(name, set()) & signature_ids
        if overlap:
            raise ValueError(
                f"{context}: augmented {population} contains held-out {name!r} signatures: "
                f"count={len(overlap)} sample={sorted(overlap)[:5]}. "
                f"Regenerate augmented {population} pairs using the frozen benchmark development populations."
            )


def require_validation_members(augmented: Mapping[str, set[str]], name: str, validation: set[str]) -> None:
    """Reject validation copies outside the actual source validation split."""
    outside = augmented.get(name, set()) - validation
    if outside:
        raise ValueError(
            f"Augmented validation contains {name!r} signatures outside its validation split: "
            f"count={len(outside)} sample={sorted(outside)[:5]}. "
            "Regenerate augmented validation pairs using this seed and the frozen benchmark validation populations; "
            "training, test, and omitted signatures must not enter model selection."
        )


def require_training_members(augmented: Mapping[str, set[str]], name: str, training: set[str]) -> None:
    """Reject copies outside the actual source training split, including omitted IDs."""
    outside = augmented.get(name, set()) - training
    if outside:
        raise ValueError(
            f"Augmented training contains {name!r} signatures outside its training split: "
            f"count={len(outside)} sample={sorted(outside)[:5]}. "
            "Regenerate augmented training pairs using this seed and the frozen benchmark training populations; "
            "validation, test, and omitted signatures must not enter training."
        )
