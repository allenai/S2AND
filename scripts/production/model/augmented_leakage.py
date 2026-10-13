"""Match augmented pair members to their original benchmark identities."""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

PRODUCTION_AUGMENTED_SOURCES = frozenset(
    {"aminer", "arnetminer", "inspire", "kisti", "medline", "pubmed", "qian", "zbmath"}
)


def training_members_from_sources(name: str, files: Mapping[str, Path], *, random_seed: int) -> set[str]:
    """Resolve the trainer's actual training split without reading held-out inputs.

    Args:
        name: Benchmark dataset name.
        files: Model source files containing signatures, papers, and clusters.
        random_seed: Seed used by the production pairwise trainer.

    Returns:
        Signature IDs in the benchmark's training blocks.
    """

    from s2and.data import ANDData

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
    train, _, _ = dataset.split_cluster_signatures()
    return {signature_id for members in train.values() for signature_id in members}


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


def reject_heldout_members(augmented: Mapping[str, set[str]], heldout: Mapping[str, set[str]], *, context: str) -> None:
    """Reject augmented training members present in a held-out population."""
    for name, signature_ids in heldout.items():
        overlap = augmented.get(name, set()) & signature_ids
        if overlap:
            raise ValueError(
                f"{context}: augmented training contains held-out {name!r} signatures: "
                f"count={len(overlap)} sample={sorted(overlap)[:5]}. "
                "Regenerate augmented training pairs using the frozen benchmark training populations."
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
