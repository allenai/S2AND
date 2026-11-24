"""Match augmented pair members to their original benchmark identities."""

from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping


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
