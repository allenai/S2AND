"""Match augmented pair members to their original benchmark identities."""

from collections import defaultdict
from collections.abc import Iterable, Mapping


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
