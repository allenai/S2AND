"""Hold the ORCID exclusions and reference-label reader that several stages share."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from s2and.text import normalize_orcid
from scripts.common.block_roles import normalize_block_key

# The September 25 review identifies mixed identities under these ORCIDs.
# Exclude whole groups because the review does not establish each row's owner.
EXCLUDED_ORCID_GROUPS = frozenset(
    {
        "0000-0000-0000-0001",
        "0000-0001-2345-6789",
        "0000-0001-7511-2910",
        "0000-0002-1028-8971",
        "0000-0002-1825-0097",
        "0000-0002-3752-9683",
        "0000-0002-4011-3590",
        "0000-0002-5024-6186",
        "0000-0002-5571-8349",
        "0000-0002-5982-8983",
        "0000-0003-1855-6181",
        "0000-0003-3393-8460",
        "0000-0003-3895-7409",
        "0009-0004-4499-3929",
        # The November 20 review identifies five more mixed-identity groups.
        "0000-0001-5373-2326",
        "0000-0001-8915-3377",
        "0000-0002-0786-7307",
        "0009-0005-6326-4470",
        "0009-0009-5657-3485",
    }
)


def read_external_reference_identity_evidence(
    path: Path, *, dataset: str | None = None, include_unscored: bool = True, require_scored: bool = False
) -> tuple[dict[str, str], dict[str, str]]:
    """Read all known identities and the subset eligible to score."""
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    required = {"signature_id", "orcid"}
    if require_scored:
        required.add("scored")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"External reference labels missing required columns: {missing}")
    if dataset is not None and "dataset" in frame.columns:
        frame = frame[frame["dataset"].astype(str).map(normalize_block_key).eq(normalize_block_key(dataset))]
    eligible = frame["scored"].eq(True).fillna(False) if "scored" in frame else [True] * len(frame)
    identities: dict[str, str] = {}
    labels: dict[str, str] = {}
    for signature_id, raw_orcid, scored in zip(frame["signature_id"], frame["orcid"], eligible, strict=True):
        if not scored and not include_unscored:
            continue
        if signature_id is None or raw_orcid is None or pd.isna(signature_id) or pd.isna(raw_orcid):
            continue
        sid = str(signature_id)
        orcid = normalize_orcid(str(raw_orcid))
        if not sid or not orcid:
            continue
        existing = identities.get(sid)
        if existing is not None and existing != orcid:
            raise ValueError(f"Conflicting external ORCID labels for signature_id={sid!r}: {existing} vs {orcid}")
        identities[sid] = orcid
        if scored:
            labels[sid] = orcid
    return identities, labels
