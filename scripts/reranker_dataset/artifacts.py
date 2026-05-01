"""Disabled-by-default artifact cache contracts for row generation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _normalize_mapping(values: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    if not values:
        return ()
    return tuple(sorted((str(key), str(value)) for key, value in values.items()))


def _normalize_candidate_signature_ids(
    values: Mapping[str, Sequence[str]] | None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not values:
        return ()
    return tuple(
        (str(component_key), tuple(str(signature_id) for signature_id in signature_ids))
        for component_key, signature_ids in sorted(values.items())
    )


@dataclass(frozen=True)
class ArtifactCacheKey:
    """Content-addressed key for a reusable reranker-dataset artifact."""

    namespace: str
    dataset_digest: str
    model_digest: str
    suppress_orcid: bool
    feature_schema_digest: str
    feature_preset: str
    constraint_flags: tuple[tuple[str, str], ...] = ()
    candidate_signature_ids: tuple[tuple[str, tuple[str, ...]], ...] = ()
    extra: tuple[tuple[str, str], ...] = ()

    @classmethod
    def build(
        cls,
        *,
        namespace: str,
        dataset_digest: str,
        model_digest: str,
        suppress_orcid: bool,
        feature_schema_digest: str,
        feature_preset: str,
        constraint_flags: Mapping[str, Any] | None = None,
        candidate_signature_ids: Mapping[str, Sequence[str]] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> ArtifactCacheKey:
        """Build a canonical artifact cache key from raw inputs."""

        return cls(
            namespace=str(namespace),
            dataset_digest=str(dataset_digest),
            model_digest=str(model_digest),
            suppress_orcid=bool(suppress_orcid),
            feature_schema_digest=str(feature_schema_digest),
            feature_preset=str(feature_preset),
            constraint_flags=_normalize_mapping(constraint_flags),
            candidate_signature_ids=_normalize_candidate_signature_ids(candidate_signature_ids),
            extra=_normalize_mapping(extra),
        )

    @property
    def digest(self) -> str:
        """Return the stable key digest."""

        return hashlib.sha256(_canonical_json(self.to_json_dict()).encode("utf-8")).hexdigest()

    def to_json_dict(self) -> dict[str, Any]:
        """Return the canonical key payload."""

        return {
            "namespace": self.namespace,
            "dataset_digest": self.dataset_digest,
            "model_digest": self.model_digest,
            "suppress_orcid": bool(self.suppress_orcid),
            "feature_schema_digest": self.feature_schema_digest,
            "feature_preset": self.feature_preset,
            "constraint_flags": list(self.constraint_flags),
            "candidate_signature_ids": [
                [component_key, list(signature_ids)] for component_key, signature_ids in self.candidate_signature_ids
            ],
            "extra": list(self.extra),
        }


@dataclass(frozen=True)
class ArtifactCacheDecision:
    """Result metadata for a cache lookup or write decision."""

    key: ArtifactCacheKey
    hit: bool
    reason: str


class NullArtifactStore:
    """Disabled artifact store used until Phase 3b cache reuse is explicitly enabled."""

    enabled = False

    def get(self, key: ArtifactCacheKey) -> tuple[None, ArtifactCacheDecision]:
        """Return a miss for every lookup."""

        return None, ArtifactCacheDecision(key=key, hit=False, reason="cache_disabled")

    def put(self, key: ArtifactCacheKey, value: Any) -> ArtifactCacheDecision:
        """Ignore writes while preserving auditable cache-decision metadata."""

        del value
        return ArtifactCacheDecision(key=key, hit=False, reason="cache_disabled")
