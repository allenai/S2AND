"""Disabled-by-default artifact cache contracts for row generation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SAFE_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


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


class FilesystemArtifactStore:
    """Local content-addressed artifact store for reranker-dataset artifacts."""

    enabled = True

    def __init__(self, root: Path | str) -> None:
        """Initialize the store under a local filesystem root."""

        self.root = Path(root)

    def get(self, key: ArtifactCacheKey) -> tuple[bytes | None, ArtifactCacheDecision]:
        """Read artifact bytes when present and valid, otherwise return a miss."""

        artifact_path = self.artifact_path(key)
        metadata_path = self.metadata_path(key)
        if not artifact_path.exists() and not metadata_path.exists():
            return None, ArtifactCacheDecision(key=key, hit=False, reason="cache_miss")
        if not artifact_path.exists() or not metadata_path.exists():
            return None, ArtifactCacheDecision(key=key, hit=False, reason="cache_incomplete")

        payload = artifact_path.read_bytes()
        self._validate_metadata(key=key, payload=payload, metadata=self._read_metadata(metadata_path))
        return payload, ArtifactCacheDecision(key=key, hit=True, reason="cache_hit")

    def put(
        self,
        key: ArtifactCacheKey,
        value: bytes,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactCacheDecision:
        """Write artifact bytes and validating metadata under the key digest."""

        self.write_bytes(key, value, metadata=metadata)
        return ArtifactCacheDecision(key=key, hit=False, reason="cache_written")

    def read_bytes(self, key: ArtifactCacheKey) -> bytes:
        """Read artifact bytes and raise when the artifact is absent or invalid."""

        value, decision = self.get(key)
        if not decision.hit or value is None:
            msg = f"Artifact cache miss for {key.digest}: {decision.reason}"
            raise FileNotFoundError(msg)
        return value

    def write_bytes(
        self,
        key: ArtifactCacheKey,
        value: bytes,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Write artifact bytes with a JSON metadata sidecar."""

        if not isinstance(value, bytes):
            raise TypeError("FilesystemArtifactStore only stores bytes payloads")

        directory = self.artifact_dir(key)
        directory.mkdir(parents=True, exist_ok=True)
        self.artifact_path(key).write_bytes(value)
        self.metadata_path(key).write_text(
            json.dumps(
                self._metadata_payload(key=key, payload=value, metadata=metadata),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def artifact_dir(self, key: ArtifactCacheKey) -> Path:
        """Return the content-addressed artifact directory for a key."""

        namespace = _validate_namespace(key.namespace)
        return self.root / namespace / key.digest[:2] / key.digest

    def artifact_path(self, key: ArtifactCacheKey) -> Path:
        """Return the payload path for a key."""

        return self.artifact_dir(key) / "artifact.bin"

    def metadata_path(self, key: ArtifactCacheKey) -> Path:
        """Return the metadata sidecar path for a key."""

        return self.artifact_dir(key) / "metadata.json"

    def _metadata_payload(
        self,
        *,
        key: ArtifactCacheKey,
        payload: bytes,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "digest": key.digest,
            "key": key.to_json_dict(),
            "payload_digest": hashlib.sha256(payload).hexdigest(),
            "metadata": dict(metadata or {}),
        }

    def _read_metadata(self, path: Path) -> dict[str, Any]:
        raw_metadata = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_metadata, dict):
            raise ValueError("Artifact metadata must be a JSON object")
        return raw_metadata

    def _validate_metadata(
        self,
        *,
        key: ArtifactCacheKey,
        payload: bytes,
        metadata: Mapping[str, Any],
    ) -> None:
        if metadata.get("digest") != key.digest:
            raise ValueError("Artifact metadata digest mismatch")
        if metadata.get("key") != key.to_json_dict():
            raise ValueError("Artifact metadata key mismatch")
        if metadata.get("payload_digest") != hashlib.sha256(payload).hexdigest():
            raise ValueError("Artifact payload digest mismatch")
        if not isinstance(metadata.get("metadata", {}), dict):
            raise ValueError("Artifact metadata payload must be a JSON object")


def _validate_namespace(namespace: str) -> str:
    if not _SAFE_NAMESPACE_RE.fullmatch(namespace):
        raise ValueError(f"Artifact namespace is not filesystem-safe: {namespace!r}")
    return namespace
