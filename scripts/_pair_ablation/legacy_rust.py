"""DANGEROUS, PRACTICE-ONLY access to pre-cutover legacy Arrow artifacts.

This module deliberately calls ``s2and_rust.RustFeaturizer.from_arrow_paths``
without going through S2AND's maintained Arrow validators.  That exception is
ONLY for the one-off pair-ablation rehearsal on the extant, pre-canonical
artifacts.  It MUST NOT be imported by production code, used to build release
artifacts, or treated as a compatibility path.  In particular, do not weaken a
maintained validator to make this adapter reusable.

The adapter refuses a dataset manifest (or referenced directory manifest) that
declares ``canonical_v2``.  Once regenerated artifacts land, this module is the
wrong ingestion path and must fail rather than silently bypass the release
contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from s2and.thread_config import resolve_n_jobs

_CANONICAL_V2 = "canonical_v2"
_REQUIRED_MANIFEST_PATHS = (
    "signatures",
    "papers",
    "paper_authors",
    "specter2",
    "name_counts_index",
)
_DIRECT_OPTIONAL_PATHS = (
    "signatures_batch_index",
    "papers_batch_index",
    "paper_authors_batch_index",
    "cluster_seeds",
    "cluster_seed_disallows",
)
_SPECTER2_BATCH_INDEX = "specter2_batch_index"
_HASH_CHUNK_BYTES = 8 * 1024 * 1024

ValidatedLabeledPair = tuple[str, str, float]


@dataclass(frozen=True)
class LegacyArrowArtifacts:
    """Resolved legacy inputs accepted by this experiment-only adapter."""

    dataset: str
    manifest_path: Path
    manifest_sha256: str
    source_paths: Mapping[str, Path]
    rust_paths: Mapping[str, Path]


@dataclass(frozen=True)
class ArtifactDigest:
    """Content identity for one file or deterministic directory tree."""

    path: Path
    kind: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class PairFeatureMatrices:
    """Rust pair features and the exact ID-to-index mapping used to make them."""

    main: np.ndarray
    labels: np.ndarray
    nameless: np.ndarray | None
    indexed_pairs: tuple[tuple[int, int], ...]
    signature_ids: tuple[str, ...]
    main_feature_indices: tuple[int, ...]
    nameless_feature_indices: tuple[int, ...] | None

    def as_training_tuple(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Return the tuple consumed by the existing pairwise training helpers."""

        return self.main, self.labels, self.nameless


def _sha256_file(path: Path, digest: Any | None = None) -> tuple[str, int]:
    hasher = hashlib.sha256() if digest is None else digest
    size_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            hasher.update(chunk)
            size_bytes += len(chunk)
    return hasher.hexdigest(), size_bytes


def _digest_path(path: Path) -> ArtifactDigest:
    if path.is_file():
        sha256, size_bytes = _sha256_file(path)
        return ArtifactDigest(path=path, kind="file", size_bytes=size_bytes, sha256=sha256)
    if not path.is_dir():
        raise FileNotFoundError(f"Cannot digest missing legacy artifact path: {path}")

    hasher = hashlib.sha256(b"s2and-legacy-directory-sha256-v1\0")
    size_bytes = 0
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        child_size = child.stat().st_size
        hasher.update(relative)
        hasher.update(b"\0")
        hasher.update(str(child_size).encode("ascii"))
        hasher.update(b"\0")
        with child.open("rb") as stream:
            while chunk := stream.read(_HASH_CHUNK_BYTES):
                hasher.update(chunk)
        hasher.update(b"\0")
        size_bytes += child_size
    return ArtifactDigest(path=path, kind="directory", size_bytes=size_bytes, sha256=hasher.hexdigest())


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Legacy Arrow manifest is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Legacy Arrow manifest must contain a JSON object: {path}")
    return payload


def _canonical_v2_declaration(value: Any, location: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key == "normalization_version" and child == _CANONICAL_V2:
                return child_location
            found = _canonical_v2_declaration(child, child_location)
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _canonical_v2_declaration(child, f"{location}[{index}]")
            if found is not None:
                return found
    return None


def _refuse_canonical_v2(payload: Mapping[str, Any], path: Path) -> None:
    location = _canonical_v2_declaration(payload)
    if location is not None:
        raise ValueError(
            "PRACTICE-ONLY legacy adapter refuses canonical_v2 artifacts; use the maintained validated "
            f"Arrow path instead. Declaration {location} is in {path}"
        )


def _resolve_declared_path(manifest_dir: Path, key: str, raw_value: Any) -> Path:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"Legacy Arrow manifest paths[{key!r}] must be a non-empty string")
    candidate = Path(raw_value)
    resolved = (manifest_dir / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Legacy Arrow manifest paths[{key!r}] does not exist: {resolved}")
    return resolved


def resolve_legacy_arrow_manifest(manifest: str | Path) -> LegacyArrowArtifacts:
    """Resolve one pre-cutover manifest with SPECTER2 explicitly selected.

    ``specter2`` is intentionally exposed to Rust as ``specter`` because the
    native API has one embedding slot.  Its batch index is aliased in the same
    way.  The manifest's SPECTER1 entry is never used.
    """

    manifest_path = Path(manifest)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Legacy Arrow manifest does not exist: {manifest_path}")

    payload = _read_json_object(manifest_path)
    _refuse_canonical_v2(payload, manifest_path)
    dataset = payload.get("dataset")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError(f"Legacy Arrow manifest dataset must be a non-empty string: {manifest_path}")
    raw_paths = payload.get("paths")
    if not isinstance(raw_paths, dict):
        raise ValueError(f"Legacy Arrow manifest paths must be a JSON object: {manifest_path}")

    missing = [key for key in _REQUIRED_MANIFEST_PATHS if key not in raw_paths]
    if missing:
        raise ValueError(f"Legacy Arrow manifest is missing required paths: {missing}")

    source_keys = list(_REQUIRED_MANIFEST_PATHS)
    source_keys.extend(key for key in _DIRECT_OPTIONAL_PATHS if key in raw_paths)
    if _SPECTER2_BATCH_INDEX in raw_paths:
        source_keys.append(_SPECTER2_BATCH_INDEX)
    source_paths = {key: _resolve_declared_path(manifest_path.parent, key, raw_paths[key]) for key in source_keys}

    # A newly regenerated directory sidecar is also enough to stop this legacy
    # adapter, even if an old dataset manifest still points at it.
    for path in dict.fromkeys(source_paths.values()):
        child_manifest = path / "manifest.json" if path.is_dir() else None
        if child_manifest is not None and child_manifest.is_file():
            child_payload = _read_json_object(child_manifest)
            _refuse_canonical_v2(child_payload, child_manifest)

    rust_paths = {
        key: source_paths[key]
        for key in (*_REQUIRED_MANIFEST_PATHS[:3], "name_counts_index", *_DIRECT_OPTIONAL_PATHS)
        if key in source_paths
    }
    rust_paths["specter"] = source_paths["specter2"]
    if _SPECTER2_BATCH_INDEX in source_paths:
        rust_paths["specter_batch_index"] = source_paths[_SPECTER2_BATCH_INDEX]

    manifest_sha256, _ = _sha256_file(manifest_path)
    return LegacyArrowArtifacts(
        dataset=dataset,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        source_paths=MappingProxyType(source_paths),
        rust_paths=MappingProxyType(rust_paths),
    )


def _require_current_legacy_artifacts(artifacts: LegacyArrowArtifacts) -> None:
    if not isinstance(artifacts, LegacyArrowArtifacts):
        raise TypeError("artifacts must come from resolve_legacy_arrow_manifest")
    current = resolve_legacy_arrow_manifest(artifacts.manifest_path)
    if (
        current.dataset != artifacts.dataset
        or current.manifest_sha256 != artifacts.manifest_sha256
        or dict(current.source_paths) != dict(artifacts.source_paths)
        or dict(current.rust_paths) != dict(artifacts.rust_paths)
    ):
        raise ValueError(
            "Legacy Arrow manifest changed after it was resolved; resolve it again and record a new artifact identity"
        )


def digest_legacy_artifacts(
    artifacts: LegacyArrowArtifacts,
    *,
    digest_cache: MutableMapping[Path, ArtifactDigest] | None = None,
) -> dict[str, ArtifactDigest]:
    """Hash every source artifact used by Rust, deduplicating aliased paths.

    Directory hashing reads the complete tree and can be expensive for the
    name-count index.  Pass one caller-owned ``digest_cache`` across datasets
    in the same immutable input snapshot to hash shared paths only once.  Cache
    values are still complete SHA-256 content digests, not metadata
    fingerprints.  Clear the cache if any source path may have changed.
    """

    _require_current_legacy_artifacts(artifacts)
    digest_by_path: MutableMapping[Path, ArtifactDigest] = {} if digest_cache is None else digest_cache
    output: dict[str, ArtifactDigest] = {}
    for key, path in artifacts.source_paths.items():
        digest = digest_by_path.get(path)
        if digest is None:
            digest = _digest_path(path)
            digest_by_path[path] = digest
        elif not isinstance(digest, ArtifactDigest) or digest.path != path:
            raise ValueError(f"digest_cache contains an invalid digest for legacy artifact path: {path}")
        output[key] = digest
    return output


def current_artifact_identity(
    artifacts: LegacyArrowArtifacts,
    *,
    include_path_digests: bool = False,
    digest_cache: MutableMapping[Path, ArtifactDigest] | None = None,
) -> dict[str, Any]:
    """Return JSON-serializable provenance for the exact current inputs.

    ``digest_cache`` is used only when ``include_path_digests`` is true.  A
    caller can reuse it across datasets whose manifests reference shared files
    or directories; see :func:`digest_legacy_artifacts` for cache lifetime.
    """

    _require_current_legacy_artifacts(artifacts)
    identity: dict[str, Any] = {
        "adapter": "practice_only_legacy_arrow_rust_v1",
        "dataset": artifacts.dataset,
        "manifest": {
            "path": str(artifacts.manifest_path),
            "sha256": artifacts.manifest_sha256,
        },
        "source_paths": {key: str(path) for key, path in artifacts.source_paths.items()},
        "rust_paths": {key: str(path) for key, path in artifacts.rust_paths.items()},
        "embedding_alias": {
            "manifest_path_key": "specter2",
            "rust_path_key": "specter",
        },
    }
    if include_path_digests:
        identity["path_digests"] = {
            key: {
                "path": str(value.path),
                "kind": value.kind,
                "size_bytes": value.size_bytes,
                "sha256": value.sha256,
            }
            for key, value in digest_legacy_artifacts(artifacts, digest_cache=digest_cache).items()
        }
    return identity


def _normalize_signature_id(value: Any, *, context: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str | Integral):
        raise TypeError(f"{context} must be a string or integer signature ID, got {type(value).__name__}")
    normalized = str(value)
    if not normalized.strip():
        raise ValueError(f"{context} must not be empty or whitespace")
    return normalized


def validate_labeled_pairs(pairs: Iterable[Sequence[Any]]) -> tuple[ValidatedLabeledPair, ...]:
    """Normalize IDs and require finite binary labels without changing row order."""

    validated: list[ValidatedLabeledPair] = []
    for row_index, pair in enumerate(pairs):
        if isinstance(pair, str | bytes) or not isinstance(pair, Sequence) or len(pair) != 3:
            raise ValueError(f"Labeled pair row {row_index} must contain exactly (left_id, right_id, label)")
        left = _normalize_signature_id(pair[0], context=f"Labeled pair row {row_index} left_id")
        right = _normalize_signature_id(pair[1], context=f"Labeled pair row {row_index} right_id")
        if left == right:
            raise ValueError(f"Labeled pair row {row_index} is a self-pair for signature ID {left!r}")
        raw_label = pair[2]
        if not isinstance(raw_label, Real):
            raise TypeError(f"Labeled pair row {row_index} label must be numeric, got {type(raw_label).__name__}")
        label = float(raw_label)
        if not math.isfinite(label) or label not in (0.0, 1.0):
            raise ValueError(f"Labeled pair row {row_index} label must be finite and binary, got {raw_label!r}")
        validated.append((left, right, label))
    return tuple(validated)


def signature_ids_for_labeled_pairs(pairs: Iterable[Sequence[Any]]) -> tuple[str, ...]:
    """Return first-seen unique signature IDs from validated pair rows."""

    validated = validate_labeled_pairs(pairs)
    seen: set[str] = set()
    ordered: list[str] = []
    for left, right, _ in validated:
        for signature_id in (left, right):
            if signature_id not in seen:
                seen.add(signature_id)
                ordered.append(signature_id)
    return tuple(ordered)


def _load_s2and_rust() -> Any:
    import s2and_rust

    return s2and_rust


def build_legacy_rust_featurizer(
    artifacts: LegacyArrowArtifacts,
    *,
    n_jobs: int,
    signature_ids: Iterable[Any] | None = None,
    name_tuples: Any = "filtered",
) -> Any:
    """Build Rust directly from legacy Arrow, always with preprocessing off."""

    _require_current_legacy_artifacts(artifacts)
    selected_ids: list[str] | None = None
    if signature_ids is not None:
        selected_ids = [
            _normalize_signature_id(value, context=f"signature_ids[{index}]")
            for index, value in enumerate(signature_ids)
        ]
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("signature_ids must not contain duplicates")
    threads = resolve_n_jobs(n_jobs)
    return _load_s2and_rust().RustFeaturizer.from_arrow_paths(
        {key: str(path) for key, path in artifacts.rust_paths.items()},
        signature_ids=selected_ids,
        name_tuples=name_tuples,
        preprocess=False,
        num_threads=threads,
    )


def feature_indices(featurization_info: Any, *, context: str) -> tuple[int, ...]:
    """Resolve a FeaturizationInfo-like object to sorted native columns."""

    features_to_use = getattr(featurization_info, "features_to_use", None)
    group_to_index = getattr(featurization_info, "feature_group_to_index", None)
    if isinstance(features_to_use, str | bytes) or not isinstance(features_to_use, Sequence):
        raise TypeError(f"{context}.features_to_use must be a sequence")
    if not isinstance(group_to_index, Mapping):
        raise TypeError(f"{context}.feature_group_to_index must be a mapping")
    selected: set[int] = set()
    for group in features_to_use:
        if not isinstance(group, str) or group not in group_to_index:
            raise ValueError(f"{context} contains unknown feature group {group!r}")
        group_indices = group_to_index[group]
        if isinstance(group_indices, str | bytes) or not isinstance(group_indices, Sequence):
            raise TypeError(f"{context}.feature_group_to_index[{group!r}] must be a sequence")
        for index in group_indices:
            if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
                raise ValueError(f"{context} feature index must be a non-negative integer, got {index!r}")
            selected.add(int(index))
    return tuple(sorted(selected))


def featurize_labeled_pairs(
    rust_featurizer: Any,
    pairs: Iterable[Sequence[Any]],
    *,
    featurization_info: Any,
    nameless_featurization_info: Any | None,
    n_jobs: int,
    nan_value: float = math.nan,
) -> PairFeatureMatrices:
    """Index arbitrary labeled IDs and make main and nameless Rust matrices."""

    validated_pairs = validate_labeled_pairs(pairs)
    main_indices = feature_indices(featurization_info, context="featurization_info")
    nameless_indices = (
        None
        if nameless_featurization_info is None
        else feature_indices(nameless_featurization_info, context="nameless_featurization_info")
    )

    raw_signature_ids = tuple(rust_featurizer.signature_ids())
    signature_ids = tuple(
        _normalize_signature_id(value, context=f"rust_featurizer.signature_ids()[{index}]")
        for index, value in enumerate(raw_signature_ids)
    )
    if len(signature_ids) != len(set(signature_ids)):
        raise ValueError("Rust featurizer returned duplicate signature IDs")
    index_by_id = {signature_id: index for index, signature_id in enumerate(signature_ids)}
    required_ids = {signature_id for left, right, _ in validated_pairs for signature_id in (left, right)}
    missing_ids = sorted(required_ids - index_by_id.keys())
    if missing_ids:
        raise KeyError(f"Labeled pairs reference signature IDs absent from Rust featurizer: {missing_ids[:10]}")

    indexed_pairs = tuple((index_by_id[left], index_by_id[right]) for left, right, _ in validated_pairs)
    labels = np.asarray([label for _, _, label in validated_pairs], dtype=np.float64)
    union_indices = tuple(sorted(set(main_indices).union(() if nameless_indices is None else nameless_indices)))
    if not indexed_pairs:
        combined = np.empty((0, len(union_indices)), dtype=np.float64)
    else:
        combined = np.asarray(
            rust_featurizer.featurize_pairs_matrix_indexed(
                list(indexed_pairs),
                list(union_indices),
                resolve_n_jobs(n_jobs),
                float(nan_value),
            ),
            dtype=np.float64,
        )
        expected_shape = (len(indexed_pairs), len(union_indices))
        if combined.shape != expected_shape:
            raise ValueError(
                f"Rust pair feature matrix has shape {combined.shape}; expected {expected_shape} for selected indices"
            )

    position_by_index = {feature_index: position for position, feature_index in enumerate(union_indices)}
    main = combined[:, [position_by_index[index] for index in main_indices]]
    nameless = (
        None if nameless_indices is None else combined[:, [position_by_index[index] for index in nameless_indices]]
    )
    return PairFeatureMatrices(
        main=main,
        labels=labels,
        nameless=nameless,
        indexed_pairs=indexed_pairs,
        signature_ids=signature_ids,
        main_feature_indices=main_indices,
        nameless_feature_indices=nameless_indices,
    )


__all__ = [
    "ArtifactDigest",
    "LegacyArrowArtifacts",
    "PairFeatureMatrices",
    "build_legacy_rust_featurizer",
    "current_artifact_identity",
    "digest_legacy_artifacts",
    "feature_indices",
    "featurize_labeled_pairs",
    "resolve_legacy_arrow_manifest",
    "signature_ids_for_labeled_pairs",
    "validate_labeled_pairs",
]
