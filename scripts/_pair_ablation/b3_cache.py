"""Content-addressed raw Rust feature caches for pair-ablation B-cubed evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from s2and import model as model_module
from s2and.consts import LARGE_INTEGER
from s2and.incremental_linking.policy import require_rust_featurizer_name_counts_binding_for_clusterer
from s2and.rust_calls import (
    build_block_upper_triangle_feature_matrix_indexed_rust,
    get_constraints_block_upper_triangle_indexed_rust,
)
from scripts._pair_ablation.evaluation import B3EvaluationPlan

B3_RAW_FEATURE_CACHE_SCHEMA = "pair_ablation_b3_raw_features_v2"
B3_CACHE_BUILDER_IDENTITY_SCHEMA = "pair_ablation_b3_cache_builder_v1"
_ARRAY_FILENAMES = ("main.npy", "nameless.npy", "staged_labels.npy")
_CACHE_FILENAMES = frozenset((*_ARRAY_FILENAMES, "layout.json", "manifest.json"))


@dataclass(frozen=True, slots=True)
class B3BlockLayout:
    """One block's exact signatures and row span in cached condensed matrices."""

    block_key: str
    signatures: tuple[str, ...]
    row_start: int
    row_stop: int

    @property
    def pair_count(self) -> int:
        """Return the number of condensed upper-triangle rows for this block."""

        return self.row_stop - self.row_start

    def payload(self) -> dict[str, object]:
        """Return the JSON representation stored with the cache."""

        return {
            "block_key": self.block_key,
            "signatures": list(self.signatures),
            "row_start": self.row_start,
            "row_stop": self.row_stop,
        }


@dataclass(frozen=True, slots=True)
class B3RawFeatureStore:
    """Strictly validated raw matrices that can be rescored by many pairwise models."""

    cache_dir: Path
    cache_digest: str
    cache_identity: dict[str, object]
    plan: B3EvaluationPlan
    layout: tuple[B3BlockLayout, ...]
    main: np.ndarray
    nameless: np.ndarray
    staged_labels: np.ndarray


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_value(value: object) -> object:
    return json.loads(_canonical_json_bytes(value))


def _json_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def b3_cache_builder_identity(
    *,
    implementation_sha256: Mapping[str, str],
    runtime_versions: Mapping[str, str],
) -> str:
    """Digest the full implementation and runtime identity that builds raw features."""

    if not implementation_sha256:
        raise ValueError("implementation_sha256 must not be empty")
    if not runtime_versions:
        raise ValueError("runtime_versions must not be empty")
    for path, digest in implementation_sha256.items():
        if not isinstance(path, str) or not path:
            raise ValueError("implementation_sha256 keys must be non-empty strings")
        _require_sha256(digest, context=f"implementation_sha256[{path!r}]")
    for package, version in runtime_versions.items():
        if not isinstance(package, str) or not package or not isinstance(version, str) or not version:
            raise ValueError("runtime_versions must contain non-empty string keys and values")
    return _json_digest(
        {
            "schema": B3_CACHE_BUILDER_IDENTITY_SCHEMA,
            "implementation_sha256": dict(implementation_sha256),
            "runtime_versions": dict(runtime_versions),
        }
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is not allowed: {value}")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Invalid JSON cache artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Cache JSON artifact must contain an object: {path}")
    return payload


def _feature_indices(featurizer_info: Any, *, context: str) -> tuple[int, ...]:
    if featurizer_info is None:
        raise ValueError(f"{context} is required for the B3 raw-feature cache")
    indices = tuple(int(index) for index in model_module._selected_feature_indices(featurizer_info))
    if not indices:
        raise ValueError(f"{context} selects no feature columns")
    return indices


def _feature_view_identity(clusterer: Any) -> dict[str, object]:
    main_info = getattr(clusterer, "featurizer_info", None)
    nameless_info = getattr(clusterer, "nameless_featurizer_info", None)
    return {
        "main": {
            "featurizer_version": str(getattr(main_info, "featurizer_version", "")),
            "selected_indices": list(_feature_indices(main_info, context="clusterer.featurizer_info")),
        },
        "nameless": {
            "featurizer_version": str(getattr(nameless_info, "featurizer_version", "")),
            "selected_indices": list(_feature_indices(nameless_info, context="clusterer.nameless_featurizer_info")),
        },
    }


def _constraint_policy_identity(clusterer: Any) -> dict[str, object]:
    return {
        "use_default_constraints_as_supervision": bool(
            getattr(clusterer, "use_default_constraints_as_supervision", True)
        ),
        "dont_merge_cluster_seeds": bool(getattr(clusterer, "dont_merge_cluster_seeds", True)),
        "suppress_orcid": bool(getattr(clusterer, "suppress_orcid", False)),
        "incremental_dont_use_cluster_seeds": False,
        "partial_supervision": "none",
        "label_staging": "constraint_distance_minus_large_integer",
        "large_integer": float(LARGE_INTEGER),
    }


def _cache_identity(
    *,
    plan: B3EvaluationPlan,
    feature_artifact_identity: Mapping[str, Any],
    rust_featurizer_identity: Mapping[str, Any],
    clusterer: Any,
    rust_version: str,
    rust_extension_sha256: str,
    cache_builder_identity: str,
) -> dict[str, object]:
    if not rust_version:
        raise ValueError("rust_version must not be empty")
    return {
        "schema": B3_RAW_FEATURE_CACHE_SCHEMA,
        "evaluation_plan_digest": plan.plan_digest,
        "evaluation_plan": plan.identity_payload(),
        "feature_artifact_identity": _canonical_json_value(dict(feature_artifact_identity)),
        "rust_featurizer_identity": _canonical_json_value(dict(rust_featurizer_identity)),
        "rust_version": str(rust_version),
        "rust_extension_sha256": _require_sha256(
            rust_extension_sha256,
            context="rust_extension_sha256",
        ),
        "cache_builder_identity": _require_sha256(
            cache_builder_identity,
            context="cache_builder_identity",
        ),
        "feature_views": _feature_view_identity(clusterer),
        "constraint_policy": _constraint_policy_identity(clusterer),
    }


def _layout_for_plan(plan: B3EvaluationPlan) -> tuple[B3BlockLayout, ...]:
    layout: list[B3BlockLayout] = []
    row_start = 0
    for block in plan.blocks:
        pair_count = len(block.signatures) * (len(block.signatures) - 1) // 2
        row_stop = row_start + pair_count
        layout.append(B3BlockLayout(block.block_key, block.signatures, row_start, row_stop))
        row_start = row_stop
    return tuple(layout)


def _layout_payload(plan: B3EvaluationPlan, layout: tuple[B3BlockLayout, ...]) -> dict[str, object]:
    return {
        "schema": B3_RAW_FEATURE_CACHE_SCHEMA,
        "evaluation_plan_digest": plan.plan_digest,
        "row_count": layout[-1].row_stop if layout else 0,
        "blocks": [block.payload() for block in layout],
    }


def _cached_feature_indices(cache_identity: Mapping[str, object]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    feature_views = cache_identity.get("feature_views")
    if not isinstance(feature_views, dict):
        raise RuntimeError("B3 raw-feature cache identity has malformed feature views")
    typed_feature_views = cast(dict[str, object], feature_views)

    def selected(view_name: str) -> tuple[int, ...]:
        view = typed_feature_views.get(view_name)
        if not isinstance(view, dict):
            raise RuntimeError(f"B3 raw-feature cache identity has malformed {view_name} view")
        raw_indices = cast(dict[str, object], view).get("selected_indices")
        if not isinstance(raw_indices, list) or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in raw_indices
        ):
            raise RuntimeError(f"B3 raw-feature cache identity has invalid {view_name} feature indices")
        indices = tuple(cast(list[int], raw_indices))
        if not indices or len(indices) != len(set(indices)):
            raise RuntimeError(f"B3 raw-feature cache identity has invalid {view_name} feature indices")
        return indices

    return selected("main"), selected("nameless")


def _validate_feature_chunk(
    values: np.ndarray,
    *,
    expected_rows: int,
    expected_columns: int,
    context: str,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64, order="C")
    expected_shape = (expected_rows, expected_columns)
    if matrix.shape != expected_shape:
        raise RuntimeError(f"{context} has shape {matrix.shape}; expected {expected_shape}")
    if np.isinf(matrix).any():
        raise RuntimeError(f"{context} contains infinite feature values")
    return matrix


def _build_cache_files(
    cache_dir: Path,
    *,
    plan: B3EvaluationPlan,
    layout: tuple[B3BlockLayout, ...],
    cache_digest: str,
    cache_identity: dict[str, object],
    rust_featurizer: Any,
    clusterer: Any,
    pair_chunk_size: int,
) -> None:
    if pair_chunk_size <= 0:
        raise ValueError("pair_chunk_size must be positive")
    if bool(getattr(clusterer, "use_cache", False)):
        raise ValueError("B3 raw-feature caching requires clusterer.use_cache=False")
    if getattr(clusterer, "nameless_classifier", None) is None:
        raise ValueError("B3 raw-feature caching requires a nameless classifier")

    require_rust_featurizer_name_counts_binding_for_clusterer(
        clusterer,
        rust_featurizer,
        context="pair-ablation B3 raw-feature cache",
    )
    signature_index_by_id = model_module._build_signature_index_by_id(rust_featurizer)
    main_indices, nameless_indices = _cached_feature_indices(cache_identity)
    row_count = layout[-1].row_stop if layout else 0
    main_output = np.lib.format.open_memmap(
        cache_dir / "main.npy",
        mode="w+",
        dtype=np.float64,
        shape=(row_count, len(main_indices)),
    )
    nameless_output = np.lib.format.open_memmap(
        cache_dir / "nameless.npy",
        mode="w+",
        dtype=np.float64,
        shape=(row_count, len(nameless_indices)),
    )
    label_output = np.lib.format.open_memmap(
        cache_dir / "staged_labels.npy",
        mode="w+",
        dtype=np.float64,
        shape=(row_count,),
    )
    use_constraints = bool(getattr(clusterer, "use_default_constraints_as_supervision", True))
    try:
        for block in layout:
            if block.pair_count == 0:
                continue
            try:
                block_signature_indices = [
                    int(signature_index_by_id[signature_id]) for signature_id in block.signatures
                ]
            except KeyError as exc:
                raise ValueError(
                    f"B3 evaluation plan references a signature absent from Rust: {exc.args[0]!r}"
                ) from exc
            block_offset = 0
            while block_offset < block.pair_count:
                chunk_rows = min(pair_chunk_size, block.pair_count - block_offset)
                staged_labels = np.full(chunk_rows, np.nan, dtype=np.float64)
                if use_constraints:
                    left, right, constraint_values = get_constraints_block_upper_triangle_indexed_rust(
                        block_signature_indices,
                        start_offset=block_offset,
                        max_pairs=chunk_rows,
                        dont_merge_cluster_seeds=bool(getattr(clusterer, "dont_merge_cluster_seeds", True)),
                        incremental_dont_use_cluster_seeds=False,
                        num_threads=int(clusterer.n_jobs),
                        featurizer=rust_featurizer,
                        suppress_orcid=bool(getattr(clusterer, "suppress_orcid", False)),
                    )
                    if len(left) != chunk_rows or len(right) != chunk_rows or len(constraint_values) != chunk_rows:
                        raise RuntimeError(
                            "Rust constraint row count mismatch while building B3 raw cache: "
                            f"block={block.block_key!r} offset={block_offset} expected={chunk_rows} "
                            f"left={len(left)} right={len(right)} values={len(constraint_values)}"
                        )
                    for row_offset, value in enumerate(constraint_values):
                        if value is not None:
                            staged_labels[row_offset] = float(value - LARGE_INTEGER)
                main_chunk = _validate_feature_chunk(
                    build_block_upper_triangle_feature_matrix_indexed_rust(
                        block_signature_indices,
                        start_offset=block_offset,
                        max_pairs=chunk_rows,
                        selected_indices=list(main_indices),
                        num_threads=int(clusterer.n_jobs),
                        nan_value=np.nan,
                        featurizer=rust_featurizer,
                    ),
                    expected_rows=chunk_rows,
                    expected_columns=len(main_indices),
                    context=f"main B3 features block={block.block_key!r} offset={block_offset}",
                )
                nameless_chunk = _validate_feature_chunk(
                    build_block_upper_triangle_feature_matrix_indexed_rust(
                        block_signature_indices,
                        start_offset=block_offset,
                        max_pairs=chunk_rows,
                        selected_indices=list(nameless_indices),
                        num_threads=int(clusterer.n_jobs),
                        nan_value=np.nan,
                        featurizer=rust_featurizer,
                    ),
                    expected_rows=chunk_rows,
                    expected_columns=len(nameless_indices),
                    context=f"nameless B3 features block={block.block_key!r} offset={block_offset}",
                )
                row_start = block.row_start + block_offset
                row_stop = row_start + chunk_rows
                main_output[row_start:row_stop] = main_chunk
                nameless_output[row_start:row_stop] = nameless_chunk
                label_output[row_start:row_stop] = staged_labels
                block_offset += chunk_rows
        main_output.flush()
        nameless_output.flush()
        label_output.flush()
    finally:
        del main_output, nameless_output, label_output

    layout_payload = _layout_payload(plan, layout)
    _write_json(cache_dir / "layout.json", layout_payload)
    arrays = {
        "main.npy": {
            "sha256": _sha256_file(cache_dir / "main.npy"),
            "shape": [row_count, len(main_indices)],
            "dtype": "float64",
        },
        "nameless.npy": {
            "sha256": _sha256_file(cache_dir / "nameless.npy"),
            "shape": [row_count, len(nameless_indices)],
            "dtype": "float64",
        },
        "staged_labels.npy": {
            "sha256": _sha256_file(cache_dir / "staged_labels.npy"),
            "shape": [row_count],
            "dtype": "float64",
        },
    }
    _write_json(
        cache_dir / "manifest.json",
        {
            "schema": B3_RAW_FEATURE_CACHE_SCHEMA,
            "cache_digest": cache_digest,
            "cache_identity": cache_identity,
            "evaluation_plan_digest": plan.plan_digest,
            "layout_sha256": _sha256_file(cache_dir / "layout.json"),
            "arrays": arrays,
        },
    )


def _contains_infinite(values: np.ndarray, *, row_chunk_size: int = 100_000) -> bool:
    for row_start in range(0, len(values), row_chunk_size):
        if np.isinf(values[row_start : row_start + row_chunk_size]).any():
            return True
    return False


def _load_b3_raw_feature_store(
    cache_dir: Path,
    *,
    plan: B3EvaluationPlan,
    cache_digest: str,
    cache_identity: dict[str, object],
) -> B3RawFeatureStore:
    if not cache_dir.is_dir():
        raise RuntimeError(f"B3 raw-feature cache directory is missing: {cache_dir}")
    actual_files = frozenset(path.name for path in cache_dir.iterdir() if path.is_file())
    if actual_files != _CACHE_FILENAMES:
        raise RuntimeError(
            f"B3 raw-feature cache file set mismatch: expected={sorted(_CACHE_FILENAMES)} "
            f"actual={sorted(actual_files)} path={cache_dir}"
        )
    manifest = _read_json_object(cache_dir / "manifest.json")
    if manifest.get("schema") != B3_RAW_FEATURE_CACHE_SCHEMA:
        raise RuntimeError(f"B3 raw-feature cache schema mismatch: {cache_dir}")
    if manifest.get("cache_digest") != cache_digest or cache_dir.name != cache_digest:
        raise RuntimeError(f"B3 raw-feature cache digest/path mismatch: {cache_dir}")
    if manifest.get("cache_identity") != cache_identity:
        raise RuntimeError(f"B3 raw-feature cache identity mismatch: {cache_dir}")
    if manifest.get("evaluation_plan_digest") != plan.plan_digest:
        raise RuntimeError(f"B3 raw-feature cache evaluation-plan mismatch: {cache_dir}")

    layout_path = cache_dir / "layout.json"
    if manifest.get("layout_sha256") != _sha256_file(layout_path):
        raise RuntimeError(f"B3 raw-feature cache layout hash mismatch: {layout_path}")
    layout = _layout_for_plan(plan)
    if _read_json_object(layout_path) != _layout_payload(plan, layout):
        raise RuntimeError(f"B3 raw-feature cache layout content mismatch: {layout_path}")
    row_count = layout[-1].row_stop if layout else 0

    arrays_metadata = manifest.get("arrays")
    if not isinstance(arrays_metadata, dict) or set(arrays_metadata) != set(_ARRAY_FILENAMES):
        raise RuntimeError(f"B3 raw-feature cache array manifest is malformed: {cache_dir}")
    main_indices, nameless_indices = _cached_feature_indices(cache_identity)
    expected_shapes = {
        "main.npy": (row_count, len(main_indices)),
        "nameless.npy": (row_count, len(nameless_indices)),
        "staged_labels.npy": (row_count,),
    }
    loaded: dict[str, np.ndarray] = {}
    for filename in _ARRAY_FILENAMES:
        path = cache_dir / filename
        metadata = arrays_metadata[filename]
        if not isinstance(metadata, dict):
            raise RuntimeError(f"B3 raw-feature cache metadata is malformed: {path}")
        if metadata.get("sha256") != _sha256_file(path):
            raise RuntimeError(f"B3 raw-feature cache array hash mismatch: {path}")
        expected_shape = expected_shapes[filename]
        if metadata.get("shape") != list(expected_shape) or metadata.get("dtype") != "float64":
            raise RuntimeError(f"B3 raw-feature cache array metadata mismatch: {path}")
        try:
            values = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot load B3 raw-feature cache array: {path}") from exc
        if values.shape != expected_shape or values.dtype != np.dtype(np.float64):
            raise RuntimeError(
                f"B3 raw-feature cache array shape/dtype mismatch: path={path} "
                f"shape={values.shape} dtype={values.dtype}"
            )
        if not values.flags.c_contiguous:
            raise RuntimeError(f"B3 raw-feature cache array must be C-contiguous: {path}")
        if _contains_infinite(values):
            raise RuntimeError(f"B3 raw-feature cache array contains infinite values: {path}")
        loaded[filename] = values
    return B3RawFeatureStore(
        cache_dir=cache_dir,
        cache_digest=cache_digest,
        cache_identity=cache_identity,
        plan=plan,
        layout=layout,
        main=loaded["main.npy"],
        nameless=loaded["nameless.npy"],
        staged_labels=loaded["staged_labels.npy"],
    )


def build_or_load_b3_raw_feature_store(
    cache_root: Path,
    *,
    plan: B3EvaluationPlan,
    rust_featurizer: Any,
    feature_artifact_identity: Mapping[str, Any],
    rust_featurizer_identity: Mapping[str, Any],
    clusterer: Any,
    rust_version: str,
    rust_extension_sha256: str,
    cache_builder_identity: str,
    pair_chunk_size: int = 1_000_000,
    validated_stores: MutableMapping[str, B3RawFeatureStore] | None = None,
) -> B3RawFeatureStore:
    """Build or strictly load a model-independent raw B3 feature cache."""

    cache_identity = _cache_identity(
        plan=plan,
        feature_artifact_identity=feature_artifact_identity,
        rust_featurizer_identity=rust_featurizer_identity,
        clusterer=clusterer,
        rust_version=rust_version,
        rust_extension_sha256=rust_extension_sha256,
        cache_builder_identity=cache_builder_identity,
    )
    cache_digest = _json_digest(cache_identity)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_root / cache_digest
    if validated_stores is not None and cache_digest in validated_stores:
        store = validated_stores[cache_digest]
        if (
            store.cache_digest != cache_digest
            or store.cache_identity != cache_identity
            or store.plan.plan_digest != plan.plan_digest
            or store.cache_dir.resolve() != cache_dir.resolve()
        ):
            raise RuntimeError(f"Memoized B3 raw-feature store identity mismatch: {cache_digest}")
        return store
    if cache_dir.exists():
        store = _load_b3_raw_feature_store(
            cache_dir,
            plan=plan,
            cache_digest=cache_digest,
            cache_identity=cache_identity,
        )
    else:
        layout = _layout_for_plan(plan)
        with tempfile.TemporaryDirectory(
            prefix=f".{cache_digest}.tmp-{os.getpid()}-",
            dir=cache_root,
        ) as raw_temp_dir:
            temporary_dir = Path(raw_temp_dir)
            _build_cache_files(
                temporary_dir,
                plan=plan,
                layout=layout,
                cache_digest=cache_digest,
                cache_identity=cache_identity,
                rust_featurizer=rust_featurizer,
                clusterer=clusterer,
                pair_chunk_size=pair_chunk_size,
            )
            try:
                temporary_dir.replace(cache_dir)
            except OSError:
                if not cache_dir.exists():
                    raise
        store = _load_b3_raw_feature_store(
            cache_dir,
            plan=plan,
            cache_digest=cache_digest,
            cache_identity=cache_identity,
        )
    if validated_stores is not None:
        validated_stores[cache_digest] = store
    return store


def score_b3_raw_feature_store(
    store: B3RawFeatureStore,
    *,
    clusterer: Any,
    total_ram_bytes: int,
) -> dict[str, np.ndarray]:
    """Score cached raw views through canonical S2AND prediction semantics."""

    if total_ram_bytes <= 0:
        raise ValueError("total_ram_bytes must be positive")
    if getattr(clusterer, "nameless_classifier", None) is None:
        raise ValueError("Cached B3 scoring requires a nameless classifier")
    expected_feature_views = store.cache_identity.get("feature_views")
    if expected_feature_views != _feature_view_identity(clusterer):
        raise ValueError("Clusterer feature views do not match the B3 raw-feature cache")
    expected_policy = store.cache_identity.get("constraint_policy")
    if expected_policy != _constraint_policy_identity(clusterer):
        raise ValueError("Clusterer constraint policy does not match the B3 raw-feature cache")
    predictions, _ = model_module._predict_and_combine(
        clusterer.classifier,
        clusterer.nameless_classifier,
        store.main,
        store.staged_labels,
        store.nameless,
        store.cache_digest,
        num_threads=int(clusterer.n_jobs),
        total_ram_bytes=total_ram_bytes,
    )
    distances = np.asarray(predictions, dtype=np.float64)
    if distances.shape != store.staged_labels.shape:
        raise RuntimeError(f"Cached B3 prediction shape mismatch: {distances.shape} != {store.staged_labels.shape}")
    if not np.isfinite(distances).all():
        raise RuntimeError("Cached B3 prediction produced non-finite distances")
    return {block.block_key: distances[block.row_start : block.row_stop] for block in store.layout}


__all__ = [
    "B3_CACHE_BUILDER_IDENTITY_SCHEMA",
    "B3BlockLayout",
    "B3RawFeatureStore",
    "b3_cache_builder_identity",
    "build_or_load_b3_raw_feature_store",
    "score_b3_raw_feature_store",
]
