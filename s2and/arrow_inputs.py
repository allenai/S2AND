"""Canonical validation helpers for Arrow-backed runtime inputs."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from s2and.consts import NORMALIZATION_VERSION_LEGACY_COMPAT, VALID_NORMALIZATION_VERSIONS


@dataclass(frozen=True)
class ArrowBatchIndexContract:
    """One Arrow table and its raw-planner batch lookup index contract."""

    table_key: str
    key_column: str
    index_key: str
    max_record_batch_rows: int


RAW_PLANNER_ARROW_BATCH_INDEX_CONTRACTS = (
    ArrowBatchIndexContract("signatures", "signature_id", "signatures_batch_index", 16_384),
    ArrowBatchIndexContract("papers", "paper_id", "papers_batch_index", 16_384),
    ArrowBatchIndexContract("paper_authors", "paper_id", "paper_authors_batch_index", 16_384),
    ArrowBatchIndexContract("specter", "paper_id", "specter_batch_index", 2_048),
)
RAW_PLANNER_ARROW_KEY_COLUMNS = {
    contract.table_key: contract.key_column for contract in RAW_PLANNER_ARROW_BATCH_INDEX_CONTRACTS
}
RAW_PLANNER_ARROW_BATCH_INDEX_KEYS = {
    contract.table_key: contract.index_key for contract in RAW_PLANNER_ARROW_BATCH_INDEX_CONTRACTS
}
RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS = {
    contract.table_key: contract.max_record_batch_rows for contract in RAW_PLANNER_ARROW_BATCH_INDEX_CONTRACTS
}
FILTERED_READ_ARROW_TABLE_KEYS = ("signatures", "papers", "paper_authors")
DECLARED_ARROW_SIDECAR_KEYS = (
    "cluster_seeds",
    "cluster_seed_disallows",
    "altered_cluster_signatures",
    "name_counts_index",
)
UNSUPPORTED_ARROW_NAME_ALIAS_KEYS = frozenset({"name_pairs", "name_tuples"})
DIRECTORY_ARTIFACT_KEYS = frozenset({"name_counts_index"})
NAME_COUNTS_INDEX_MANIFEST_FILES = ("first", "last", "first_last", "last_first_initial")
NAME_COUNTS_INDEX_SCHEMA_VERSION = "name_counts_index_v1"


def _name_counts_index_error(path: Path) -> str | None:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return f"{manifest_path} (missing manifest.json)"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"{manifest_path} (invalid manifest: {exc})"
    schema_version = manifest.get("schema_version")
    if schema_version != NAME_COUNTS_INDEX_SCHEMA_VERSION:
        return (
            f"{manifest_path} (unsupported schema_version {schema_version!r}; "
            f"expected {NAME_COUNTS_INDEX_SCHEMA_VERSION!r})"
        )
    normalization_version = manifest.get("normalization_version", NORMALIZATION_VERSION_LEGACY_COMPAT)
    if normalization_version not in VALID_NORMALIZATION_VERSIONS:
        return (
            f"{manifest_path} (invalid normalization_version {normalization_version!r}; "
            f"expected one of {sorted(VALID_NORMALIZATION_VERSIONS)})"
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        return f"{manifest_path} (missing files mapping)"
    for file_key in NAME_COUNTS_INDEX_MANIFEST_FILES:
        entry = files.get(file_key)
        if not isinstance(entry, Mapping):
            return f"{manifest_path} (missing files.{file_key})"
        path_value = entry.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            return f"{manifest_path} (missing files.{file_key}.path)"
        resolved = Path(path_value)
        if not resolved.is_absolute():
            resolved = path / resolved
        if not resolved.is_file():
            return f"{resolved} (missing files.{file_key}.path target)"
    return None


def read_name_counts_index_normalization_version(path: Any) -> str:
    """Read the normalization_version recorded in a name_counts_index/ manifest.

    An absent field means the artifact predates the normalization contract and is
    treated as "legacy_compat". Invalid tokens raise ValueError.
    """

    manifest_path = Path(os.fspath(path)) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    value = manifest.get("normalization_version", NORMALIZATION_VERSION_LEGACY_COMPAT)
    if value not in VALID_NORMALIZATION_VERSIONS:
        raise ValueError(
            f"{manifest_path} has invalid normalization_version {value!r}; "
            f"expected one of {sorted(VALID_NORMALIZATION_VERSIONS)}"
        )
    return str(value)


def require_name_counts_index_artifact(
    path: Any,
    *,
    context: str,
    producer_hint: str,
) -> str:
    """Require a manifest-backed name-count index directory and return its normalized path."""

    path_text = os.fspath(path) if isinstance(path, os.PathLike) else str(path)
    if not path_text.strip():
        missing_files = {"name_counts_index": "<empty>"}
    elif path_text == ".":
        missing_files = {"name_counts_index": "."}
    else:
        index_path = Path(path_text)
        if not index_path.exists():
            missing_files = {"name_counts_index": str(index_path)}
        elif not index_path.is_dir():
            missing_files = {"name_counts_index": f"{index_path} (expected directory)"}
        else:
            index_error = _name_counts_index_error(index_path)
            missing_files = {"name_counts_index": index_error} if index_error is not None else {}
    if missing_files:
        raise MissingArrowArtifactError(
            context=context,
            required_keys=("name_counts_index",),
            missing_keys=(),
            missing_files=missing_files,
            producer_hint=producer_hint,
        )
    return path_text


def _missing_or_wrong_kind_artifacts(paths: Mapping[str, str], keys: Iterable[str]) -> dict[str, str]:
    missing: dict[str, str] = {}
    for key in keys:
        if key not in paths:
            continue
        path = Path(paths[key])
        if not path.exists():
            missing[key] = str(path)
        elif key in DIRECTORY_ARTIFACT_KEYS and not path.is_dir():
            missing[key] = f"{path} (expected directory)"
        elif key == "name_counts_index":
            index_error = _name_counts_index_error(path)
            if index_error is not None:
                missing[key] = index_error
        elif key not in DIRECTORY_ARTIFACT_KEYS and not path.is_file():
            missing[key] = f"{path} (expected file)"
    return missing


class MissingArrowArtifactError(ValueError):
    """Raised when a strict Arrow production route is missing required artifacts."""

    def __init__(
        self,
        *,
        context: str,
        required_keys: Sequence[str],
        missing_keys: Sequence[str],
        missing_files: Mapping[str, str],
        producer_hint: str,
    ) -> None:
        self.context = str(context)
        self.required_keys = tuple(str(key) for key in required_keys)
        self.missing_keys = tuple(str(key) for key in missing_keys)
        self.missing_files = {str(key): str(value) for key, value in missing_files.items()}
        self.producer_hint = str(producer_hint)
        details = [f"{self.context} is missing required Arrow artifacts"]
        if self.missing_keys:
            details.append(f"missing mapping keys: {', '.join(self.missing_keys)}")
        if self.missing_files:
            formatted_files = "; ".join(f"{key}={value}" for key, value in sorted(self.missing_files.items()))
            details.append(f"missing files: {formatted_files}")
        if self.producer_hint:
            details.append(f"producer hint: {self.producer_hint}")
        super().__init__(". ".join(details))


def _normalize_arrow_path_values(
    paths: Mapping[Any, Any],
    *,
    omit_none: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    normalized: dict[str, str] = {}
    invalid: dict[str, str] = {}
    for key, value in paths.items():
        key_text = str(key)
        if value is None:
            if omit_none:
                continue
            invalid[key_text] = "<None>"
            continue
        path_text = os.fspath(value) if isinstance(value, os.PathLike) else str(value)
        if not path_text.strip():
            invalid[key_text] = "<empty>"
            continue
        if path_text == ".":
            invalid[key_text] = "."
            continue
        normalized[key_text] = path_text
    return normalized, invalid


def normalize_arrow_paths(paths: Mapping[Any, Any], *, omit_none: bool = False) -> dict[str, str]:
    """Return Arrow path mappings with explicit rejection of missing path values."""

    normalized, invalid = _normalize_arrow_path_values(paths, omit_none=omit_none)
    if invalid:
        key, reason = next(iter(invalid.items()))
        if reason == "<None>":
            raise ValueError(f"Arrow path for {key!r} is None")
        if reason == "<empty>":
            raise ValueError(f"Arrow path for {key!r} is empty")
        raise ValueError(f"Arrow path for {key!r} resolves to the current directory")
    return normalized


def required_filtered_read_batch_index_keys(paths: Mapping[str, str]) -> tuple[str, ...]:
    """Return required batch-index keys for filtered Arrow reads over these paths."""

    required = [RAW_PLANNER_ARROW_BATCH_INDEX_KEYS[table_key] for table_key in FILTERED_READ_ARROW_TABLE_KEYS]
    if "specter" in paths:
        required.append(RAW_PLANNER_ARROW_BATCH_INDEX_KEYS["specter"])
    return tuple(required)


def require_filtered_arrow_batch_indexes(
    paths: Mapping[str, str],
    *,
    context: str = "RustFeaturizer.from_arrow_paths",
    producer_hint: str = "generate raw-planner batch indexes with scripts/convert_to_arrow.py",
) -> None:
    """Require batch indexes for production filtered Arrow featurizer builds."""

    required = required_filtered_read_batch_index_keys(paths)
    missing_keys = [key for key in required if key not in paths]
    missing_files = _missing_or_wrong_kind_artifacts(paths, required)
    if missing_keys or missing_files:
        raise MissingArrowArtifactError(
            context=context,
            required_keys=required,
            missing_keys=missing_keys,
            missing_files=missing_files,
            producer_hint=producer_hint,
        )


def require_arrow_artifacts(
    arrow_paths: Mapping[str, Any],
    *,
    required_keys: Sequence[str],
    context: str,
    producer_hint: str,
) -> dict[str, str]:
    """Require specific Arrow path keys and existing files, returning normalized paths."""

    required = tuple(str(key) for key in required_keys)
    missing_keys = [key for key in required if key not in arrow_paths]
    normalized, invalid_paths = _normalize_arrow_path_values(arrow_paths)
    missing_files = _missing_or_wrong_kind_artifacts(normalized, required)
    missing_files.update({key: invalid_paths[key] for key in required if key in invalid_paths})
    if missing_keys or missing_files:
        raise MissingArrowArtifactError(
            context=context,
            required_keys=required,
            missing_keys=missing_keys,
            missing_files=missing_files,
            producer_hint=producer_hint,
        )
    return normalized


def validate_arrow_prediction_artifacts(
    arrow_paths: Mapping[str, Any],
    *,
    require_specter: bool,
    require_name_counts_index: bool,
    require_cluster_seeds: bool = False,
    require_batch_indexes: bool = False,
    expected_normalization_version: str | None = None,
    context: str = "Arrow prediction",
    producer_hint: str = (
        "generate a complete Arrow bundle with scripts/convert_to_arrow.py or use the published "
        "s2and-release-arrow bundle"
    ),
) -> dict[str, str]:
    """Validate strict production Arrow prediction artifacts and return normalized paths."""

    required = {"signatures", "papers", "paper_authors"}
    if require_specter:
        required.add("specter")
    if require_name_counts_index:
        required.add("name_counts_index")
    if require_cluster_seeds:
        required.add("cluster_seeds")

    missing_keys = sorted(key for key in required if key not in arrow_paths)
    normalized, invalid_paths = _normalize_arrow_path_values(arrow_paths)
    for key in UNSUPPORTED_ARROW_NAME_ALIAS_KEYS.intersection(normalized):
        invalid_paths[key] = "name aliases must be supplied via the name_tuples argument, not Arrow path bundles"

    if require_specter and "specter" not in normalized and "specter2" in normalized:
        normalized["specter"] = normalized["specter2"]
        invalid_paths.pop("specter", None)
        if "specter" in missing_keys:
            missing_keys.remove("specter")
    if require_specter and "specter" in normalized:
        invalid_paths.pop("specter2", None)
    if require_specter and "specter_batch_index" not in normalized and "specter2_batch_index" in normalized:
        normalized["specter_batch_index"] = normalized["specter2_batch_index"]
        invalid_paths.pop("specter_batch_index", None)
    if require_specter and "specter_batch_index" in normalized:
        invalid_paths.pop("specter2_batch_index", None)
    if not require_specter:
        normalized.pop("specter", None)
        normalized.pop("specter_batch_index", None)
        normalized.pop("specter2", None)
        normalized.pop("specter2_batch_index", None)
        invalid_paths.pop("specter", None)
        invalid_paths.pop("specter_batch_index", None)
        invalid_paths.pop("specter2", None)
        invalid_paths.pop("specter2_batch_index", None)

    if require_batch_indexes:
        for key in required_filtered_read_batch_index_keys(normalized):
            required.add(key)
            if key not in normalized:
                missing_keys.append(key)

    required_or_declared_keys = {
        key
        for key in normalized
        if key in required or key.endswith("_batch_index") or key in DECLARED_ARROW_SIDECAR_KEYS
    }
    missing_files = _missing_or_wrong_kind_artifacts(normalized, required_or_declared_keys)
    missing_files.update(invalid_paths)
    if missing_keys or missing_files:
        raise MissingArrowArtifactError(
            context=context,
            required_keys=sorted(required),
            missing_keys=sorted(set(missing_keys)),
            missing_files=missing_files,
            producer_hint=producer_hint,
        )
    if expected_normalization_version is not None and "name_counts_index" in normalized:
        artifact_version = read_name_counts_index_normalization_version(normalized["name_counts_index"])
        if artifact_version != expected_normalization_version:
            raise MissingArrowArtifactError(
                context=context,
                required_keys=sorted(required),
                missing_keys=(),
                missing_files={
                    "name_counts_index": (
                        f"normalization_version mismatch: artifact is {artifact_version!r} but the model "
                        f"feature contract requires {expected_normalization_version!r}; regenerate the "
                        "artifact bundle and model as one release unit "
                        "(docs/normalization_migration_blocked.md)"
                    )
                },
                producer_hint=producer_hint,
            )
    return normalized
