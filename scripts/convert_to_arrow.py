"""Convert S2AND runtime inputs into direct-Rust Arrow artifacts.

The runtime bundle writer emits bounded Arrow IPC file-format tables plus the
current S2AND raw-planner batch-index sidecars (S2ABI002).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from s2and.arrow_inputs import require_name_counts_index_artifact  # noqa: E402

logger = logging.getLogger(__name__)


BENCHMARK_DATASETS = ("aminer", "arnetminer", "inspire", "kisti", "medline", "pubmed", "qian", "zbmath")
ROOT_MANIFEST_SCHEMA = "inference_arrow_bundle_v1"
ARROW_SCHEMA_CONTRACT_PATH = _PROJECT_ROOT / "s2and" / "arrow_schema_contract.json"


@dataclass(frozen=True)
class RuntimeDatasetSources:
    """Source files for one table-shaped runtime dataset."""

    dataset: str
    source_dir: Path
    signatures_path: Path
    papers_path: Path
    clusters_path: Path | None = None
    specter_path: Path | None = None
    specter2_path: Path | None = None


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as infile:
        return json.load(infile)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _replace_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            encoding="utf-8",
            delete=False,
        ) as tmp_file:
            tmp_file.write(encoded)
            tmp_path = Path(tmp_file.name)
        tmp_path.replace(path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _resolve_manifest_path(path_value: Any, base_dir: Path | None) -> Path:
    path = Path(str(path_value))
    if path.is_absolute() or base_dir is None:
        return path
    return base_dir / path


def _manifest_relative_path(path_value: Any, manifest_dir: Path) -> str:
    path = Path(str(path_value))
    try:
        return os.path.relpath(str(path.resolve()), str(manifest_dir.resolve()))
    except ValueError:
        return str(path)


def _portable_manifest_paths(paths: Mapping[str, Any], manifest_dir: Path) -> dict[str, str]:
    return {str(key): _manifest_relative_path(value, manifest_dir) for key, value in paths.items()}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(args: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=_PROJECT_ROOT,
            check=True,
            capture_output=True,
            encoding="utf-8",
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip()


def _git_commit_metadata() -> dict[str, Any]:
    status = _git_output(["status", "--porcelain"])
    return {
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "git_dirty": None if status is None else bool(status),
    }


class _RootManifestLock:
    """Small same-directory lock for root manifest read-modify-write."""

    def __init__(self, path: Path, *, attempts: int = 50, sleep_seconds: float = 0.1) -> None:
        self.path = path
        self.attempts = attempts
        self.sleep_seconds = sleep_seconds
        self._fd: int | None = None
        self._payload: str | None = None

    def _try_create_lock_file(self) -> bool:
        payload = f"{os.getpid()}\n"
        fd: int | None = None
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, payload.encode("ascii"))
        except FileExistsError:
            return False
        except Exception:
            self.path.unlink(missing_ok=True)
            raise
        finally:
            if fd is not None:
                os.close(fd)
        self._payload = payload
        return True

    def __enter__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, self.attempts + 1):
            if self._try_create_lock_file():
                self._fd = None
                return
            if _remove_dead_pid_lock(self.path) and self._try_create_lock_file():
                self._fd = None
                return
            if attempt == self.attempts:
                lock_pid = _lock_pid(self.path)
                pid_context = f" held by pid {lock_pid}" if lock_pid is not None else ""
                raise TimeoutError(
                    f"timed out waiting for root manifest lock {self.path}{pid_context} "
                    f"after {self.attempts} attempts; remove the lock file if no converter is running"
                )
            time.sleep(self.sleep_seconds)

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self._payload is None:
            return
        try:
            current_payload = self.path.read_text(encoding="ascii")
        except FileNotFoundError:
            return
        except OSError:
            return
        if current_payload == self._payload:
            self.path.unlink(missing_ok=True)
        self._payload = None


def _lock_pid(path: Path) -> int | None:
    try:
        raw_pid = path.read_text(encoding="ascii").splitlines()[0].strip()
    except OSError:
        return None
    except IndexError:
        return None
    if not raw_pid:
        return None
    try:
        pid = int(raw_pid)
    except ValueError:
        return None
    return pid if pid > 0 else None


def _pid_is_running(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, int(pid))
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _remove_dead_pid_lock(path: Path) -> bool:
    try:
        raw_pid = path.read_text(encoding="ascii").splitlines()[0].strip()
    except OSError:
        return False
    except IndexError:
        raw_pid = ""
    if raw_pid:
        try:
            pid = int(raw_pid)
        except ValueError:
            pid = None
    else:
        pid = None
    if pid is not None and pid > 0 and _pid_is_running(pid):
        return False
    try:
        path.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def _root_manifest_entries(root_manifest_path: Path, dataset_name: str) -> list[dict[str, Any]]:
    if not root_manifest_path.exists():
        return []
    try:
        root_manifest = _load_json(root_manifest_path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"existing root manifest is invalid JSON: {root_manifest_path}") from exc
    if not isinstance(root_manifest, Mapping):
        raise TypeError(f"existing root manifest must contain an object: {root_manifest_path}")

    entries = _root_manifest_entries_from_manifest(root_manifest, root_manifest_path)
    return [entry for entry in entries if entry["dataset"] != dataset_name]


def _validate_existing_root_manifest(root_manifest_path: Path) -> None:
    if not root_manifest_path.exists():
        return
    try:
        root_manifest = _load_json(root_manifest_path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"existing root manifest is invalid JSON: {root_manifest_path}") from exc
    if not isinstance(root_manifest, Mapping):
        raise TypeError(f"existing root manifest must contain an object: {root_manifest_path}")
    _root_manifest_entries_from_manifest(root_manifest, root_manifest_path)


def _root_manifest_entries_from_manifest(
    root_manifest: Mapping[str, Any],
    root_manifest_path: Path,
) -> list[dict[str, Any]]:
    if root_manifest.get("schema") == ROOT_MANIFEST_SCHEMA:
        raw_entries = root_manifest.get("dataset_manifests")
        if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, str | bytes):
            raise ValueError(f"existing root manifest dataset_manifests must be a list: {root_manifest_path}")
        entries = _validated_root_manifest_entries(raw_entries, root_manifest_path)
    else:
        raise ValueError(
            "existing root manifest has unsupported schema "
            f"{root_manifest.get('schema')!r}; expected {ROOT_MANIFEST_SCHEMA!r}: {root_manifest_path}"
        )
    return entries


def _root_manifest_replay_bundle_entries_from_manifest(
    root_manifest: Mapping[str, Any],
    root_manifest_path: Path,
) -> list[dict[str, Any]]:
    raw_entries = root_manifest.get("replay_bundles", [])
    if raw_entries is None:
        return []
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, str | bytes):
        raise ValueError(f"existing root manifest replay_bundles must be a list: {root_manifest_path}")
    entries: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"existing root manifest replay_bundles[{index}] must be an object: {root_manifest_path}")
        manifest_path = raw_entry.get("manifest_path")
        if manifest_path is None:
            raise ValueError(
                f"existing root manifest replay_bundles[{index}] is missing manifest_path: {root_manifest_path}"
            )
        manifest_path_text = str(manifest_path)
        if not manifest_path_text:
            raise ValueError(
                f"existing root manifest replay_bundles[{index}] has empty manifest_path: {root_manifest_path}"
            )
        entry = dict(raw_entry)
        manifest_path_parts = PurePosixPath(manifest_path_text.replace("\\", "/")).parts
        entry["manifest_path"] = manifest_path_text
        entry["bundle"] = str(raw_entry.get("bundle") or manifest_path_parts[0])
        entries.append(entry)
    return entries


def _validated_root_manifest_entries(raw_entries: Any, root_manifest_path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(
                f"existing root manifest dataset_manifests[{index}] must be an object: {root_manifest_path}"
            )
        if raw_entry.get("dataset") is None:
            raise ValueError(
                f"existing root manifest dataset_manifests[{index}] is missing dataset: {root_manifest_path}"
            )
        if raw_entry.get("manifest_path") is None:
            raise ValueError(
                f"existing root manifest dataset_manifests[{index}] is missing manifest_path: {root_manifest_path}"
            )
        entry = dict(raw_entry)
        entry["dataset"] = str(raw_entry["dataset"])
        entry["dataset_dir"] = str(raw_entry.get("dataset_dir", ""))
        entry["manifest_path"] = str(raw_entry["manifest_path"])
        entries.append(entry)
    return entries


def _entry_manifest_path(output_root: Path, entry: Mapping[str, Any]) -> Path:
    return _resolve_manifest_path(entry["manifest_path"], output_root)


def _int_manifest_value(manifest: Mapping[str, Any], key: str) -> int:
    value = manifest.get(key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _optional_int_value(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_int_value(*values: Any) -> int:
    for value in values:
        parsed = _optional_int_value(value)
        if parsed is not None:
            return parsed
    return 0


def _specter_report_row_count(manifest: Mapping[str, Any]) -> int | None:
    reports = manifest.get("specter")
    if not isinstance(reports, Mapping):
        return None
    preferred_keys = ("specter2", "specter")
    for key in preferred_keys:
        report = reports.get(key)
        if isinstance(report, Mapping):
            row_count = _optional_int_value(report.get("row_count"))
            if row_count is not None:
                return row_count
    for report in reports.values():
        if isinstance(report, Mapping):
            row_count = _optional_int_value(report.get("row_count"))
            if row_count is not None:
                return row_count
    return None


def _validation_requirements_from_manifest(manifest: Mapping[str, Any]) -> dict[str, bool]:
    paths = manifest.get("paths", {})
    path_map = paths if isinstance(paths, Mapping) else {}
    return {
        "require_embeddings": bool(path_map.get("specter") or path_map.get("specter2")),
        "require_name_counts_index": bool(path_map.get("name_counts_index")),
    }


def _dataset_manifest_audit(manifest: Mapping[str, Any]) -> dict[str, Any]:
    paths = manifest.get("paths", {})
    path_keys = set(paths) if isinstance(paths, Mapping) else set()
    validation = manifest.get("validation", {})
    validation_map = validation if isinstance(validation, Mapping) else {}
    physical_layout = manifest.get("physical_layout", {})
    physical_tables = physical_layout.get("tables", {}) if isinstance(physical_layout, Mapping) else {}
    batch_index_count = sum(1 for key in path_keys if str(key).endswith("_batch_index"))
    if isinstance(physical_tables, Mapping):
        batch_index_count = max(
            batch_index_count,
            sum(
                1
                for layout in physical_tables.values()
                if isinstance(layout, Mapping) and layout.get("batch_index_present")
            ),
        )
    sidecar_keys = sorted(
        key
        for key in path_keys
        if str(key).endswith("_batch_index")
        or str(key) in {"cluster_seeds", "cluster_seed_disallows", "altered_cluster_signatures", "name_counts_index"}
    )
    return {
        "conversion_kind": manifest.get("conversion_kind"),
        "source_id": manifest.get("source_path") or manifest.get("source_dir"),
        "signature_count": _int_manifest_value(manifest, "signature_count"),
        "paper_count": _int_manifest_value(manifest, "paper_count"),
        "embedding_row_count": _first_int_value(
            validation_map.get("specter_count"),
            manifest.get("paper_embedding_count"),
            _specter_report_row_count(manifest),
        ),
        "cluster_seed_count": _first_int_value(
            validation_map.get("cluster_seed_count"),
            manifest.get("cluster_seeds_require_count"),
        ),
        "cluster_seed_disallow_count": _first_int_value(
            validation_map.get("cluster_seed_disallow_count"),
            manifest.get("cluster_seeds_disallow_count"),
        ),
        "altered_cluster_signature_count": _first_int_value(
            validation_map.get("altered_cluster_signature_count"),
            manifest.get("altered_cluster_signature_count"),
        ),
        "missing_embedding_count": _first_int_value(validation_map.get("missing_specter_paper_count")),
        "batch_index_count": int(batch_index_count),
        "sidecar_keys": sidecar_keys,
        "validation_present": bool(validation_map),
    }


def _enrich_root_manifest_entry(output_root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    enriched = dict(entry)
    manifest_path = _entry_manifest_path(output_root, entry)
    if not manifest_path.exists():
        enriched["manifest_exists"] = False
        return enriched
    enriched["manifest_exists"] = True
    manifest_stat = manifest_path.stat()
    enriched["manifest_size_bytes"] = int(manifest_stat.st_size)
    enriched["manifest_sha256"] = _file_sha256(manifest_path)
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise TypeError(f"dataset manifest must contain an object: {manifest_path}")
    enriched["audit"] = _dataset_manifest_audit(manifest)
    enriched["validation_requirements"] = _validation_requirements_from_manifest(manifest)
    return enriched


def _enrich_replay_bundle_entry(output_root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    enriched = dict(entry)
    manifest_path = _entry_manifest_path(output_root, entry)
    if not manifest_path.exists():
        enriched["manifest_exists"] = False
        return enriched
    enriched["manifest_exists"] = True
    manifest_stat = manifest_path.stat()
    enriched["manifest_size_bytes"] = int(manifest_stat.st_size)
    enriched["manifest_sha256"] = _file_sha256(manifest_path)
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise TypeError(f"replay bundle manifest must contain an object: {manifest_path}")
    nested_entries = _root_manifest_entries_from_manifest(manifest, manifest_path)
    bundle_root = manifest_path.parent
    enriched_nested_entries = [_enrich_root_manifest_entry(bundle_root, nested) for nested in nested_entries]
    enriched["datasets"] = [entry["dataset"] for entry in enriched_nested_entries]
    enriched["dataset_manifests"] = enriched_nested_entries
    enriched["audit"] = {
        "schema": manifest.get("schema"),
        **_root_manifest_audit(enriched_nested_entries),
    }
    return enriched


def _root_manifest_audit(dataset_manifests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    datasets_with_missing_manifests = [
        str(entry["dataset"]) for entry in dataset_manifests if not bool(entry.get("manifest_exists", False))
    ]
    audits: list[Mapping[str, Any]] = []
    for entry in dataset_manifests:
        audit = entry.get("audit")
        if isinstance(audit, Mapping):
            audits.append(audit)
    return {
        "dataset_count": len(dataset_manifests),
        "datasets_with_missing_manifests": datasets_with_missing_manifests,
        "total_signature_count": sum(int(audit.get("signature_count", 0) or 0) for audit in audits),
        "total_paper_count": sum(int(audit.get("paper_count", 0) or 0) for audit in audits),
        "total_embedding_row_count": sum(int(audit.get("embedding_row_count", 0) or 0) for audit in audits),
        "total_missing_embedding_count": sum(int(audit.get("missing_embedding_count", 0) or 0) for audit in audits),
        "total_batch_index_count": sum(int(audit.get("batch_index_count", 0) or 0) for audit in audits),
    }


def _root_manifest_validation_commands(
    output_root: Path,
    dataset_manifests: Sequence[Mapping[str, Any]],
    *,
    dataset_dir_prefix: str = "",
) -> list[str]:
    commands = []
    for entry in dataset_manifests:
        if not bool(entry.get("manifest_exists", False)):
            continue
        dataset_dir = str(entry.get("dataset_dir") or "").replace("\\", "/")
        if not dataset_dir:
            continue
        if dataset_dir_prefix:
            dataset_dir = f"{dataset_dir_prefix.rstrip('/')}/{dataset_dir}"
        dataset_dir = _manifest_relative_path(output_root / dataset_dir, _PROJECT_ROOT).replace("\\", "/")
        command_parts = ["uv run python scripts/convert_to_arrow.py validate", f"--dataset-dir {dataset_dir}"]
        requirements = entry.get("validation_requirements")
        if isinstance(requirements, Mapping):
            if requirements.get("require_embeddings"):
                command_parts.append("--require-embeddings")
            if requirements.get("require_name_counts_index"):
                command_parts.append("--require-name-counts-index")
        commands.append(" ".join(command_parts))
    return commands


def _replay_bundle_validation_commands(output_root: Path, replay_bundles: Sequence[Mapping[str, Any]]) -> list[str]:
    commands: list[str] = []
    for bundle in replay_bundles:
        if not bool(bundle.get("manifest_exists", False)):
            continue
        manifest_path = str(bundle.get("manifest_path") or "").replace("\\", "/")
        if not manifest_path:
            continue
        bundle_prefix = str(PurePosixPath(manifest_path).parent)
        nested_entries = bundle.get("dataset_manifests", [])
        if isinstance(nested_entries, Sequence) and not isinstance(nested_entries, str | bytes):
            commands.extend(
                _root_manifest_validation_commands(output_root, nested_entries, dataset_dir_prefix=bundle_prefix)
            )
    return commands


def _write_root_manifest(
    output_root: Path,
    *,
    dataset_manifests: Sequence[Mapping[str, Any]],
    replay_bundles: Sequence[Mapping[str, Any]] = (),
    output_root_label: str | None = None,
) -> dict[str, Any]:
    enriched_dataset_manifests = [_enrich_root_manifest_entry(output_root, entry) for entry in dataset_manifests]
    enriched_replay_bundles = [_enrich_replay_bundle_entry(output_root, entry) for entry in replay_bundles]
    validation_commands = _root_manifest_validation_commands(output_root, enriched_dataset_manifests)
    validation_commands.extend(_replay_bundle_validation_commands(output_root, enriched_replay_bundles))
    payload: dict[str, Any] = {
        "schema": ROOT_MANIFEST_SCHEMA,
        "output_root": str(output_root_label or output_root),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "generator": {
            "script": "scripts/convert_to_arrow.py",
            **_git_commit_metadata(),
        },
        "datasets": [entry["dataset"] for entry in enriched_dataset_manifests],
        "dataset_manifests": enriched_dataset_manifests,
        "audit": _root_manifest_audit(enriched_dataset_manifests),
        "validation_command_cwd": str(_PROJECT_ROOT),
        "validation_commands": validation_commands,
    }
    if enriched_replay_bundles:
        payload["replay_bundles"] = enriched_replay_bundles
        payload["replay_audit"] = {
            "bundle_count": len(enriched_replay_bundles),
            "bundles_with_missing_manifests": [
                str(entry.get("bundle")) for entry in enriched_replay_bundles if not entry.get("manifest_exists")
            ],
            "total_dataset_count": sum(
                int(entry.get("audit", {}).get("dataset_count", 0))
                for entry in enriched_replay_bundles
                if isinstance(entry.get("audit"), Mapping)
            ),
        }
    _replace_json(output_root / "manifest.json", payload)
    return payload


def _upsert_root_manifest(output_root: Path, *, dataset_name: str, dataset_dir: Path) -> None:
    root_manifest_path = output_root / "manifest.json"
    lock_path = root_manifest_path.with_suffix(root_manifest_path.suffix + ".lock")
    with _RootManifestLock(lock_path):
        existing_root_manifest: Mapping[str, Any] = {}
        if root_manifest_path.exists():
            loaded_root_manifest = _load_json(root_manifest_path)
            if not isinstance(loaded_root_manifest, Mapping):
                raise TypeError(f"existing root manifest must contain an object: {root_manifest_path}")
            existing_root_manifest = loaded_root_manifest
        dataset_manifests = _root_manifest_entries(root_manifest_path, dataset_name)
        dataset_manifests.append(
            {
                "dataset": dataset_name,
                "dataset_dir": _manifest_relative_path(dataset_dir, output_root),
                "manifest_path": _manifest_relative_path(dataset_dir / "manifest.json", output_root),
            }
        )
        dataset_manifests.sort(key=lambda entry: entry["dataset"])
        replay_bundles = _root_manifest_replay_bundle_entries_from_manifest(existing_root_manifest, root_manifest_path)
        _write_root_manifest(
            output_root,
            dataset_manifests=dataset_manifests,
            replay_bundles=replay_bundles,
            output_root_label=str(existing_root_manifest.get("output_root") or output_root),
        )


def _mapping_by_id(rows: Any, *, id_key: str, label: str) -> dict[str, Mapping[str, Any]]:
    if isinstance(rows, Mapping):
        return {str(key): value for key, value in rows.items()}
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
        raise TypeError(f"{label} must be a JSON object or list")
    mapped: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError(f"{label} rows must be objects")
        row_id = row.get(id_key)
        if row_id is None:
            raise ValueError(f"{label} row is missing {id_key!r}")
        row_key = str(row_id)
        if row_key in mapped:
            raise ValueError(f"{label} contains duplicate {id_key}: {row_key!r}")
        mapped[row_key] = row
    return mapped


def _altered_values(payload: Mapping[str, Any]) -> list[str]:
    values = payload.get("altered_cluster_signatures") or []
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError("altered_cluster_signatures must be a list when present")
    return [str(value) for value in values]


def _require_groups_from_service_payload(value: Any) -> list[list[str]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        values = list(value.values())
        if all(
            not isinstance(item, Mapping) and (not isinstance(item, Sequence) or isinstance(item, str | bytes))
            for item in values
        ):
            groups_by_component: dict[str, list[str]] = {}
            for signature_id, component_key in value.items():
                groups_by_component.setdefault(str(component_key), []).append(str(signature_id))
            return list(groups_by_component.values())
        if all(isinstance(item, Sequence) and not isinstance(item, str | bytes) for item in values):
            return [[str(signature_id) for signature_id in members] for members in values]
        raise TypeError("cluster_seeds.require must be either signature->cluster or cluster->signature-list")
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        groups: list[list[str]] = []
        for item in value:
            if not isinstance(item, Sequence) or isinstance(item, str | bytes):
                raise TypeError("cluster_seeds.require list entries must be signature-id lists")
            groups.append([str(signature_id) for signature_id in item])
        return groups
    raise TypeError("cluster_seeds.require must be an object or list")


def _disallow_pairs_from_service_payload(value: Any) -> list[tuple[str, str]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        pairs: list[tuple[str, str]] = []
        for left, right_values in value.items():
            if isinstance(right_values, Mapping):
                iterable_values = right_values.keys()
            elif isinstance(right_values, Sequence) and not isinstance(right_values, str | bytes):
                iterable_values = right_values
            else:
                raise TypeError("cluster_seeds.disallow object values must be signature-id lists or objects")
            pairs.extend((str(left), str(right)) for right in iterable_values)
        return pairs
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        pairs = []
        for item in value:
            if not isinstance(item, Sequence) or isinstance(item, str | bytes) or len(item) != 2:
                raise TypeError("cluster_seeds.disallow list entries must be pairs")
            pairs.append((str(item[0]), str(item[1])))
        return pairs
    raise TypeError("cluster_seeds.disallow must be an object or list")


def _cluster_seeds_payload(payload: Mapping[str, Any]) -> Any:
    cluster_seeds = payload.get("cluster_seeds")
    if not isinstance(cluster_seeds, Mapping) or not ({"require", "disallow"} & set(cluster_seeds)):
        return cluster_seeds
    unexpected_keys = sorted(set(cluster_seeds).difference({"require", "disallow"}))
    if unexpected_keys:
        raise ValueError(f"service-shaped cluster_seeds contains unsupported keys: {unexpected_keys}")

    legacy_cluster_seeds: dict[str, dict[str, str]] = {}

    def add_pair(left: str, right: str, constraint: str) -> None:
        existing = legacy_cluster_seeds.setdefault(left, {}).get(right)
        if existing is not None and existing != constraint:
            raise ValueError(
                f"cluster_seeds contains conflicting constraints for pair {(left, right)!r}: "
                f"{existing!r} and {constraint!r}"
            )
        legacy_cluster_seeds[left][right] = constraint

    for group in _require_groups_from_service_payload(cluster_seeds.get("require")):
        if not group:
            raise ValueError("cluster_seeds.require cannot contain an empty group")
        root = group[0]
        if len(group) == 1:
            add_pair(root, root, "require")
        else:
            for signature_id in group[1:]:
                add_pair(root, signature_id, "require")
    for left, right in _disallow_pairs_from_service_payload(cluster_seeds.get("disallow")):
        add_pair(left, right, "disallow")
    return legacy_cluster_seeds


def _specter_mapping(payload: Any) -> dict[str, np.ndarray]:
    if isinstance(payload, dict):
        return {str(key): np.asarray(value, dtype=np.float32) for key, value in payload.items()}
    if isinstance(payload, tuple) and len(payload) == 2:
        matrix, keys = payload
        matrix_array = np.asarray(matrix, dtype=np.float32)
        return {str(key): np.asarray(matrix_array[index], dtype=np.float32) for index, key in enumerate(keys)}
    raise TypeError(f"Unsupported SPECTER payload type: {type(payload).__name__}")


def _write_specter_arrow(
    *,
    source_path: Path,
    output_path: Path,
    needed_paper_ids: set[str],
    overwrite: bool,
) -> dict[str, Any]:
    import pyarrow as pa

    if output_path.exists() and not overwrite:
        return {"path": str(output_path), "reused": True}

    with source_path.open("rb") as infile:
        specter_by_paper_id = _specter_mapping(pickle.load(infile))
    selected_items: list[tuple[str, np.ndarray]] = []
    empty_vector_count = 0
    for paper_id, vector in specter_by_paper_id.items():
        if str(paper_id) not in needed_paper_ids:
            continue
        if vector.size == 0:
            empty_vector_count += 1
            continue
        selected_items.append((paper_id, vector))
    if empty_vector_count:
        logger.warning(
            "Dropped %d SPECTER embeddings with zero-size vectors from %s",
            empty_vector_count,
            source_path,
        )
    selected_items.sort(key=lambda item: item[0])
    if not selected_items:
        raise ValueError(f"No SPECTER embeddings from {source_path} matched the dataset papers")

    dimension = int(selected_items[0][1].shape[0])
    for paper_id, vector in selected_items:
        if int(vector.shape[0]) != dimension:
            raise ValueError(
                f"SPECTER dimension mismatch in {source_path}: paper_id={paper_id!r} "
                f"expected={dimension} got={vector.shape[0]}"
            )

    matrix = np.vstack([vector for _paper_id, vector in selected_items]).astype(np.float32, copy=False)
    flat = pa.array(np.ravel(matrix), type=pa.float32())
    table = pa.table(
        {
            "paper_id": pa.array([paper_id for paper_id, _vector in selected_items], type=pa.string()),
            "embedding": pa.FixedSizeListArray.from_arrays(flat, dimension),
        }
    )
    from s2and.incremental_linking.feature_block import RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS, write_arrow_ipc_table

    write_arrow_ipc_table(
        table,
        output_path,
        max_record_batch_rows=RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS["specter"],
    )
    return {
        "path": str(output_path),
        "reused": False,
        "row_count": int(table.num_rows),
        "dimension": dimension,
        "source_path": str(source_path),
        "dropped_empty_embedding_count": empty_vector_count,
    }


def _add_extra_specter_index_and_layout(
    *,
    paths: dict[str, str],
    raw_planner_index_metrics: dict[str, Any],
    physical_layout: dict[str, Any],
    table_key: str,
    output_dir: Path,
    overwrite: bool,
) -> None:
    from s2and.incremental_linking.feature_block import (
        RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
        arrow_ipc_physical_layout,
        write_arrow_batch_lookup_index,
    )

    arrow_path = paths.get(table_key)
    if arrow_path is None:
        return
    index_key = f"{table_key}_batch_index"
    index_path, index_metrics = write_arrow_batch_lookup_index(
        arrow_path,
        output_dir / f"{Path(arrow_path).stem}.specter_batch_index.bin",
        key_column="paper_id",
        table_name="specter",
        max_record_batch_rows=RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS["specter"],
        overwrite=overwrite,
    )
    paths[index_key] = index_path
    raw_planner_index_metrics[index_key] = index_metrics
    physical_layout["tables"][table_key] = {
        "key": "paper_id",
        "max_record_batch_rows": RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS["specter"],
        "batch_index_path_key": index_key,
        "batch_index_present": True,
        **arrow_ipc_physical_layout(arrow_path),
    }


def _source_file(source_dir: Path, dataset: str, preferred_name: str, fallback_name: str | None = None) -> Path:
    candidates = [source_dir / preferred_name]
    if fallback_name is not None:
        candidates.append(source_dir / fallback_name)
    for path in candidates:
        if path.exists():
            return path
    formatted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing source file. Tried: {formatted}")


def _optional_source_file(
    source_dir: Path, dataset: str, preferred_name: str, fallback_name: str | None = None
) -> Path | None:
    candidates = [source_dir / preferred_name]
    if fallback_name is not None:
        candidates.append(source_dir / fallback_name)
    for path in candidates:
        if path.exists():
            return path
    return None


def benchmark_dataset_sources(source_root: Path, dataset: str) -> RuntimeDatasetSources:
    source_dir = source_root / dataset
    return RuntimeDatasetSources(
        dataset=dataset,
        source_dir=source_dir,
        signatures_path=_source_file(source_dir, dataset, f"{dataset}_signatures.json", "signatures.json"),
        papers_path=_source_file(source_dir, dataset, f"{dataset}_papers.json", "papers.json"),
        clusters_path=_optional_source_file(source_dir, dataset, f"{dataset}_clusters.json", "clusters.json"),
        specter_path=_optional_source_file(source_dir, dataset, f"{dataset}_specter.pickle", "specter.pickle"),
        specter2_path=_optional_source_file(source_dir, dataset, f"{dataset}_specter2.pkl", "specter2.pkl"),
    )


def linker_replay_dataset_sources(raw_root: Path, embeddings_root: Path, dataset: str) -> RuntimeDatasetSources:
    raw_dir = raw_root / dataset
    embeddings_dir = embeddings_root / dataset
    return RuntimeDatasetSources(
        dataset=dataset,
        source_dir=raw_dir,
        signatures_path=_source_file(raw_dir, dataset, "signatures.json"),
        papers_path=_source_file(raw_dir, dataset, "papers.json"),
        specter2_path=_source_file(embeddings_dir, dataset, "specter2.pkl"),
    )


def discover_benchmark_datasets(source_root: Path) -> list[str]:
    discovered: list[str] = []
    for dataset in BENCHMARK_DATASETS:
        source_dir = source_root / dataset
        if source_dir.exists() and (source_dir / f"{dataset}_signatures.json").exists():
            discovered.append(dataset)
    if discovered:
        return discovered
    return [
        child.name for child in sorted(source_root.iterdir()) if child.is_dir() and any(child.glob("*_signatures.json"))
    ]


def discover_linker_replay_datasets(raw_root: Path, embeddings_root: Path) -> list[str]:
    return [
        child.name
        for child in sorted(raw_root.iterdir())
        if child.is_dir()
        and (child / "signatures.json").exists()
        and (child / "papers.json").exists()
        and (embeddings_root / child.name / "specter2.pkl").exists()
    ]


def convert_service_json_to_arrow(
    *,
    input_json: Path,
    output_root: Path,
    dataset_name: str,
    name_counts_index_root: Path | None = None,
    n_jobs: int,
    overwrite: bool,
    skip_name_counts_index: bool,
    overwrite_name_counts_index: bool = False,
    copy_source_json: bool = False,
    validate: bool = True,
) -> dict[str, Any]:
    """Write one service-shaped inference request as an Arrow dataset."""

    from s2and.data import ANDData
    from s2and.incremental_linking.feature_block import (
        FEATURE_BLOCK_ARROW_MANIFEST_SCHEMA_VERSION,
        raw_planner_arrow_physical_layout,
        write_arrow_ipc_table,
        write_name_counts_index,
        write_raw_arrow_batch_lookup_indexes,
    )
    from scripts.arrow_conversion_helpers import write_feature_block_arrow_from_anddata

    output_dir = output_root / dataset_name
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory already contains files for dataset {dataset_name!r}: {output_dir}. "
            "Use --overwrite to regenerate it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_existing_root_manifest(output_root / "manifest.json")

    start = time.perf_counter()
    payload = _load_json(input_json)
    if not isinstance(payload, Mapping):
        raise TypeError("input JSON must contain an object")
    load_seconds = time.perf_counter() - start

    signatures = _mapping_by_id(payload.get("signatures"), id_key="signature_id", label="signatures")
    papers = _mapping_by_id(payload.get("papers"), id_key="paper_id", label="papers")
    altered = _altered_values(payload)
    specter_embeddings = payload.get("paper_embeddings")
    if specter_embeddings is None:
        specter_embeddings = payload.get("specter_embeddings")

    start = time.perf_counter()
    dataset = ANDData(
        signatures=signatures,
        papers=papers,
        name=dataset_name,
        mode="inference",
        clusters=None,
        specter_embeddings=specter_embeddings,
        cluster_seeds=_cluster_seeds_payload(payload),
        altered_cluster_signatures=altered,
        block_type="s2",
        train_pairs=None,
        val_pairs=None,
        test_pairs=None,
        train_pairs_size=1000,
        val_pairs_size=1000,
        test_pairs_size=1000,
        n_jobs=n_jobs,
        load_name_counts=not skip_name_counts_index,
        preprocess=True,
        random_seed=42,
        name_tuples="filtered",
        use_orcid_id=True,
        compute_reference_features=False,
    )
    anddata_seconds = time.perf_counter() - start

    start = time.perf_counter()
    paths = write_feature_block_arrow_from_anddata(
        dataset,
        output_dir,
        signature_ids=list(dataset.signatures),
        include_specter=specter_embeddings is not None,
        include_empty_cluster_seeds=True,
        overwrite=overwrite,
    )
    write_arrow_seconds = time.perf_counter() - start

    import pyarrow as pa

    altered_arrow_path = output_dir / "altered_cluster_signatures.arrow"
    if overwrite or not altered_arrow_path.exists():
        table = pa.table({"signature_id": pa.array(altered, type=pa.string())})
        write_arrow_ipc_table(table, altered_arrow_path)
    paths["altered_cluster_signatures"] = str(altered_arrow_path)

    if copy_source_json:
        source_paths = {
            "signatures_json": output_dir / "signatures.json",
            "papers_json": output_dir / "papers.json",
            "cluster_seeds_json": output_dir / "cluster_seeds.json",
        }
        if overwrite or not source_paths["signatures_json"].exists():
            _write_json(source_paths["signatures_json"], signatures)
        if overwrite or not source_paths["papers_json"].exists():
            _write_json(source_paths["papers_json"], papers)
        if overwrite or not source_paths["cluster_seeds_json"].exists():
            _write_json(source_paths["cluster_seeds_json"], payload.get("cluster_seeds") or {})
        paths.update({key: str(path) for key, path in source_paths.items()})

    start = time.perf_counter()
    paths, raw_planner_index_metrics = write_raw_arrow_batch_lookup_indexes(
        paths,
        output_dir,
        overwrite=overwrite,
    )
    write_raw_planner_indexes_seconds = time.perf_counter() - start
    physical_layout = raw_planner_arrow_physical_layout(paths)

    name_counts_index_metrics: dict[str, Any] = {"skipped": True}
    write_name_counts_index_seconds = 0.0
    if not skip_name_counts_index:
        start = time.perf_counter()
        index_root = output_root if name_counts_index_root is None else name_counts_index_root
        name_counts_index_path, name_counts_index_metrics = write_name_counts_index(
            index_root,
            overwrite=overwrite_name_counts_index,
        )
        write_name_counts_index_seconds = time.perf_counter() - start
        paths["name_counts_index"] = name_counts_index_path

    manifest_paths = _portable_manifest_paths(paths, output_dir)
    manifest = {
        "schema": FEATURE_BLOCK_ARROW_MANIFEST_SCHEMA_VERSION,
        "dataset": dataset_name,
        "source_path": str(input_json),
        "conversion_kind": "service-json",
        "signature_count": len(dataset.signatures),
        "paper_count": len(dataset.papers),
        "paper_embedding_count": len(specter_embeddings or {}),
        "cluster_seeds_require_count": len(dataset.cluster_seeds_require),
        "cluster_seeds_disallow_count": len(dataset.cluster_seeds_disallow),
        "altered_cluster_signature_count": len(altered),
        "altered_cluster_signatures": altered,
        "paths": manifest_paths,
        "physical_layout": physical_layout,
        "raw_planner_batch_indexes": raw_planner_index_metrics,
        "name_counts_index": name_counts_index_metrics,
        "name_tuples": "default packaged filtered aliases",
        "timings_seconds": {
            "load_json_seconds": load_seconds,
            "anddata_seconds": anddata_seconds,
            "write_arrow_seconds": write_arrow_seconds,
            "write_raw_planner_indexes_seconds": write_raw_planner_indexes_seconds,
            "write_name_counts_index_seconds": write_name_counts_index_seconds,
        },
    }
    if validate:
        manifest["validation"] = validate_arrow_dataset_manifest(
            manifest,
            require_embeddings=specter_embeddings is not None,
            require_name_counts_index=not skip_name_counts_index,
            base_dir=output_dir,
        )
    _replace_json(output_dir / "manifest.json", manifest)
    _upsert_root_manifest(output_root, dataset_name=dataset_name, dataset_dir=output_dir)
    return manifest


def convert_runtime_dataset_to_arrow(
    *,
    sources: RuntimeDatasetSources,
    output_dir: Path,
    root_manifest_dir: Path,
    name_counts_index_root: Path | None,
    n_jobs: int,
    overwrite: bool,
    skip_name_counts_index: bool,
    overwrite_name_counts_index: bool = False,
    include_empty_cluster_seeds: bool = False,
    selected_embedding: str | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Write one benchmark or linker-replay dataset as Arrow artifacts."""

    from s2and.data import ANDData
    from s2and.incremental_linking.feature_block import (
        FEATURE_BLOCK_ARROW_MANIFEST_SCHEMA_VERSION,
        raw_planner_arrow_physical_layout,
        write_name_counts_index,
        write_raw_arrow_batch_lookup_indexes,
    )
    from scripts.arrow_conversion_helpers import write_feature_block_arrow_from_anddata

    dataset_name = sources.dataset
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory already contains files for dataset {dataset_name!r}: {output_dir}. "
            "Use --overwrite to regenerate it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_existing_root_manifest(root_manifest_dir / "manifest.json")

    start = time.perf_counter()
    dataset = ANDData(
        signatures=str(sources.signatures_path),
        papers=str(sources.papers_path),
        name=dataset_name,
        mode="train" if sources.clusters_path is not None else "inference",
        specter_embeddings=None,
        clusters=str(sources.clusters_path) if sources.clusters_path is not None else None,
        block_type="s2",
        train_pairs=None,
        val_pairs=None,
        test_pairs=None,
        train_pairs_size=100000,
        val_pairs_size=10000,
        test_pairs_size=10000,
        n_jobs=n_jobs,
        load_name_counts=not skip_name_counts_index,
        preprocess=True,
        random_seed=42,
        name_tuples="filtered",
        use_orcid_id=True,
        compute_reference_features=False,
    )
    anddata_seconds = time.perf_counter() - start

    start = time.perf_counter()
    paths = write_feature_block_arrow_from_anddata(
        dataset,
        output_dir,
        signature_ids=list(dataset.signatures),
        include_specter=False,
        include_empty_cluster_seeds=include_empty_cluster_seeds,
        overwrite=overwrite,
    )
    write_common_seconds = time.perf_counter() - start

    if sources.clusters_path is not None:
        output_clusters_path = output_dir / f"{dataset_name}_clusters.json"
        if overwrite or not output_clusters_path.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sources.clusters_path, output_clusters_path)
        paths["clusters"] = str(output_clusters_path)

    needed_paper_ids = {str(signature.paper_id) for signature in dataset.signatures.values()}
    specter_reports: dict[str, Any] = {}
    if sources.specter_path is not None:
        specter_reports["specter"] = _write_specter_arrow(
            source_path=sources.specter_path,
            output_path=output_dir / "specter.arrow",
            needed_paper_ids=needed_paper_ids,
            overwrite=overwrite,
        )
        paths["specter"] = str(output_dir / "specter.arrow")
    if sources.specter2_path is not None:
        specter_reports["specter2"] = _write_specter_arrow(
            source_path=sources.specter2_path,
            output_path=output_dir / "specter2.arrow",
            needed_paper_ids=needed_paper_ids,
            overwrite=overwrite,
        )
        paths["specter2"] = str(output_dir / "specter2.arrow")
        if selected_embedding == "specter2" or paths.get("specter") is None:
            paths["specter"] = str(output_dir / "specter2.arrow")

    start = time.perf_counter()
    paths, raw_planner_index_metrics = write_raw_arrow_batch_lookup_indexes(
        paths,
        output_dir,
        overwrite=overwrite,
    )
    write_raw_planner_indexes_seconds = time.perf_counter() - start
    physical_layout = raw_planner_arrow_physical_layout(paths)
    if sources.specter2_path is not None:
        _add_extra_specter_index_and_layout(
            paths=paths,
            raw_planner_index_metrics=raw_planner_index_metrics,
            physical_layout=physical_layout,
            table_key="specter2",
            output_dir=output_dir,
            overwrite=overwrite,
        )

    name_counts_index_metrics: dict[str, Any] = {"skipped": True}
    write_name_counts_index_seconds = 0.0
    if not skip_name_counts_index:
        start = time.perf_counter()
        index_root = root_manifest_dir if name_counts_index_root is None else name_counts_index_root
        name_counts_index_path, name_counts_index_metrics = write_name_counts_index(
            index_root,
            overwrite=overwrite_name_counts_index,
        )
        write_name_counts_index_seconds = time.perf_counter() - start
        paths["name_counts_index"] = name_counts_index_path

    manifest_paths = _portable_manifest_paths(paths, output_dir)
    manifest = {
        "schema": FEATURE_BLOCK_ARROW_MANIFEST_SCHEMA_VERSION,
        "dataset": dataset_name,
        "source_dir": str(sources.source_dir),
        "conversion_kind": "table-runtime",
        "signature_count": len(dataset.signatures),
        "paper_count": len(dataset.papers),
        "cluster_count": len(dataset.clusters or {}),
        "paths": manifest_paths,
        "specter": specter_reports,
        "physical_layout": physical_layout,
        "raw_planner_batch_indexes": raw_planner_index_metrics,
        "name_counts_index": name_counts_index_metrics,
        "name_tuples": "default packaged filtered aliases",
        "timings_seconds": {
            "anddata_seconds": anddata_seconds,
            "write_common_seconds": write_common_seconds,
            "write_raw_planner_indexes_seconds": write_raw_planner_indexes_seconds,
            "write_name_counts_index_seconds": write_name_counts_index_seconds,
        },
    }
    if validate:
        manifest["validation"] = validate_arrow_dataset_manifest(
            manifest,
            require_embeddings=sources.specter_path is not None or sources.specter2_path is not None,
            require_name_counts_index=not skip_name_counts_index,
            base_dir=output_dir,
        )
    _replace_json(output_dir / "manifest.json", manifest)
    _upsert_root_manifest(root_manifest_dir, dataset_name=dataset_name, dataset_dir=output_dir)
    return manifest


def _read_arrow_table(path: str | Path) -> Any:
    import pyarrow as pa

    with pa.memory_map(str(path), "r") as source:
        return pa.ipc.open_file(source).read_all()


def _ensure_arrow_column_type(table: Any, column: str, predicate: Callable[[Any], bool], expected: str) -> None:
    field_index = table.schema.get_field_index(column)
    if field_index < 0:
        raise KeyError(f"Arrow table is missing required column {column!r}")
    field_type = table.schema.field(field_index).type
    if not predicate(field_type):
        raise ValueError(f"Arrow column {column!r} expected {expected}, got {field_type}")


def _ensure_string_column(table: Any, column: str) -> None:
    import pyarrow as pa

    _ensure_arrow_column_type(table, column, pa.types.is_string, "string")


def _ensure_integer_column(table: Any, column: str) -> None:
    import pyarrow as pa

    _ensure_arrow_column_type(table, column, pa.types.is_integer, "integer")


def _ensure_specter_embedding_column(table: Any) -> None:
    import pyarrow as pa

    field_index = table.schema.get_field_index("embedding")
    if field_index < 0:
        raise KeyError("Arrow table is missing required column 'embedding'")
    field_type = table.schema.field(field_index).type
    if not (pa.types.is_fixed_size_list(field_type) and pa.types.is_float32(field_type.value_type)):
        raise ValueError(f"Arrow column 'embedding' expected fixed_size_list<float32>, got {field_type}")


def _arrow_contract_required_columns(table_name: str) -> list[Mapping[str, Any]]:
    contract = _load_json(ARROW_SCHEMA_CONTRACT_PATH)
    if not isinstance(contract, Mapping):
        raise TypeError(f"Arrow schema contract must contain an object: {ARROW_SCHEMA_CONTRACT_PATH}")
    tables = contract.get("tables")
    if not isinstance(tables, Mapping):
        raise ValueError(f"Arrow schema contract is missing tables: {ARROW_SCHEMA_CONTRACT_PATH}")
    columns = tables.get(table_name)
    if not isinstance(columns, Sequence) or isinstance(columns, str | bytes):
        raise ValueError(f"Arrow schema contract is missing table {table_name!r}: {ARROW_SCHEMA_CONTRACT_PATH}")
    return [column for column in columns if isinstance(column, Mapping) and bool(column.get("required"))]


def _ensure_arrow_contract_column_type(table: Any, table_name: str, column: str, datatype: str) -> None:
    import pyarrow as pa

    def is_string_list(value: Any) -> bool:
        return pa.types.is_list(value) and pa.types.is_string(value.value_type)

    def is_float32_fixed_size_list(value: Any) -> bool:
        return pa.types.is_fixed_size_list(value) and pa.types.is_float32(value.value_type)

    match datatype:
        case "string":
            predicate = pa.types.is_string
        case "int64":
            predicate = pa.types.is_int64
        case "bool":
            predicate = pa.types.is_boolean
        case "list<string>":
            predicate = is_string_list
        case "fixed_size_list<float32>":
            predicate = is_float32_fixed_size_list
        case _:
            raise ValueError(f"Arrow schema contract has unsupported datatype {datatype!r} for {table_name}.{column}")
    _ensure_arrow_column_type(table, column, predicate, datatype)


def _ensure_arrow_contract_required_columns(table: Any, table_name: str) -> None:
    for column_spec in _arrow_contract_required_columns(table_name):
        column = str(column_spec["name"])
        datatype = str(column_spec["datatype"])
        try:
            _ensure_arrow_contract_column_type(table, table_name, column, datatype)
        except KeyError as exc:
            raise KeyError(f"{table_name} Arrow table is missing required column {column!r}") from exc


def _table_values(table: Any, column: str) -> list[Any]:
    if column not in table.column_names:
        raise ValueError(f"Arrow table is missing required column {column!r}")
    return table[column].to_pylist()


def _required_string_values(table: Any, column: str, *, label: str) -> list[str]:
    values = _table_values(table, column)
    out: list[str] = []
    for row_index, value in enumerate(values):
        if value is None:
            raise ValueError(f"{label} contains null value at row {row_index}")
        text = str(value)
        if not text:
            raise ValueError(f"{label} contains empty value at row {row_index}")
        out.append(text)
    return out


def _duplicate_values(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        key = str(value)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return sorted(duplicates)


def _ensure_unique(values: Sequence[Any], *, label: str) -> None:
    duplicates = _duplicate_values(values)
    if duplicates:
        raise ValueError(f"{label} contains duplicate ids: {duplicates[:10]}")


def _ensure_subset(values: Sequence[Any], allowed: set[str], *, label: str) -> None:
    missing = sorted({str(value) for value in values if str(value) not in allowed})
    if missing:
        raise ValueError(f"{label} contains ids missing from parent table: {missing[:10]}")


def validate_arrow_dataset_manifest(
    manifest: Mapping[str, Any],
    *,
    require_embeddings: bool,
    require_name_counts_index: bool,
    base_dir: Path | None = None,
    require_complete_embeddings: bool = False,
) -> dict[str, Any]:
    """Validate the generated Arrow tables and return compact audit metrics."""

    if not isinstance(manifest.get("paths"), Mapping):
        raise ValueError("manifest is missing paths")
    paths = {str(key): str(_resolve_manifest_path(value, base_dir)) for key, value in manifest["paths"].items()}
    required_paths = ["signatures", "papers", "paper_authors"]
    if require_embeddings:
        if "specter" not in paths and "specter2" in paths:
            paths["specter"] = paths["specter2"]
        required_paths.append("specter")
    if require_name_counts_index:
        required_paths.append("name_counts_index")
    missing_paths = [key for key in required_paths if key not in paths or not Path(paths[key]).exists()]
    if missing_paths:
        raise FileNotFoundError(f"manifest is missing required path keys/files: {missing_paths}")

    signatures = _read_arrow_table(paths["signatures"])
    papers = _read_arrow_table(paths["papers"])
    paper_authors = _read_arrow_table(paths["paper_authors"])
    _ensure_arrow_contract_required_columns(signatures, "signatures")
    _ensure_arrow_contract_required_columns(papers, "papers")
    _ensure_arrow_contract_required_columns(paper_authors, "paper_authors")
    signature_ids = _required_string_values(signatures, "signature_id", label="signatures.signature_id")
    signature_paper_ids = _required_string_values(signatures, "paper_id", label="signatures.paper_id")
    paper_ids = _required_string_values(papers, "paper_id", label="papers.paper_id")
    paper_author_paper_ids = _required_string_values(paper_authors, "paper_id", label="paper_authors.paper_id")
    _required_string_values(paper_authors, "author_name", label="paper_authors.author_name")
    paper_author_positions = _table_values(paper_authors, "position")
    _ensure_unique(signature_ids, label="signatures.signature_id")
    _ensure_unique(paper_ids, label="papers.paper_id")
    paper_id_set = set(paper_ids)
    signature_id_set = set(signature_ids)
    _ensure_subset(signature_paper_ids, paper_id_set, label="signatures.paper_id")
    _ensure_subset(paper_author_paper_ids, paper_id_set, label="paper_authors.paper_id")
    _ensure_unique(
        [
            f"{paper_id}\x00{position}"
            for paper_id, position in zip(paper_author_paper_ids, paper_author_positions, strict=True)
        ],
        label="paper_authors.(paper_id,position)",
    )

    metrics: dict[str, Any] = {
        "signature_count": int(signatures.num_rows),
        "paper_count": int(papers.num_rows),
        "paper_author_count": int(paper_authors.num_rows),
        "required_paths_present": True,
    }
    if int(manifest.get("signature_count", signatures.num_rows)) != signatures.num_rows:
        raise ValueError("manifest signature_count does not match signatures.arrow")
    if int(manifest.get("paper_count", papers.num_rows)) != papers.num_rows:
        raise ValueError("manifest paper_count does not match papers.arrow")

    if require_embeddings:
        specter = _read_arrow_table(paths["specter"])
        _ensure_arrow_contract_required_columns(specter, "specter")
        specter_paper_ids = _required_string_values(specter, "paper_id", label="specter.paper_id")
        _ensure_unique(specter_paper_ids, label="specter.paper_id")
        missing_embeddings = sorted(set(signature_paper_ids) - set(specter_paper_ids))
        metrics["specter_count"] = int(specter.num_rows)
        metrics["missing_specter_paper_count"] = int(len(missing_embeddings))
        metrics["missing_specter_paper_examples"] = missing_embeddings[:10]
        if require_complete_embeddings and missing_embeddings:
            raise ValueError(
                "require_complete_embeddings=True but specter Arrow is missing embeddings for referenced "
                f"paper ids: {missing_embeddings[:10]}"
            )

    cluster_seed_path = paths.get("cluster_seeds")
    if cluster_seed_path is not None and Path(cluster_seed_path).exists():
        cluster_seeds = _read_arrow_table(cluster_seed_path)
        _ensure_string_column(cluster_seeds, "signature_id")
        _ensure_string_column(cluster_seeds, "cluster_id")
        seed_signature_ids = [str(value) for value in _table_values(cluster_seeds, "signature_id")]
        seed_cluster_ids = [str(value) for value in _table_values(cluster_seeds, "cluster_id")]
        _ensure_unique(seed_signature_ids, label="cluster_seeds.signature_id")
        _ensure_subset(seed_signature_ids, signature_id_set, label="cluster_seeds.signature_id")
        empty_cluster_ids = [cluster_id for cluster_id in seed_cluster_ids if not cluster_id]
        if empty_cluster_ids:
            raise ValueError("cluster_seeds.cluster_id contains empty values")
        metrics["cluster_seed_count"] = int(cluster_seeds.num_rows)
    else:
        seed_signature_ids = []

    disallow_path = paths.get("cluster_seed_disallows")
    if disallow_path is not None and Path(disallow_path).exists():
        disallows = _read_arrow_table(disallow_path)
        _ensure_string_column(disallows, "signature_id_1")
        _ensure_string_column(disallows, "signature_id_2")
        left_ids = [str(value) for value in _table_values(disallows, "signature_id_1")]
        right_ids = [str(value) for value in _table_values(disallows, "signature_id_2")]
        _ensure_subset(left_ids, signature_id_set, label="cluster_seed_disallows.signature_id_1")
        _ensure_subset(right_ids, signature_id_set, label="cluster_seed_disallows.signature_id_2")
        normalized_pairs: set[tuple[str, str]] = set()
        for left, right in zip(left_ids, right_ids, strict=True):
            if left == right:
                raise ValueError(f"cluster_seed_disallows contains self-pair: {left!r}")
            pair = tuple(sorted((left, right)))
            if pair in normalized_pairs:
                raise ValueError(f"cluster_seed_disallows contains duplicate undirected pair: {pair!r}")
            normalized_pairs.add(pair)
        metrics["cluster_seed_disallow_count"] = int(disallows.num_rows)

    altered_path = paths.get("altered_cluster_signatures")
    if altered_path is not None and Path(altered_path).exists():
        altered = _read_arrow_table(altered_path)
        _ensure_string_column(altered, "signature_id")
        altered_signature_ids = [str(value) for value in _table_values(altered, "signature_id")]
        _ensure_unique(altered_signature_ids, label="altered_cluster_signatures.signature_id")
        _ensure_subset(altered_signature_ids, signature_id_set, label="altered_cluster_signatures.signature_id")
        if altered_signature_ids:
            _ensure_subset(
                altered_signature_ids,
                set(seed_signature_ids),
                label="altered_cluster_signatures.signature_id",
            )
        metrics["altered_cluster_signature_count"] = int(altered.num_rows)

    if require_name_counts_index:
        require_name_counts_index_artifact(
            paths["name_counts_index"],
            context="convert_to_arrow dataset validation",
            producer_hint="rerun scripts/convert_to_arrow.py name-counts-index or rebuild the release bundle",
        )
        metrics["name_counts_index_present"] = True

    physical_layout = manifest.get("physical_layout")
    if isinstance(physical_layout, Mapping):
        from s2and.incremental_linking.feature_block import (
            RAW_PLANNER_ARROW_KEY_COLUMNS,
            validate_arrow_batch_lookup_index,
        )

        tables = physical_layout.get("tables", {})
        if isinstance(tables, Mapping):
            for table_name, raw_layout in tables.items():
                if not isinstance(raw_layout, Mapping):
                    raise ValueError(f"physical_layout.tables.{table_name} must be an object")
                table_key = str(table_name)
                max_rows = int(raw_layout.get("max_record_batch_rows", 0))
                actual_max_rows = int(raw_layout.get("actual_max_batch_rows", 0))
                if max_rows > 0 and actual_max_rows > max_rows:
                    raise ValueError(
                        f"physical_layout.tables.{table_name} exceeds max batch rows: "
                        f"{actual_max_rows} > {max_rows}"
                    )
                if bool(raw_layout.get("batch_index_present", False)):
                    if table_key not in paths:
                        raise FileNotFoundError(
                            f"physical_layout.tables.{table_name} has batch_index_present but manifest.paths "
                            f"is missing {table_key!r}"
                        )
                    index_key = str(raw_layout.get("batch_index_path_key") or f"{table_key}_batch_index")
                    if index_key not in paths:
                        raise FileNotFoundError(
                            f"physical_layout.tables.{table_name} has batch_index_present but manifest.paths "
                            f"is missing {index_key!r}"
                        )
                    if not Path(paths[index_key]).exists():
                        raise FileNotFoundError(
                            f"physical_layout.tables.{table_name} batch index is missing: {paths[index_key]}"
                        )
                    key_column = str(raw_layout.get("key") or RAW_PLANNER_ARROW_KEY_COLUMNS.get(table_key, ""))
                    if not key_column:
                        raise ValueError(f"physical_layout.tables.{table_name} is missing key for batch index")
                    validate_arrow_batch_lookup_index(
                        paths[table_key],
                        paths[index_key],
                        key_column=key_column,
                        expected_row_count=int(raw_layout["row_count"]) if "row_count" in raw_layout else None,
                    )
    return metrics


def validate_arrow_dataset_dir(
    dataset_dir: Path,
    *,
    require_embeddings: bool,
    require_name_counts_index: bool,
    require_complete_embeddings: bool = False,
) -> dict[str, Any]:
    manifest = _load_json(dataset_dir / "manifest.json")
    if not isinstance(manifest, Mapping):
        raise TypeError(f"dataset manifest must contain an object: {dataset_dir / 'manifest.json'}")
    return validate_arrow_dataset_manifest(
        manifest,
        require_embeddings=require_embeddings,
        require_name_counts_index=require_name_counts_index,
        require_complete_embeddings=require_complete_embeddings,
        base_dir=dataset_dir,
    )


def _print_report(report: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            {
                "dataset": report["dataset"],
                "signature_count": report["signature_count"],
                "paper_count": report["paper_count"],
                "paths": report["paths"],
                "timings_seconds": report.get("timings_seconds", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _run_service_json(args: argparse.Namespace) -> None:
    dataset_name = str(args.dataset_name or args.input_json.stem)
    report = convert_service_json_to_arrow(
        input_json=args.input_json,
        output_root=args.output_root,
        dataset_name=dataset_name,
        name_counts_index_root=args.name_counts_index_root,
        n_jobs=int(args.n_jobs),
        overwrite=bool(args.overwrite),
        skip_name_counts_index=bool(args.skip_name_counts_index),
        overwrite_name_counts_index=bool(args.overwrite_name_counts_index),
        copy_source_json=bool(args.copy_source_json),
        validate=not bool(args.skip_validation),
    )
    _print_report(report)


def _selected_runtime_dataset_names(
    *,
    datasets: Sequence[str] | None,
    run_full: bool,
    discover: Callable[[], Sequence[str]],
    command: str,
) -> list[str]:
    if datasets is not None:
        return [str(dataset) for dataset in datasets]
    if run_full:
        return [str(dataset) for dataset in discover()]
    raise ValueError(f"{command} requires --datasets DATASET... for a bounded run or --run-full for full discovery")


def _run_benchmark(args: argparse.Namespace) -> None:
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_names = _selected_runtime_dataset_names(
        datasets=getattr(args, "datasets", None),
        run_full=bool(getattr(args, "run_full", False)),
        discover=lambda: discover_benchmark_datasets(args.source_root),
        command="benchmark",
    )
    if not dataset_names:
        raise ValueError(f"No benchmark datasets found under {args.source_root}")
    name_counts_index_overwritten = False
    reports = []
    for dataset_name in dataset_names:
        start = time.perf_counter()
        overwrite_name_counts_index = bool(args.overwrite_name_counts_index) and not name_counts_index_overwritten
        report = convert_runtime_dataset_to_arrow(
            sources=benchmark_dataset_sources(args.source_root, dataset_name),
            output_dir=output_root / dataset_name,
            root_manifest_dir=output_root,
            name_counts_index_root=args.name_counts_index_root,
            n_jobs=int(args.n_jobs),
            overwrite=bool(args.overwrite),
            skip_name_counts_index=bool(args.skip_name_counts_index),
            overwrite_name_counts_index=overwrite_name_counts_index,
            selected_embedding=None,
            validate=not bool(args.skip_validation),
        )
        if overwrite_name_counts_index:
            name_counts_index_overwritten = True
        report["total_seconds"] = time.perf_counter() - start
        reports.append(report)
        print(json.dumps({"dataset": dataset_name, "total_seconds": report["total_seconds"]}, sort_keys=True))
    print(json.dumps({"datasets": [report["dataset"] for report in reports]}, indent=2, sort_keys=True))


def _run_linker_replay(args: argparse.Namespace) -> None:
    output_root = args.output_root
    datasets_root = output_root / "datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)
    dataset_names = _selected_runtime_dataset_names(
        datasets=getattr(args, "datasets", None),
        run_full=bool(getattr(args, "run_full", False)),
        discover=lambda: discover_linker_replay_datasets(args.raw_root, args.embeddings_root),
        command="linker-replay",
    )
    if not dataset_names:
        raise ValueError(f"No linker replay datasets found under {args.raw_root}")
    name_counts_index_overwritten = False
    reports = []
    for dataset_name in dataset_names:
        start = time.perf_counter()
        overwrite_name_counts_index = bool(args.overwrite_name_counts_index) and not name_counts_index_overwritten
        report = convert_runtime_dataset_to_arrow(
            sources=linker_replay_dataset_sources(args.raw_root, args.embeddings_root, dataset_name),
            output_dir=datasets_root / dataset_name,
            root_manifest_dir=output_root,
            name_counts_index_root=args.name_counts_index_root,
            n_jobs=int(args.n_jobs),
            overwrite=bool(args.overwrite),
            skip_name_counts_index=bool(args.skip_name_counts_index),
            overwrite_name_counts_index=overwrite_name_counts_index,
            selected_embedding="specter2",
            validate=not bool(args.skip_validation),
        )
        if overwrite_name_counts_index:
            name_counts_index_overwritten = True
        report["total_seconds"] = time.perf_counter() - start
        reports.append(report)
        print(json.dumps({"dataset": dataset_name, "total_seconds": report["total_seconds"]}, sort_keys=True))
    print(json.dumps({"datasets": [report["dataset"] for report in reports]}, indent=2, sort_keys=True))


def _run_name_counts_index(args: argparse.Namespace) -> None:
    from s2and.incremental_linking.feature_block import write_name_counts_index

    index_path, metrics = write_name_counts_index(args.output_root, overwrite=bool(args.overwrite))
    print(json.dumps({"name_counts_index": index_path, "metrics": metrics}, indent=2, sort_keys=True))


def _run_validate(args: argparse.Namespace) -> None:
    metrics = validate_arrow_dataset_dir(
        args.dataset_dir,
        require_embeddings=bool(args.require_embeddings),
        require_name_counts_index=bool(args.require_name_counts_index),
        require_complete_embeddings=bool(args.require_complete_embeddings),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _run_refresh_root_manifest(args: argparse.Namespace) -> None:
    root_manifest_path = args.output_root / "manifest.json"
    root_manifest = _load_json(root_manifest_path)
    if not isinstance(root_manifest, Mapping):
        raise TypeError(f"root manifest must contain an object: {root_manifest_path}")
    dataset_manifests = _root_manifest_entries_from_manifest(root_manifest, root_manifest_path)
    replay_bundles = _root_manifest_replay_bundle_entries_from_manifest(root_manifest, root_manifest_path)
    refreshed = _write_root_manifest(
        args.output_root,
        dataset_manifests=dataset_manifests,
        replay_bundles=replay_bundles,
        output_root_label=args.output_root_label or str(root_manifest.get("output_root") or args.output_root),
    )
    print(
        json.dumps(
            {
                "dataset_count": len(refreshed["dataset_manifests"]),
                "replay_bundle_count": len(refreshed.get("replay_bundles", [])),
                "manifest_path": str(root_manifest_path),
                "validation_command_count": len(refreshed.get("validation_commands", [])),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _add_common_runtime_args(parser: argparse.ArgumentParser, *, default_n_jobs: int) -> None:
    parser.add_argument("--name-counts-index-root", type=Path, default=None)
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--overwrite-name-counts-index",
        action="store_true",
        help="Rebuild the shared name-counts index once before reusing it for all converted datasets.",
    )
    parser.add_argument("--skip-name-counts-index", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")


def _add_runtime_dataset_selection_args(parser: argparse.ArgumentParser) -> None:
    dataset_selection = parser.add_mutually_exclusive_group(required=True)
    dataset_selection.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Convert only the named datasets.",
    )
    dataset_selection.add_argument(
        "--run-full",
        action="store_true",
        help="Discover and convert every eligible dataset under the configured roots.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    service = subparsers.add_parser("service-json", help="Convert one service-shaped inference JSON payload.")
    service.add_argument("--input-json", type=Path, required=True)
    service.add_argument("--output-root", type=Path, default=Path("scratch/inference_arrow"))
    service.add_argument("--dataset-name", default=None)
    service.add_argument("--copy-source-json", action="store_true")
    _add_common_runtime_args(service, default_n_jobs=4)
    service.set_defaults(func=_run_service_json)

    benchmark = subparsers.add_parser("benchmark", help="Convert benchmark dataset JSON/pickle files.")
    benchmark.add_argument("--source-root", type=Path, default=Path("s2and/data/s2and_mini"))
    benchmark.add_argument("--output-root", type=Path, default=Path("s2and/data/s2and_mini_arrow"))
    _add_runtime_dataset_selection_args(benchmark)
    _add_common_runtime_args(benchmark, default_n_jobs=20)
    benchmark.set_defaults(func=_run_benchmark)

    linker_replay = subparsers.add_parser("linker-replay", help="Convert linker replay raw JSON plus SPECTER2 files.")
    linker_replay.add_argument("--raw-root", type=Path, required=True)
    linker_replay.add_argument("--embeddings-root", type=Path, required=True)
    linker_replay.add_argument("--output-root", type=Path, required=True)
    _add_runtime_dataset_selection_args(linker_replay)
    _add_common_runtime_args(linker_replay, default_n_jobs=20)
    linker_replay.set_defaults(func=_run_linker_replay)

    name_counts = subparsers.add_parser(
        "name-counts-index", help="Generate the shared manifest-backed name-count index."
    )
    name_counts.add_argument("--output-root", type=Path, required=True)
    name_counts.add_argument("--overwrite", action="store_true")
    name_counts.set_defaults(func=_run_name_counts_index)

    validate = subparsers.add_parser("validate", help="Validate one generated Arrow dataset manifest.")
    validate.add_argument("--dataset-dir", type=Path, required=True)
    validate.add_argument("--require-embeddings", action="store_true")
    validate.add_argument("--require-name-counts-index", action="store_true")
    validate.add_argument(
        "--require-complete-embeddings",
        action="store_true",
        help="Fail if any referenced paper is missing from the embedding table.",
    )
    validate.set_defaults(func=_run_validate)

    refresh_root = subparsers.add_parser(
        "refresh-root-manifest",
        help="Refresh root manifest checksums, audits, replay bundle metadata, and validation commands.",
    )
    refresh_root.add_argument("--output-root", type=Path, required=True)
    refresh_root.add_argument(
        "--output-root-label",
        default=None,
        help="Optional logical root to write into output_root, e.g. the public S3 prefix.",
    )
    refresh_root.set_defaults(func=_run_refresh_root_manifest)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
