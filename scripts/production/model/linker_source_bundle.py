"""Assemble and validate the final linker source bundle and data root."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and.incremental_linking_training.classic import load_bundle  # noqa: E402
from scripts.convert_to_arrow import _write_root_manifest  # noqa: E402
from scripts.production.model.train_linker_and_finalize import (  # noqa: E402
    _preflight_source_rows,
    _validate_source_bundle_support_files,
)
from scripts.verification.validate_local_arrow_release import validate_release_root  # noqa: E402

SPEC_SCHEMA = "s2and_linker_source_member_spec_v1"
MANIFEST_SCHEMA = "s2and_linker_source_bundle_manifest_v1"
SOURCE_MANIFEST_NAME = "source_bundle_manifest.json"


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _relative(value: Any) -> Path:
    path = Path(str(value).replace("\\", "/"))
    if not str(value) or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"expected a relative path without '..': {value!r}")
    return path


def _load_spec(path: Path) -> list[dict[str, str]]:
    payload = _read_json(path)
    if set(payload) != {"schema", "members"} or payload["schema"] != SPEC_SCHEMA:
        raise ValueError(f"{path} must contain {SPEC_SCHEMA!r} schema and members")
    raw_members = payload["members"]
    if not isinstance(raw_members, list) or not raw_members:
        raise ValueError(f"{path} members must be a nonempty list")
    members = []
    for raw in raw_members:
        if not isinstance(raw, Mapping) or set(raw) != {"path", "role"}:
            raise ValueError(f"{path} members must contain exactly path/role")
        member = {"path": _relative(raw["path"]).as_posix(), "role": str(raw["role"]).strip()}
        if not member["role"]:
            raise ValueError(f"{path} contains an empty role")
        members.append(member)
    paths = [member["path"] for member in members]
    if len(set(paths)) != len(paths) or "bundle.json" not in paths:
        raise ValueError(f"{path} must contain unique paths including bundle.json")
    return sorted(members, key=lambda item: item["path"])


def _root_entries(root: Path) -> list[dict[str, Any]]:
    entries = _read_json(root / "manifest.json").get("dataset_manifests")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{root / 'manifest.json'} must contain dataset_manifests")
    if any(not isinstance(entry, Mapping) or not entry.get("manifest_path") for entry in entries):
        raise ValueError(f"{root / 'manifest.json'} has an invalid dataset entry")
    return [dict(entry) for entry in entries]


def _entry(root_name: str, root: Path, path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "root": root_name,
        "path": path.relative_to(root).as_posix(),
        "role": role,
        "byte_count": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _verify(entry: Mapping[str, Any], roots: Mapping[str, Path]) -> None:
    root_name = str(entry.get("root"))
    if root_name not in roots:
        raise ValueError(f"invalid inventory root: {root_name!r}")
    path = roots[root_name] / _relative(entry.get("path"))
    if not path.is_file():
        raise ValueError(f"inventoried file is missing: {path}")
    if entry.get("byte_count") != path.stat().st_size:
        raise ValueError(f"byte_count mismatch for {path}")
    actual = _sha256(path)
    if entry.get("sha256") != actual:
        raise ValueError(f"sha256 mismatch for {path}: expected={entry.get('sha256')} actual={actual}")


def _name_count_sha(path: Path) -> str:
    generation = _read_json(path).get("artifact_generation")
    files = generation.get("files") if isinstance(generation, Mapping) else None
    entry = files.get("name_counts_index") if isinstance(files, Mapping) else None
    if not isinstance(entry, Mapping) or not isinstance(entry.get("sha256"), str):
        raise ValueError(f"{path} does not bind name_counts_index")
    return str(entry["sha256"])


def _arrow_specs(
    root_name: str,
    root: Path,
    entries: Sequence[Mapping[str, Any]],
    prefix: Path,
    role_prefix: str,
) -> list[tuple[str, Path, Path, str]]:
    return [
        (root_name, root, prefix / _relative(entry["manifest_path"]), f"arrow.{role_prefix}.{entry['dataset']}")
        for entry in entries
    ]


def _all_arrow_specs(source_root: Path, data_root: Path) -> list[tuple[str, Path, Path, str]]:
    specs = _arrow_specs("source_bundle", source_root, _root_entries(source_root), Path(), "source")
    specs += _arrow_specs("data_root", data_root, _root_entries(data_root), Path(), "benchmark")
    specs += _arrow_specs(
        "data_root",
        data_root,
        _root_entries(data_root / "linker_replay"),
        Path("linker_replay"),
        "replay",
    )
    return specs


def validate_source_bundle(source_bundle_root: Path, data_root: Path) -> dict[str, Any]:
    """Validate one assembled source bundle and final data root."""

    source_bundle_root, data_root = source_bundle_root.resolve(), data_root.resolve()
    roots = {"source_bundle": source_bundle_root, "data_root": data_root}
    manifest_path = source_bundle_root / SOURCE_MANIFEST_NAME
    manifest = _read_json(manifest_path)
    members, manifests = manifest.get("members"), manifest.get("manifests")
    if manifest.get("schema") != MANIFEST_SCHEMA or not isinstance(members, list) or not isinstance(manifests, list):
        raise ValueError(f"invalid {SOURCE_MANIFEST_NAME}")
    for item in [*members, *manifests]:
        if not isinstance(item, Mapping):
            raise ValueError("inventory entries must be objects")
        _verify(item, roots)

    declared = {str(item["path"]) for item in members if item.get("root") == "source_bundle"}
    actual = {
        path.relative_to(source_bundle_root).as_posix()
        for path in source_bundle_root.rglob("*")
        if path.is_file()
        and path.name not in {SOURCE_MANIFEST_NAME, "manifest.json"}
        and path.relative_to(source_bundle_root).parts[0] not in {"datasets", "name_counts_index"}
    }
    if declared != actual:
        raise ValueError(
            f"source member inventory mismatch: missing={sorted(declared - actual)} "
            f"undeclared={sorted(actual - declared)}"
        )
    assets = _read_json(source_bundle_root / "bundle.json").get("assets", {})
    candidates = assets.get("candidate_members", {}).get("datasets", {}) if isinstance(assets, Mapping) else {}
    if not isinstance(candidates, Mapping):
        raise ValueError("bundle.json candidate_members.datasets must be a mapping")
    missing = sorted({_relative(value).as_posix() for value in candidates.values()} - declared)
    if missing:
        raise ValueError(f"bundle.json references uninventoried candidate files: {missing}")

    arrows = {
        (str(item["root"]), str(item["path"])) for item in manifests if str(item.get("role", "")).startswith("arrow.")
    }
    arrow_specs = _all_arrow_specs(source_bundle_root, data_root)
    if arrows != {(name, path.as_posix()) for name, _root, path, _role in arrow_specs}:
        raise ValueError("nested Arrow manifest inventory mismatch")
    count_hashes = {_name_count_sha(roots[root] / path) for root, path in arrows}
    if count_hashes != {manifest.get("name_counts_manifest_sha256")}:
        raise ValueError(f"name-count binding mismatch: {sorted(count_hashes)}")

    validate_release_root(source_bundle_root, include_replay_bundles=False)
    validate_release_root(data_root)
    bundle = load_bundle(source_bundle_root)
    _validate_source_bundle_support_files(bundle, require_training_contract=True)
    source_summary, _arrow_paths = _preflight_source_rows(
        bundle,
        table_keys=None,
        datasets=None,
        limit_rows=None,
        require_full_tables=True,
        name_counts_index_root=None,
    )
    return {
        "source_manifest_sha256": _sha256(manifest_path),
        "data_root_manifest_sha256": _sha256(data_root / "manifest.json"),
        "name_counts_manifest_sha256": next(iter(count_hashes)),
        "selected_source_rows": int(source_summary["total_selected_rows"]),
    }


def assemble_source_bundle(
    *,
    member_spec_path: Path,
    source_root: Path,
    benchmark_arrow_root: Path,
    replay_arrow_root: Path,
    output_source_bundle: Path,
    output_data_root: Path,
) -> dict[str, Any]:
    """Copy reviewed inputs into two fresh roots, inventory, and validate."""

    member_spec_path, source_root, benchmark_arrow_root, replay_arrow_root = (
        path.resolve() for path in (member_spec_path, source_root, benchmark_arrow_root, replay_arrow_root)
    )
    output_source_bundle, output_data_root = output_source_bundle.resolve(), output_data_root.resolve()
    if output_source_bundle.exists() or output_data_root.exists():
        raise FileExistsError("assembly output directories must not exist")
    members = _load_spec(member_spec_path)
    benchmark_entries = _root_entries(benchmark_arrow_root)

    shutil.copytree(replay_arrow_root, output_source_bundle)
    for member in members:
        source, destination = source_root / member["path"], output_source_bundle / member["path"]
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copytree(benchmark_arrow_root, output_data_root)
    shutil.copytree(replay_arrow_root, output_data_root / "linker_replay")
    _write_root_manifest(
        output_data_root,
        dataset_manifests=benchmark_entries,
        replay_bundles=[{"bundle": "linker_replay", "manifest_path": "linker_replay/manifest.json"}],
    )

    inventory_members = [
        _entry("source_bundle", output_source_bundle, output_source_bundle / item["path"], item["role"])
        for item in members
    ]
    for root_name, root, path, role in (
        ("source_bundle", output_source_bundle, output_source_bundle / "LICENSE.txt", "source.helper"),
        ("data_root", output_data_root, output_data_root / "LICENSE.txt", "data.helper"),
        (
            "data_root",
            output_data_root,
            output_data_root / "linker_replay" / "LICENSE.txt",
            "replay.helper",
        ),
    ):
        inventory_members.append(_entry(root_name, root, path, role))

    specs = [
        ("source_bundle", output_source_bundle, Path("manifest.json"), "source.root"),
        ("source_bundle", output_source_bundle, Path("name_counts_index/manifest.json"), "source.name_counts"),
        ("data_root", output_data_root, Path("manifest.json"), "data.root"),
        ("data_root", output_data_root, Path("name_counts_index/manifest.json"), "data.name_counts"),
        ("data_root", output_data_root, Path("linker_replay/manifest.json"), "replay.root"),
        (
            "data_root",
            output_data_root,
            Path("linker_replay/name_counts_index/manifest.json"),
            "replay.name_counts",
        ),
    ]
    arrow_specs = _all_arrow_specs(output_source_bundle, output_data_root)
    specs += arrow_specs
    inventory_manifests = [_entry(name, root, root / path, role) for name, root, path, role in specs]
    count_hashes = {
        _name_count_sha((output_source_bundle if item["root"] == "source_bundle" else output_data_root) / item["path"])
        for item in inventory_manifests
        if str(item["role"]).startswith("arrow.")
    }
    if len(count_hashes) != 1:
        raise ValueError(f"input Arrow roots bind different name counts: {sorted(count_hashes)}")
    source_manifest = {
        "schema": MANIFEST_SCHEMA,
        "assembly_spec_sha256": _sha256(member_spec_path),
        "name_counts_manifest_sha256": next(iter(count_hashes)),
        "members": inventory_members,
        "manifests": inventory_manifests,
    }
    (output_source_bundle / SOURCE_MANIFEST_NAME).write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validate_source_bundle(output_source_bundle, output_data_root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    assemble = commands.add_parser("assemble-source-bundle")
    for option in (
        "member-spec",
        "source-root",
        "benchmark-arrow-root",
        "replay-arrow-root",
        "output-source-bundle",
        "output-data-root",
    ):
        assemble.add_argument(f"--{option}", type=Path, required=True)
    validate = commands.add_parser("validate-source-bundle")
    validate.add_argument("--source-bundle-root", type=Path, required=True)
    validate.add_argument("--data-root", type=Path, required=True)
    values = vars(parser.parse_args(argv))
    command = values.pop("command")
    report = (
        assemble_source_bundle(**values) if command == "assemble-source-bundle" else validate_source_bundle(**values)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
