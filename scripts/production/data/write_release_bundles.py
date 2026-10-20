"""Write the published inference, dataset, and source bundles from one prepared snapshot.

The snapshot is the output of ``package_corrected_datasets.py``. The dataset bundle
trains and evaluates models from published Arrow data alone: one complete benchmark
Arrow dataset per source with its held-out blocks and frozen test pairs, augmented
Arrow data with fixed pairs, and the incremental linker tasks. The source bundle
holds the editable records that regenerate the Arrow data. The inference bundle
holds the common count index and, when supplied, the complete model.

Each bundle mirrors one S3 prefix. Users sync bundles into one local root.
Files are hardlinked when the output shares the snapshot's volume, otherwise copied.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and.arrow_inputs import ArrowDataset  # noqa: E402
from s2and.arrow_training import load_signatures_from_arrow  # noqa: E402
from s2and.production_training_contract import load_model_plan  # noqa: E402
from scripts.production.model.plan_datasets import require_arrow_plan  # noqa: E402

DATASET_FILES = ("LICENSE.txt", "manifest.json", "replay_manifest.json", "bundle.json")
DATASET_OPTIONAL_FILES = ("orcid_error_query_signatures.json",)
DATASET_DIRS = ("benchmarks", "datasets", "labels", "components", "splits")
DATASET_OPTIONAL_DIRS = ("evaluation_inputs", "incremental_patent_tasks")
SOURCE_FILES = ("LICENSE.txt",)
SOURCE_OPTIONAL_FILES = ("excluded_query_views.json",)
SOURCE_DIRS = ("corrected_sources",)
SOURCE_OPTIONAL_DIRS = ("corrected_replay",)
UNKNOWN_PATTERN = "restored_unknown_signature_ids_*.json"
# Windows reports a cross-volume hardlink as ERROR_NOT_SAME_DEVICE.
_WINDOWS_NOT_SAME_DEVICE = 17


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _place(source: Path, target: Path) -> None:
    """Hardlink one immutable file, or copy it across volumes."""

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno != errno.EXDEV and getattr(error, "winerror", None) != _WINDOWS_NOT_SAME_DEVICE:
            raise
        shutil.copy2(source, target)


def _place_tree(source: Path, target: Path) -> int:
    count = 0
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix != ".lock":
            _place(path, target / path.relative_to(source))
            count += 1
    return count


def _arrow_signature_ids(dataset_dir: Path) -> set[str]:
    with ArrowDataset.open(dataset_dir) as arrow_dataset:
        with arrow_dataset.use() as lease, lease.open_file("signatures") as source:
            return set(load_signatures_from_arrow(source))


def _sidecar_ids(directory: Path) -> tuple[Path | None, set[str]]:
    sidecars = sorted(directory.glob(UNKNOWN_PATTERN))
    if len(sidecars) > 1:
        raise ValueError(f"Multiple unknown-signature lists in {directory}")
    if not sidecars:
        return None, set()
    return sidecars[0], set(_read_json(sidecars[0]))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _random_block_roles(
    name: str,
    model: Mapping[str, Path],
    evaluation: Mapping[str, Path],
    *,
    snapshot_benchmark: Path,
    benchmark: Path,
    pairwise: Path,
) -> dict[str, Path]:
    """Prove that the benchmark Arrow rebuilds the prepared populations, then declare them.

    Reads use the snapshot copy, whose manifest still resolves the shared count index.
    """

    model_ids = set(_read_json(model["signatures"]))
    evaluation_ids = set(_read_json(evaluation["signatures"]))
    blocks = _read_json(evaluation["blocks"])
    block_ids = {str(member) for members in blocks.values() for member in members}
    complete = _arrow_signature_ids(snapshot_benchmark)
    _require(not model_ids & evaluation_ids, f"{name}: development and outer-test signatures overlap")
    _require(block_ids == evaluation_ids, f"{name}: held-out blocks differ from the outer-test signatures")
    _require(model_ids | evaluation_ids <= complete, f"{name}: prepared signatures are missing from the benchmark")
    clusters = _read_json(snapshot_benchmark / f"{name}_clusters.json")
    selected: dict[str, Any] = {}
    for cluster_id, cluster in clusters.items():
        members = [member for member in cluster["signature_ids"] if str(member) in model_ids]
        if members:
            selected[cluster_id] = {**cluster, "signature_ids": members}
    _require(selected == _read_json(model["clusters"]), f"{name}: benchmark clusters do not rebuild the model labels")
    pair_ids = {
        str(row[key]) for row in _read_json(evaluation["pairs"]) for key in ("signature_id_1", "signature_id_2")
    }
    _require(pair_ids <= block_ids, f"{name}: held-out pairs use signatures outside the held-out blocks")
    sidecar, unknown = _sidecar_ids(snapshot_benchmark)
    sidecar = None if sidecar is None else benchmark / sidecar.name
    for population, files, ids in (("development", model, model_ids), ("outer-test", evaluation, evaluation_ids)):
        _, declared = _sidecar_ids(files["signatures"].parent)
        _require(declared == unknown & ids, f"{name}: the {population} unknown list is not the benchmark list")

    heldout = pairwise / "heldout" / name
    _place(evaluation["blocks"], heldout / "test_blocks.json")
    _place(evaluation["pairs"], heldout / "test_pairs.json")
    roles = {
        "arrow": benchmark / "manifest.json",
        "clusters": benchmark / f"{name}_clusters.json",
        "heldout_blocks": heldout / "test_blocks.json",
        "heldout_pairs": heldout / "test_pairs.json",
    }
    if sidecar is not None:
        roles["unknown_signatures"] = sidecar
    excluded = complete - model_ids - evaluation_ids
    if excluded:
        # Producers can reserve signatures outside both development and the outer test.
        roles["excluded_signatures"] = pairwise / "excluded" / f"{name}.json"
        roles["excluded_signatures"].parent.mkdir(parents=True, exist_ok=True)
        roles["excluded_signatures"].write_text(json.dumps(sorted(excluded)) + "\n", encoding="utf-8")
    return roles


def _fixed_pair_roles(
    name: str,
    model: Mapping[str, Path],
    evaluation: Mapping[str, Path],
    *,
    arrow_source: Path,
    dataset: Path,
    source: Path,
) -> dict[str, Path]:
    """Publish one Arrow copy with its fixed pairs, and its editable records as source."""

    target = dataset / "pairwise" / name
    for path in sorted(arrow_source.iterdir()):
        if path.is_file() and path.name != "manifest.json" and path.suffix != ".lock":
            _place(path, target / path.name)
    manifest = _read_json(arrow_source / "manifest.json")
    # The dataset moves, so only its relative path to the shared count index changes.
    index = Path(os.path.relpath(dataset / "name_counts_index", target)).as_posix()
    manifest["paths"]["name_counts_index"] = index
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _place(model["train_pairs"], target / "train_pairs.csv")
    _place(model["val_pairs"], target / "val_pairs.csv")
    _place(evaluation["pairs"], target / "test_pairs.json")
    for role in ("signatures", "papers", "specter_embeddings"):
        _place(model[role], source / name / model[role].name)
    return {
        "arrow": target / "manifest.json",
        "train_pairs": target / "train_pairs.csv",
        "val_pairs": target / "val_pairs.csv",
        "heldout_pairs": target / "test_pairs.json",
    }


def _resolve_inputs(snapshot: Path) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Path]]]]:
    pairwise = snapshot / "pairwise"
    inputs = _read_json(pairwise / "inputs.json")
    files = {
        mode: {
            name: {role: (pairwise / spec["path"]).resolve() for role, spec in roles.items()}
            for name, roles in sources.items()
        }
        for mode, sources in inputs["sources"].items()
    }
    return inputs, files


def write_release_bundles(snapshot: Path, output_root: Path, *, model: Path | None = None) -> dict[str, Any]:
    """Write ``inference``, ``dataset``, and ``source`` under ``output_root``.

    Args:
        snapshot: Prepared snapshot from ``package_corrected_datasets.py``.
        output_root: New directory for the three bundles.
        model: Optional complete model directory for the inference bundle.

    Returns:
        File counts, byte counts, and the dataset model-plan digest.

    Raises:
        FileExistsError: If ``output_root`` exists.
        ValueError: If the Arrow data does not rebuild the prepared populations.
    """

    snapshot = snapshot.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    old_plan = _read_json(snapshot / "pairwise" / "model_plan.json")
    inputs, files = _resolve_inputs(snapshot)
    inference, dataset, source = (output_root / name for name in ("inference", "dataset", "source"))

    _place_tree(snapshot / "name_counts_index", inference / "name_counts_index")
    _place(snapshot / "LICENSE.txt", inference / "LICENSE.txt")
    if model is not None:
        _place_tree(model.resolve(), inference / model.resolve().name)

    for name in DATASET_FILES:
        _place(snapshot / name, dataset / name)
    for name in DATASET_OPTIONAL_FILES:
        if (snapshot / name).is_file():
            _place(snapshot / name, dataset / name)
    for name in DATASET_DIRS:
        _place_tree(snapshot / name, dataset / name)
    for name in DATASET_OPTIONAL_DIRS:
        if (snapshot / name).is_dir():
            _place_tree(snapshot / name, dataset / name)
    for name in SOURCE_FILES:
        _place(snapshot / name, source / name)
    for name in SOURCE_OPTIONAL_FILES:
        if (snapshot / name).is_file():
            _place(snapshot / name, source / name)
    for name in SOURCE_DIRS:
        _place_tree(snapshot / name, source / name)
    for name in SOURCE_OPTIONAL_DIRS:
        if (snapshot / name).is_dir():
            _place_tree(snapshot / name, source / name)

    pairwise = dataset / "pairwise"
    plan_roles: dict[str, dict[str, Path]] = {}
    for name in old_plan["datasets"]:
        model_files, evaluation_files = files["model"][name], files["evaluation"][name]
        if "clusters" in model_files:
            plan_roles[name] = _random_block_roles(
                name,
                model_files,
                evaluation_files,
                snapshot_benchmark=snapshot / "benchmarks" / name,
                benchmark=dataset / "benchmarks" / name,
                pairwise=pairwise,
            )
        else:
            plan_roles[name] = _fixed_pair_roles(
                name,
                model_files,
                evaluation_files,
                arrow_source=(snapshot / "pairwise" / inputs["arrow"]["model"] / name).resolve(),
                dataset=dataset,
                source=source,
            )
    plan = {
        "release_version": old_plan["release_version"],
        "datasets": {
            name: {
                role: {"path": Path(os.path.relpath(path, pairwise)).as_posix(), "sha256": _sha256(path)}
                for role, path in roles.items()
            }
            for name, roles in plan_roles.items()
        },
        "eps": old_plan["eps"],
    }
    plan_path = pairwise / "model_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    require_arrow_plan(load_model_plan(plan_path).datasets)
    bundle_files = {
        bundle.name: [path for path in bundle.rglob("*") if path.is_file()] for bundle in (inference, dataset, source)
    }
    return {
        "output_root": str(output_root),
        "files": {name: len(paths) for name, paths in bundle_files.items()},
        "bytes": {name: sum(path.stat().st_size for path in paths) for name, paths in bundle_files.items()},
        "model_plan_sha256": _sha256(plan_path),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="Prepared snapshot root.")
    parser.add_argument("--output-root", type=Path, required=True, help="New directory for the three bundles.")
    parser.add_argument("--model", type=Path, help="Complete model directory for the inference bundle.")
    args = parser.parse_args(argv)
    print(json.dumps(write_release_bundles(args.snapshot, args.output_root, model=args.model), indent=2))


if __name__ == "__main__":
    main()
