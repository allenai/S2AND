"""Copy prepared pairwise sources and Arrow data into one portable directory."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and._sha256 import sha256_file  # noqa: E402
from s2and.arrow_inputs import ArrowDataset  # noqa: E402
from s2and.arrow_provenance import verify_conversion_provenance  # noqa: E402
from scripts.production.model.linker_source_bundle import (  # noqa: E402
    _dataset_manifests,
    _rebind_dataset_manifests,
)

BENCHMARKS = ("aminer", "arnetminer", "inspire", "kisti", "pubmed", "qian", "zbmath")


def package_pairwise_inputs(
    prepared_root: Path, output_root: Path, *, datasets: Sequence[str] = BENCHMARKS,
) -> dict[str, Any]:
    """Preserve source bytes and bind both Arrow collections to a shared index."""
    prepared = prepared_root.resolve()
    output = output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not datasets or len(set(datasets)) != len(datasets) or "augmented" in datasets:
        raise ValueError("Require unique benchmark dataset names without augmented")
    for name in datasets:
        if Path(name).name != name or name in {".", ".."} or ":" in name or "\\" in name:
            raise ValueError(f"Invalid dataset name: {name}")
    arrow_sources = {mode: prepared / "arrow" / folder for mode, folder in (
        ("model", "release"), ("evaluation", "evaluation"),
    )}
    expected_datasets = set(datasets) | {"augmented"}
    for source in arrow_sources.values():
        if set(_dataset_manifests(source)) != expected_datasets:
            raise ValueError(f"Arrow dataset population differs: {source}")
        if output.is_relative_to(source):
            raise ValueError("Output must remain outside source Arrow collections")
    output.mkdir(parents=True)
    shutil.copytree(
        prepared / "name_counts_global" / "name_counts_index", output / "name_counts_index",
    )
    shutil.copyfile(REPO_ROOT / "LICENSE-DATA", output / "LICENSE.txt")
    sources: dict[str, dict[str, dict[str, dict[str, str]]]] = {"model": {}, "evaluation": {}}

    def copy_file(source: Path, destination: Path) -> dict[str, str]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        original = sha256_file(source)
        if sha256_file(destination) != original:
            raise ValueError(f"Source copy differs: {destination}")
        return {"path": destination.relative_to(output).as_posix(), "sha256": original}

    for name in datasets:
        common: dict[str, dict[str, str]] = {}
        for role, filename in (("papers", f"{name}_papers.json"), ("specter_embeddings", f"{name}_specter2.pkl")):
            model = prepared / "populations" / "model_source" / name / filename
            evaluation = prepared / "populations" / "evaluation_source" / name / filename
            if sha256_file(model) != sha256_file(evaluation):
                raise ValueError(f"Model and evaluation {role} differ: {name}")
            common[role] = copy_file(model, output / "sources" / "common" / name / filename)
        for mode, folder in (("model", "model_source"), ("evaluation", "evaluation_source")):
            source = prepared / "populations" / folder / name
            destination = output / "sources" / mode / name
            files = dict(common)
            for role in ("signatures", "clusters"):
                filename = f"{name}_{role}.json"
                files[role] = copy_file(source / filename, destination / filename)
            unknown = sorted(source.glob("restored_unknown_signature_ids_*.json"))
            if len(unknown) != 1:
                raise ValueError(f"Require one uncertainty sidecar: {source}")
            files["unknown_signatures"] = copy_file(unknown[0], destination / unknown[0].name)
            if mode == "evaluation":
                for role, filename in (("pairs", "test_pairs.json"), ("blocks", "test_blocks.json")):
                    files[role] = copy_file(source / filename, destination / filename)
            sources[mode][name] = files

    augmented: dict[str, dict[str, str]] = {}
    for role, filename in (
        ("signatures", "augmented_signatures.json"), ("papers", "augmented_papers.json"),
        ("specter_embeddings", "augmented_specter2.pkl"), ("train_pairs", "train_pairs.csv"),
        ("val_pairs", "val_pairs.csv"), ("pairs", "test_pairs.json"),
    ):
        augmented[role] = copy_file(
            prepared / "augmented" / filename, output / "sources" / "augmented" / filename,
        )
    shared_roles = ("signatures", "papers", "specter_embeddings")
    sources["model"]["augmented"] = {role: augmented[role] for role in (*shared_roles, "train_pairs", "val_pairs")}
    sources["evaluation"]["augmented"] = {role: augmented[role] for role in (*shared_roles, "pairs")}
    collections = {}
    for mode, source in arrow_sources.items():
        destination = output / "arrow" / source.name
        shutil.copytree(source, destination)
        _rebind_dataset_manifests(destination, name_counts_index=output / "name_counts_index")
        for name in (*datasets, "augmented"):
            digests = {role: sources[mode][name][role]["sha256"] for role in shared_roles}
            verify_conversion_provenance(destination / name, expected_source_sha256=digests)
            with ArrowDataset.open(destination / name, require_specter=True, require_name_counts_index=True):
                pass
        collections[mode] = destination.relative_to(output).as_posix()
    result = {
        "release_version": "1.3", "sources": sources, "arrow": collections,
        "name_counts_index": "name_counts_index",
        "scope": "Pairwise data inputs; final production plans require reviewed evaluation gates",
    }
    (output / "inputs.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(package_pairwise_inputs(**vars(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
