"""Shared helpers for reusable EPS sweep scripts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from s2and._sha256 import sha256_file
from s2and.arrow_inputs import read_arrow_collection_root

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLD_ROOT = PROJECT_ROOT / "scratch" / "linking_eps_gold"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "scratch" / "linking_eps_sweeps_arrow"
# Diagnostic B3 curve that keeps one-record ORCID groups and their merge penalties.
SINGLETON_INCLUSIVE_CURVE = "eps_b3_with_singletons.tsv"
# How a sweep clustered its cached distances. Sweeps from before this value cut their own
# linkage trees and skipped later production steps such as name-conflict repair.
CLUSTERING_SOURCE = "Clusterer.predict_eps_grid_from_rust_featurizer"
# Bump when a distance rule changes within the same package version. Distance caches and
# sweep plans record it, and release EPS review rejects curves from another policy.
# 1: version field added; 2: CJK given-name veto (same-script, complete names only);
# 3: joined-initials reading of short given names; 4: joined-capitals middles read as
# initials, and the repair's reading step; 5: year-gap split after the repair; 6: the
# split also reads blocks without a name conflict, and a joined-capitals last middle word
# after other middle words reads as initials; 7: the reading step counts a speller in the
# reading mention's own cluster; 8: the year-gap split runs on blocks under 11 mentions;
# 9: the split treats 1601 and 1800 as undated and, in blocks under 11 mentions, keeps parts
# that share a first name; 10: the bridge counts the surname block, not the clustered unit,
# and a 1601 or 1800 stamp follows the part whose years span it; 11: the repair's
# three-cluster extension is removed (later rounds join the third cluster);
# 12: guarded ORCID joins within blocks and across Arrow subblocks.
DISTANCE_POLICY_VERSION = 13
# Release EPS needs curves that ran the production attachment passes.
RELEASE_ROUTE = "production"


def write_json(path: Path, payload: Any) -> None:
    """Write pretty JSON with a trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    # Reviewed EPS writes checksummed model files with this helper.
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha1_text(value: str) -> str:
    """Return a SHA1 digest for stable cache keys."""

    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def arrow_dataset_dir(arrow_root: Path, dataset: str, replay_bundle: str | None = None) -> Path:
    """Return a dataset directory declared by a validated Arrow root."""

    dataset_manifests, replay_bundles, _release_version = read_arrow_collection_root(arrow_root / "manifest.json")
    if replay_bundle is not None:
        if replay_bundle not in replay_bundles:
            raise ValueError(f"Arrow root does not declare replay bundle={replay_bundle!r}")
        dataset_manifests, _, _ = read_arrow_collection_root(replay_bundles[replay_bundle])
        replay_bundles = {}
    manifest_path = dataset_manifests.get(dataset)
    if manifest_path is None:
        matches: list[Path] = []
        for replay_manifest in replay_bundles.values():
            replay_datasets, _, _ = read_arrow_collection_root(replay_manifest)
            if dataset in replay_datasets:
                matches.append(replay_datasets[dataset])
        if len(matches) != 1:
            raise ValueError(f"Arrow root does not declare one unambiguous dataset={dataset!r}")
        manifest_path = matches[0]
    return manifest_path.parent


def read_sweep_curve(
    root: Path,
    dataset: str,
    score_mode: str,
    *,
    weight: str = "weight_query_class_balanced",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Bind each review curve to its completed plan and source model."""
    dataset_root = root / dataset
    plan_path = dataset_root / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    summary = json.loads((dataset_root / "summary.json").read_text(encoding="utf-8"))
    if summary["plan_sha256"] != sha256_file(plan_path) or plan["score_mode"] != score_mode:
        raise ValueError(f"The completed sweep does not match its plan: {dataset}")
    score_path = dataset_root / ("eps_b3.tsv" if score_mode == "orcid-b3" else "eps_overall.tsv")
    if summary["score_sha256"] != sha256_file(score_path):
        raise ValueError(f"The EPS scores change after completion: {dataset}")
    curve = pd.read_csv(score_path, sep="\t")
    if score_mode == "safe-action-proxy":
        curve = curve[
            curve["weight"].eq(weight) & curve["scope_type"].eq("overall") & curve["scope_value"].eq("all")
        ].copy()
    if curve["eps"].duplicated().any() or set(curve["eps"].round(10)) != {
        round(value, 10) for value in plan["eps_values"]
    }:
        raise ValueError(f"Incomplete EPS curve: {dataset}")
    metrics = ["precision", "recall", "f1"] if score_mode == "orcid-b3" else ["positive_recall", "negative_separation"]
    if not curve[metrics].map(lambda value: math.isfinite(value) and 0 <= value <= 1).all().all():
        raise ValueError(f"Invalid EPS metrics: {dataset}")
    return plan, curve


def read_singleton_inclusive_curve(root: Path, dataset: str, plan: Mapping[str, Any]) -> pd.DataFrame:
    """Read the completed diagnostic curve that a release review must also inspect."""
    dataset_root = root / dataset
    summary = json.loads((dataset_root / "summary.json").read_text(encoding="utf-8"))
    path = dataset_root / SINGLETON_INCLUSIVE_CURVE
    if summary.get("singleton_inclusive_score_sha256") != sha256_file(path):
        raise ValueError(f"The singleton-inclusive EPS scores change after completion: {dataset}")
    curve = pd.read_csv(path, sep="\t")
    if curve["eps"].duplicated().any() or set(curve["eps"].round(10)) != {
        round(value, 10) for value in plan["eps_values"]
    }:
        raise ValueError(f"Incomplete singleton-inclusive EPS curve: {dataset}")
    if not curve[["precision", "recall", "f1"]].map(lambda value: math.isfinite(value) and 0 <= value <= 1).all().all():
        raise ValueError(f"Invalid singleton-inclusive EPS metrics: {dataset}")
    return curve
