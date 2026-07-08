"""Training/replay data loaders for promoted incremental-linker artifacts."""

from __future__ import annotations

import json
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from s2and.data import ANDData
from s2and.incremental_linking_training.name_counts import LoadNameCountsMode, resolve_load_name_counts
from s2and.model import _ensure_lightgbm_fitted
from s2and.production_model import load_production_model
from s2and.thread_config import resolve_n_jobs


def _required_file(data_dir: Path, filename: str) -> Path:
    """Return a required file under ``data_dir``."""

    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_text_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in text if line.strip()]


def _resolve_target_block(signatures: dict[str, Any], meta: dict[str, Any] | None, block_key: str | None) -> str:
    if block_key:
        return str(block_key)
    if meta is not None:
        for key in ("target_block", "block_key"):
            value = meta.get(key)
            if value:
                return str(value)

    block_counts: Counter[str] = Counter()
    for signature in signatures.values():
        block = str(signature["author_info"].get("block", ""))
        if block:
            block_counts[block] += 1
    if not block_counts:
        raise RuntimeError("No signature blocks found in signatures.json")
    return block_counts.most_common(1)[0][0]


def _select_block_signature_ids(signatures: dict[str, Any], target_block: str) -> list[str]:
    selected = [
        str(signature_id)
        for signature_id, payload in signatures.items()
        if str(payload["author_info"].get("block", "")) == target_block
    ]
    if not selected:
        raise ValueError(f"Target block {target_block!r} did not match any signatures")
    return sorted(selected)


def _filter_papers(papers: dict[str, Any], signature_payloads: dict[str, Any]) -> dict[str, Any]:
    selected_paper_ids = {
        str(payload["paper_id"]) for payload in signature_payloads.values() if payload.get("paper_id") is not None
    }
    return {str(paper_id): payload for paper_id, payload in papers.items() if str(paper_id) in selected_paper_ids}


def _load_specter_subset(specter_path: Path, signature_payloads: dict[str, Any]) -> dict[str, Any]:
    with specter_path.open("rb") as handle:
        loaded = pickle.load(handle)

    if isinstance(loaded, dict):
        raw_embeddings = loaded
    elif isinstance(loaded, tuple) and len(loaded) == 2:
        matrix, keys = loaded
        raw_embeddings = {str(key): matrix[index, :] for index, key in enumerate(keys)}
    else:
        raise TypeError(f"Unsupported specter payload type: {type(loaded)!r}")

    selected_paper_ids = {
        str(payload["paper_id"]) for payload in signature_payloads.values() if payload.get("paper_id") is not None
    }
    return {key: raw_embeddings[key] for key in selected_paper_ids if key in raw_embeddings}


def _filter_cluster_seeds(
    cluster_seeds: dict[str, Any] | None,
    selected_signature_ids: set[str],
) -> dict[str, Any] | None:
    if cluster_seeds is None:
        return None

    filtered: dict[str, dict[str, Any]] = {}
    for root_signature_id, members in cluster_seeds.items():
        if str(root_signature_id) not in selected_signature_ids:
            continue
        filtered_members = {
            str(member_id): value for member_id, value in members.items() if str(member_id) in selected_signature_ids
        }
        if filtered_members:
            filtered[str(root_signature_id)] = filtered_members
    return filtered


def _filter_altered_signatures(
    altered_cluster_signatures: list[str] | None,
    selected_signature_ids: set[str],
) -> list[str] | None:
    if altered_cluster_signatures is None:
        return None
    return [
        str(signature_id) for signature_id in altered_cluster_signatures if str(signature_id) in selected_signature_ids
    ]


def load_giant_block_dataset(
    data_dir: Path,
    *,
    block_key: str | None,
    n_jobs: int,
    clusterer: Any | None = None,
    load_name_counts: LoadNameCountsMode = "auto",
) -> tuple[ANDData, dict[str, Any]]:
    """Load and filter a giant-block dataset from ``data_dir``."""

    signatures_path = _required_file(data_dir, "signatures.json")
    papers_path = _required_file(data_dir, "papers.json")
    specter_path = _required_file(data_dir, "specter.pickle")
    cluster_seeds_path = _required_file(data_dir, "cluster_seeds.json")
    altered_cluster_signatures_path = _required_file(data_dir, "altered_cluster_signatures.txt")
    meta_path = data_dir / "meta.json"

    signatures = _read_json(signatures_path)
    papers = _read_json(papers_path)
    cluster_seeds = _read_json(cluster_seeds_path)
    altered_cluster_signatures = _read_text_lines(altered_cluster_signatures_path)
    meta = _read_json(meta_path) if meta_path.exists() else None

    resolved_block_key = _resolve_target_block(signatures, meta, block_key)
    selected_signature_ids = _select_block_signature_ids(signatures, resolved_block_key)
    selected_signature_payloads = {signature_id: signatures[signature_id] for signature_id in selected_signature_ids}
    selected_paper_payloads = _filter_papers(papers, selected_signature_payloads)
    selected_specter_embeddings = _load_specter_subset(specter_path, selected_signature_payloads)
    filtered_cluster_seeds = _filter_cluster_seeds(cluster_seeds, set(selected_signature_ids))
    filtered_altered = _filter_altered_signatures(altered_cluster_signatures, set(selected_signature_ids))

    env_keys = ("S2AND_BACKEND", "OMP_NUM_THREADS", "RAYON_NUM_THREADS")
    previous_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["S2AND_BACKEND"] = "rust"
        thread_count = str(resolve_n_jobs(n_jobs))
        os.environ["OMP_NUM_THREADS"] = thread_count
        os.environ["RAYON_NUM_THREADS"] = thread_count

        dataset = ANDData(
            signatures=selected_signature_payloads,
            papers=selected_paper_payloads,
            name=f"{data_dir.name}_giant_block",
            mode="inference",
            specter_embeddings=selected_specter_embeddings,
            clusters=None,
            cluster_seeds=filtered_cluster_seeds,
            altered_cluster_signatures=filtered_altered,
            block_type="s2",
            train_pairs=None,
            val_pairs=None,
            test_pairs=None,
            train_pairs_size=1000,
            val_pairs_size=1000,
            test_pairs_size=1000,
            n_jobs=int(n_jobs),
            load_name_counts=resolve_load_name_counts(load_name_counts=load_name_counts, clusterer=clusterer),
            preprocess=True,
            random_seed=int(meta.get("random_seed", 0) if isinstance(meta, dict) else 0),
            name_tuples="filtered",
            use_orcid_id=False,
        )
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    load_info = {
        "target_block": resolved_block_key,
        "selected_signature_ids": selected_signature_ids,
        "selected_paper_ids": sorted(selected_paper_payloads.keys()),
        "source_meta": meta,
    }
    return dataset, load_info


def load_clusterer(model_path: Path, *, n_jobs: int) -> Any:
    """Load the production clusterer and prepare it for inference."""

    clusterer = load_production_model(model_path, require_incremental_linker=False)
    _ensure_lightgbm_fitted(clusterer.classifier)
    _ensure_lightgbm_fitted(clusterer.nameless_classifier)
    clusterer.use_cache = False
    clusterer.n_jobs = int(n_jobs)
    return clusterer


__all__ = ["load_clusterer", "load_giant_block_dataset"]
