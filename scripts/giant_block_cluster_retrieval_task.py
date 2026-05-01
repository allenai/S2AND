"""Persist giant-block subblocking and multi-letter clustering artifacts.

This runner is intentionally narrow:

1. Load a single giant extracted block from a data directory.
2. Build subblocks once with telemetry.
3. Classify the resulting subblocks by the repo's current first-name semantics.
4. Run `predict_helper` only on the multi-letter subblocks.
5. Persist the resulting manifests, telemetry, and predicted clusters.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Any

from s2and.consts import PROJECT_ROOT_PATH
from s2and.data import ANDData
from s2and.model import _ensure_lightgbm_fitted, _signature_first_for_rules
from s2and.serialization import load_pickle_with_verified_label_encoder_compat
from s2and.subblocking import make_subblocks_with_telemetry

try:
    from scripts.name_count_loading import LoadNameCountsMode, resolve_load_name_counts
except ImportError:  # pragma: no cover - direct script execution path
    from name_count_loading import LoadNameCountsMode, resolve_load_name_counts  # type: ignore

DEFAULT_MODEL_PATH = Path(PROJECT_ROOT_PATH) / "data" / "production_model_v1.2.pickle"
DEFAULT_MAXIMUM_SIZE = 15_000
DEFAULT_TOTAL_RAM_BYTES = 128 * (1 << 30)


def _required_file(data_dir: Path, filename: str) -> Path:
    """Return a required file under `data_dir`."""

    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def _read_json(path: Path) -> Any:
    """Read JSON from `path`."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_text_lines(path: Path) -> list[str]:
    """Read a newline-delimited text file and drop blank lines."""

    text = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in text if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to `path` with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _partial_subblock_dir(output_dir: Path) -> Path:
    """Return the directory that stores per-subblock partial clustering outputs."""

    return output_dir / "partial_multi_letter_subblocks"


def _build_partial_resume_identity(
    *,
    data_dir: Path,
    model_path: Path,
    target_block: str,
    maximum_size: int,
    total_ram_bytes: int | None,
) -> dict[str, Any]:
    """Build the strict identity used to validate resumed partial subblocks."""

    return {
        "data_dir": str(Path(data_dir).resolve()),
        "model_path": str(Path(model_path).resolve()),
        "target_block": str(target_block),
        "maximum_size": int(maximum_size),
        "total_ram_bytes": int(total_ram_bytes) if total_ram_bytes is not None else None,
    }


def _load_partial_multi_letter_state(
    output_dir: Path,
    *,
    expected_multi_letter_subblocks: dict[str, list[str]],
    resume_identity: dict[str, Any],
) -> tuple[dict[str, dict[str, list[str]]], list[dict[str, Any]], dict[str, str]]:
    """Load any previously persisted per-subblock clustering outputs."""

    partial_dir = _partial_subblock_dir(output_dir)
    predicted_clusters_by_subblock: dict[str, dict[str, list[str]]] = {}
    per_subblock_timings: list[dict[str, Any]] = []
    signature_to_cluster_id: dict[str, str] = {}
    if not partial_dir.exists():
        return predicted_clusters_by_subblock, per_subblock_timings, signature_to_cluster_id

    for partial_path in sorted(partial_dir.glob("subblock_*.json")):
        payload = _read_json(partial_path)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid partial subblock payload at {partial_path}: expected object")
        subblock_key = str(payload["subblock_key"])
        if subblock_key in predicted_clusters_by_subblock:
            raise RuntimeError(f"Duplicate partial subblock payload for {subblock_key!r} at {partial_path}")
        expected_signature_ids = expected_multi_letter_subblocks.get(subblock_key)
        if expected_signature_ids is None:
            raise RuntimeError(
                f"Found stale partial subblock {subblock_key!r} at {partial_path}; "
                "clear partial outputs before resuming with a different subblock manifest."
            )
        payload_identity = payload.get("resume_identity")
        if payload_identity != resume_identity:
            raise RuntimeError(
                f"Partial subblock {subblock_key!r} at {partial_path} does not match the current run identity; "
                "clear partial outputs before resuming."
            )
        clusters_payload = {
            str(cluster_id): [str(signature_id) for signature_id in members]
            for cluster_id, members in dict(payload["clusters"]).items()
        }
        partial_signature_ids = sorted(
            str(signature_id) for members in clusters_payload.values() for signature_id in members
        )
        if partial_signature_ids != sorted(str(signature_id) for signature_id in expected_signature_ids):
            raise RuntimeError(
                f"Partial subblock {subblock_key!r} at {partial_path} has signature membership that does not match "
                "the current subblock definition; clear partial outputs before resuming."
            )
        timing_payload = payload.get("timing")
        if not isinstance(timing_payload, dict):
            raise RuntimeError(f"Invalid timing payload for partial subblock {subblock_key!r} at {partial_path}")
        predicted_clusters_by_subblock[subblock_key] = clusters_payload
        per_subblock_timings.append(dict(timing_payload))
        for cluster_id, members in clusters_payload.items():
            for signature_id in members:
                signature_to_cluster_id[str(signature_id)] = str(cluster_id)

    per_subblock_timings.sort(key=lambda row: str(row["subblock_key"]))
    return predicted_clusters_by_subblock, per_subblock_timings, signature_to_cluster_id


def _resolve_target_block(signatures: dict[str, Any], meta: dict[str, Any] | None, block_key: str | None) -> str:
    """Resolve the block key to process."""

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
    """Return the signature IDs that belong to `target_block`."""

    selected = [
        str(signature_id)
        for signature_id, payload in signatures.items()
        if str(payload["author_info"].get("block", "")) == target_block
    ]
    if not selected:
        raise ValueError(f"Target block {target_block!r} did not match any signatures")
    return sorted(selected)


def _filter_papers(papers: dict[str, Any], signature_payloads: dict[str, Any]) -> dict[str, Any]:
    """Filter `papers` down to those referenced by `signature_payloads`."""

    selected_paper_ids = {
        str(payload["paper_id"]) for payload in signature_payloads.values() if payload.get("paper_id") is not None
    }
    return {str(paper_id): payload for paper_id, payload in papers.items() if str(paper_id) in selected_paper_ids}


def _load_specter_subset(specter_path: Path, signature_payloads: dict[str, Any]) -> dict[str, Any]:
    """Load and filter the specter embedding payload."""

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
    """Filter cluster-seed mappings to the selected signatures."""

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
    """Filter altered-cluster signatures to the selected signatures."""

    if altered_cluster_signatures is None:
        return None
    filtered = [
        str(signature_id) for signature_id in altered_cluster_signatures if str(signature_id) in selected_signature_ids
    ]
    return filtered


def load_dataset(
    data_dir: Path,
    *,
    block_key: str | None,
    n_jobs: int,
    clusterer: Any | None = None,
    load_name_counts: LoadNameCountsMode = "auto",
) -> tuple[ANDData, dict[str, Any]]:
    """Load and filter the giant-block dataset from `data_dir`."""

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

    os.environ.setdefault("S2AND_SKIP_FASTTEXT", "1")
    os.environ["S2AND_BACKEND"] = "rust"
    os.environ["OMP_NUM_THREADS"] = str(max(1, int(n_jobs)))
    os.environ["RAYON_NUM_THREADS"] = str(max(1, int(n_jobs)))
    resolved_load_name_counts = resolve_load_name_counts(load_name_counts=load_name_counts, clusterer=clusterer)

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
        load_name_counts=resolved_load_name_counts,
        preprocess=True,
        random_seed=int(meta.get("random_seed", 0) if isinstance(meta, dict) else 0),
        name_tuples="filtered",
        use_orcid_id=False,
        use_sinonym_overwrite=False,
        compute_reference_features=False,
    )

    load_info = {
        "target_block": resolved_block_key,
        "selected_signature_ids": selected_signature_ids,
        "selected_paper_ids": sorted(selected_paper_payloads.keys()),
        "source_meta": meta,
    }
    return dataset, load_info


def load_clusterer(model_path: Path, *, n_jobs: int) -> Any:
    """Load the production clusterer and prepare it for inference."""

    model_artifact = load_pickle_with_verified_label_encoder_compat(str(model_path))
    clusterer = model_artifact["clusterer"]
    _ensure_lightgbm_fitted(clusterer.classifier)
    _ensure_lightgbm_fitted(clusterer.nameless_classifier)
    clusterer.use_cache = False
    clusterer.n_jobs = int(n_jobs)
    return clusterer


def _classify_subblocks(
    subblocks: dict[str, list[str]],
    dataset: Any,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Split subblocks using the current repo semantics for first-name length."""

    multi_letter: dict[str, list[str]] = {}
    single_letter: dict[str, list[str]] = {}
    for subblock_key in sorted(subblocks):
        signature_ids = list(subblocks[subblock_key])
        full_first_signature_ids: list[str] = []
        initial_or_empty_signature_ids: list[str] = []
        for signature_id in signature_ids:
            signature = dataset.signatures[signature_id]
            if len(_signature_first_for_rules(signature)) <= 1:
                initial_or_empty_signature_ids.append(signature_id)
            else:
                full_first_signature_ids.append(signature_id)
        if full_first_signature_ids and initial_or_empty_signature_ids:
            multi_letter[f"{subblock_key}::multi_letter"] = full_first_signature_ids
            single_letter[f"{subblock_key}::single_letter"] = initial_or_empty_signature_ids
        elif initial_or_empty_signature_ids:
            single_letter[subblock_key] = signature_ids
        else:
            multi_letter[subblock_key] = signature_ids
    return multi_letter, single_letter


def _cluster_multi_letter_subblocks(
    *,
    clusterer: Any,
    dataset: ANDData,
    multi_letter_subblocks: dict[str, list[str]],
    output_dir: Path,
    total_ram_bytes: int | None,
    resume_identity: dict[str, Any],
) -> tuple[dict[str, dict[str, list[str]]], list[dict[str, Any]], dict[str, str]]:
    predicted_clusters_by_subblock, per_subblock_timings, signature_to_cluster_id = _load_partial_multi_letter_state(
        output_dir,
        expected_multi_letter_subblocks=multi_letter_subblocks,
        resume_identity=resume_identity,
    )
    partial_dir = _partial_subblock_dir(output_dir)
    partial_dir.mkdir(parents=True, exist_ok=True)
    timings_by_subblock = {str(row["subblock_key"]): dict(row) for row in per_subblock_timings}

    for subblock_index, subblock_key in enumerate(sorted(multi_letter_subblocks)):
        if subblock_key in predicted_clusters_by_subblock:
            continue
        signature_ids = list(multi_letter_subblocks[subblock_key])
        start = time.perf_counter()
        predicted_clusters, _ = clusterer.predict_helper(
            {subblock_key: signature_ids},
            dataset,
            dists=None,
            cluster_model_params=None,
            partial_supervision={},
            use_s2_clusters=False,
            incremental_dont_use_cluster_seeds=False,
            total_ram_bytes=total_ram_bytes,
        )
        elapsed = time.perf_counter() - start
        if not isinstance(predicted_clusters, dict):
            raise RuntimeError(
                f"predict_helper returned invalid clusters for {subblock_key!r}: {type(predicted_clusters).__name__}"
            )
        predicted_clusters_by_subblock[subblock_key] = {
            str(cluster_id): list(members) for cluster_id, members in predicted_clusters.items()
        }
        for cluster_id, members in predicted_clusters_by_subblock[subblock_key].items():
            for signature_id in members:
                signature_to_cluster_id[str(signature_id)] = str(cluster_id)
        timing_row = {
            "subblock_key": subblock_key,
            "signature_count": len(signature_ids),
            "cluster_count": len(predicted_clusters_by_subblock[subblock_key]),
            "cluster_sizes": sorted(
                (len(members) for members in predicted_clusters_by_subblock[subblock_key].values()),
                reverse=True,
            ),
            "predict_seconds": round(elapsed, 6),
        }
        timings_by_subblock[subblock_key] = timing_row
        _write_json(
            partial_dir / f"subblock_{subblock_index:03d}.json",
            {
                "subblock_key": subblock_key,
                "signature_ids": list(signature_ids),
                "resume_identity": dict(resume_identity),
                "clusters": predicted_clusters_by_subblock[subblock_key],
                "timing": timing_row,
            },
        )
        _write_json(
            output_dir / "multi_letter_subblock_timings.json",
            {"rows": [timings_by_subblock[key] for key in sorted(timings_by_subblock)]},
        )

    per_subblock_timings = [timings_by_subblock[key] for key in sorted(timings_by_subblock)]
    return predicted_clusters_by_subblock, per_subblock_timings, signature_to_cluster_id


def run_task(
    *,
    data_dir: Path,
    output_dir: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    block_key: str | None = None,
    maximum_size: int = DEFAULT_MAXIMUM_SIZE,
    n_jobs: int = 20,
    total_ram_bytes: int | None = DEFAULT_TOTAL_RAM_BYTES,
) -> dict[str, Any]:
    """Run the giant-block artifact build and persist the outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)

    model_start = time.perf_counter()
    clusterer = load_clusterer(Path(model_path), n_jobs=int(n_jobs))
    model_load_seconds = time.perf_counter() - model_start
    dataset_start = time.perf_counter()
    dataset, load_info = load_dataset(Path(data_dir), block_key=block_key, n_jobs=int(n_jobs), clusterer=clusterer)
    dataset_load_seconds = time.perf_counter() - dataset_start

    signature_ids = list(load_info["selected_signature_ids"])
    subblock_start = time.perf_counter()
    subblocks, subblocking_telemetry = make_subblocks_with_telemetry(
        signature_ids,
        dataset,
        maximum_size=int(maximum_size),
    )
    subblocking_seconds = time.perf_counter() - subblock_start

    multi_letter_subblocks, single_letter_subblocks = _classify_subblocks(subblocks, dataset)
    partial_resume_identity = _build_partial_resume_identity(
        data_dir=Path(data_dir),
        model_path=Path(model_path),
        target_block=str(load_info["target_block"]),
        maximum_size=int(maximum_size),
        total_ram_bytes=total_ram_bytes,
    )

    manifest_payload = {
        "data_dir": str(Path(data_dir).resolve()),
        "target_block": load_info["target_block"],
        "maximum_size": int(maximum_size),
        "total_ram_bytes": int(total_ram_bytes) if total_ram_bytes is not None else None,
        "signature_count": len(signature_ids),
        "subblock_count": len(subblocks),
        "multi_letter_subblock_count": len(multi_letter_subblocks),
        "single_letter_subblock_count": len(single_letter_subblocks),
        "subblocks": {key: list(value) for key, value in sorted(subblocks.items())},
        "multi_letter_subblock_keys": sorted(multi_letter_subblocks),
        "single_letter_subblock_keys": sorted(single_letter_subblocks),
    }
    telemetry_payload = {
        "load_dataset_seconds": round(dataset_load_seconds, 6),
        "load_model_seconds": round(model_load_seconds, 6),
        "make_subblocks_seconds": round(subblocking_seconds, 6),
        "predict_seconds": None,
        "make_subblocks_telemetry": subblocking_telemetry,
    }
    _write_json(output_dir / "subblock_manifest.json", manifest_payload)
    _write_json(output_dir / "subblock_telemetry.json", telemetry_payload)

    predict_start = time.perf_counter()
    predicted_clusters_by_subblock, per_subblock_timings, signature_to_cluster_id = _cluster_multi_letter_subblocks(
        clusterer=clusterer,
        dataset=dataset,
        multi_letter_subblocks=multi_letter_subblocks,
        output_dir=output_dir,
        total_ram_bytes=total_ram_bytes,
        resume_identity=partial_resume_identity,
    )
    predict_seconds = time.perf_counter() - predict_start

    summary_payload = {
        "data_dir": str(Path(data_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "model_path": str(Path(model_path).resolve()),
        "target_block": load_info["target_block"],
        "total_ram_bytes": int(total_ram_bytes) if total_ram_bytes is not None else None,
        "signature_count": len(signature_ids),
        "subblock_count": len(subblocks),
        "multi_letter_subblock_count": len(multi_letter_subblocks),
        "single_letter_subblock_count": len(single_letter_subblocks),
        "clustered_subblock_count": len(predicted_clusters_by_subblock),
        "predicted_cluster_total": sum(len(clusters) for clusters in predicted_clusters_by_subblock.values()),
        "signature_to_cluster_id_count": len(signature_to_cluster_id),
        "dataset_load_seconds": round(dataset_load_seconds, 6),
        "model_load_seconds": round(model_load_seconds, 6),
        "make_subblocks_seconds": round(subblocking_seconds, 6),
        "predict_seconds": round(predict_seconds, 6),
        "artifact_paths": {
            "subblock_manifest": str(output_dir / "subblock_manifest.json"),
            "subblock_telemetry": str(output_dir / "subblock_telemetry.json"),
            "predicted_clusters": str(output_dir / "predicted_clusters.json"),
            "signature_to_cluster_id": str(output_dir / "signature_to_cluster_id.json"),
            "multi_letter_subblock_timings": str(output_dir / "multi_letter_subblock_timings.json"),
            "partial_multi_letter_subblocks": str(_partial_subblock_dir(output_dir)),
            "run_summary": str(output_dir / "run_summary.json"),
        },
    }

    telemetry_payload["predict_seconds"] = round(predict_seconds, 6)
    _write_json(output_dir / "subblock_telemetry.json", telemetry_payload)
    _write_json(output_dir / "predicted_clusters.json", predicted_clusters_by_subblock)
    _write_json(output_dir / "signature_to_cluster_id.json", signature_to_cluster_id)
    _write_json(output_dir / "multi_letter_subblock_timings.json", {"rows": per_subblock_timings})
    _write_json(output_dir / "run_summary.json", summary_payload)

    return summary_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing extracted block files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for persisted artifacts.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--block-key", type=str, default=None, help="Optional target block override.")
    parser.add_argument("--maximum-size", type=int, default=DEFAULT_MAXIMUM_SIZE)
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--total-ram-bytes", type=int, default=DEFAULT_TOTAL_RAM_BYTES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    summary = run_task(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        block_key=args.block_key,
        maximum_size=int(args.maximum_size),
        n_jobs=int(args.n_jobs),
        total_ram_bytes=int(args.total_ram_bytes) if args.total_ram_bytes is not None else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
