"""Combine per-run strict fold manifests for the final ablation ranker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._pair_ablation.ranking import (  # noqa: E402
    RANKING_INPUT_FOLD_ENTRY_KEYS,
    RANKING_INPUT_SCHEMA_VERSION,
)
from scripts._pair_ablation.results import load_strict_json  # noqa: E402


def _requested_arm_set(arms: list[str] | None) -> frozenset[str] | None:
    if arms is None:
        return None
    if not arms:
        raise ValueError("Requested arms must not be empty")
    if any(not isinstance(arm, str) or not arm.strip() for arm in arms):
        raise ValueError("Requested arms must be non-blank strings")
    if len(arms) != len(set(arms)):
        raise ValueError("Requested arms must be unique")
    return frozenset(arms)


def combine_ranking_inputs(
    paths: list[Path],
    *,
    arms: list[str] | None = None,
) -> dict[str, Any]:
    """Combine exact per-run manifests, optionally retaining named arms only."""

    if not paths:
        raise ValueError("At least one ranking input is required")
    requested_arms = _requested_arm_set(arms)
    folds: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for raw_path in paths:
        path = raw_path.resolve()
        payload = load_strict_json(path)
        if set(payload) != {"schema_version", "folds"}:
            raise ValueError(f"Ranking input has an invalid schema: {path}")
        if payload["schema_version"] != RANKING_INPUT_SCHEMA_VERSION:
            raise ValueError(f"Ranking input has an unsupported version: {path}")
        entries = payload["folds"]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Ranking input has no folds: {path}")
        manifest_entries: list[tuple[str, dict[str, Any]]] = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != RANKING_INPUT_FOLD_ENTRY_KEYS:
                raise ValueError(f"Ranking input contains a malformed fold entry: {path}")
            for path_key in ("path", "run_manifest_path"):
                if not isinstance(entry[path_key], str) or not entry[path_key].strip():
                    raise ValueError(f"Ranking input contains a malformed {path_key}: {path}")
            for digest_key in ("result_sha256", "run_manifest_sha256"):
                digest = entry[digest_key]
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise ValueError(f"Ranking input contains a malformed {digest_key}: {path}")
            expected = entry.get("expected")
            expected_arm = expected.get("arm") if isinstance(expected, dict) else None
            if not isinstance(expected_arm, str) or not expected_arm.strip():
                raise ValueError(f"Ranking input contains a malformed expected.arm: {path}")
            normalized_entry = dict(entry)
            for path_key in ("path", "run_manifest_path"):
                raw_entry_path = Path(entry[path_key])
                resolved = (
                    raw_entry_path.resolve()
                    if raw_entry_path.is_absolute()
                    else (path.parent / raw_entry_path).resolve()
                )
                normalized_entry[path_key] = str(resolved)
            manifest_entries.append((expected_arm, normalized_entry))

        if requested_arms is not None:
            manifest_arms = {arm for arm, _entry in manifest_entries}
            missing = sorted(requested_arms - manifest_arms)
            if missing:
                raise ValueError(f"Ranking input is missing requested arms {missing}: {path}")

        for expected_arm, entry in manifest_entries:
            if requested_arms is not None and expected_arm not in requested_arms:
                continue
            result_path = entry["path"]
            if result_path in seen_paths:
                raise ValueError(f"Duplicate fold result across ranking inputs: {result_path}")
            seen_paths.add(result_path)
            folds.append(entry)
    return {"schema_version": RANKING_INPUT_SCHEMA_VERSION, "folds": folds}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", help="Retain only these arms; each input must contain every arm")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = combine_ranking_inputs(args.inputs, arms=args.arms)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"folds": len(payload["folds"]), "output": str(output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
