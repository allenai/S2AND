"""Run the active official classic pipeline."""

from __future__ import annotations

# ruff: noqa: E402
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra_path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from joint_safe_link_official_stack import (
    compare_to_expected,
    format_classic_selected_gate_tables,
    load_bundle,
    run_classic,
)

BUNDLE_ROOT = REPO_ROOT / "data" / "joint_safe_link_official_stack_20260428p"
OUTPUT_DIR = REPO_ROOT / "scratch" / "joint_safe_link_official_classic_20260428p"


def run_active_classic(
    *,
    bundle_root: Path = BUNDLE_ROOT,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, object]:
    """Run classic train/calibration/test for one bundle and return the verification payload."""

    bundle_root = bundle_root.resolve()
    output_dir = output_dir.resolve()
    bundle = load_bundle(bundle_root)
    summary = run_classic(bundle, output_dir)
    verification = {
        "summary": summary,
        "expected": bundle.expected_metrics["classic"],
        "deltas": compare_to_expected(summary, bundle.expected_metrics["classic"]),
    }
    (output_dir / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    return verification


def main() -> None:
    verification = run_active_classic()
    print(json.dumps(verification, indent=2))
    tables = format_classic_selected_gate_tables(dict(verification["summary"]))
    if tables:
        print()
        print(tables)


if __name__ == "__main__":
    main()
