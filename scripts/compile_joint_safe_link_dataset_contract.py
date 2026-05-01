"""Compile the official safe-link dataset contract artifacts."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra_path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from joint_safe_link_dataset_contract import DEFAULT_BUNDLE_ROOT, write_contract_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    args = parser.parse_args()

    summary = write_contract_artifacts(args.bundle_root)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
