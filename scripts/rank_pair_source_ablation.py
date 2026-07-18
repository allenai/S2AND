"""Rank a complete multi-seed pair-source ablation study.

The input manifest contract is documented in
``scripts._pair_ablation.ranking``. Policy constants are intentionally not
exposed as command-line flags.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._pair_ablation.ranking import rank_manifest, write_ranking_outputs  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-manifest",
        type=Path,
        required=True,
        help="Strict fold-expectation manifest; see scripts._pair_ablation.ranking.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    artifacts = rank_manifest(args.input_manifest)
    write_ranking_outputs(args.output_dir, artifacts)
    decision = artifacts.ranking["decision"]
    print(
        json.dumps(
            {
                "decision": decision["decision"],
                "output_dir": str(args.output_dir.resolve()),
                "provisional_arm": decision["provisional_arm"],
                "recommended_arm": decision["recommended_arm"],
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
