"""Hash public replay emails without changing their pairwise feature evidence.

Also supports intake-once redaction of structured email fields and free-text
email mentions (``affiliations``, ``title``, ``abstract``) across JSON and Arrow
trees, plus plaintext-email scanning of staged release bundles.

The library code lives in `scripts.common.email_redaction`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.email_redaction import redact_source_tree, scan_tree_for_plaintext_emails  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    redact_parser = subparsers.add_parser("redact-source-tree", help="Redact emails once at intake in a source tree.")
    redact_parser.add_argument("--source-root", type=Path, required=True)
    redact_parser.add_argument("--keep-free-text", action="store_true", help="Only hash structured email fields.")

    scan_parser = subparsers.add_parser("scan", help="Scan a directory tree for plaintext emails.")
    scan_parser.add_argument("--root", type=Path, required=True)
    scan_parser.add_argument("--max-findings", type=int, default=20)

    args = parser.parse_args(argv)
    if args.command == "redact-source-tree":
        print(json.dumps(redact_source_tree(args.source_root, scrub_free_text=not args.keep_free_text), indent=2))
        return 0
    findings = scan_tree_for_plaintext_emails(args.root, max_findings=args.max_findings)
    print(json.dumps({"findings": findings, "count": len(findings)}, indent=2))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
