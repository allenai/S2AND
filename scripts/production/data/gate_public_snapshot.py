"""Gate a public dataset snapshot by S2AG paper availability, abstract rights, and email redaction."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and._sha256 import sha256_file  # noqa: E402
from scripts.production.data.hash_release_emails import hash_feature_email, scrub_text_emails  # noqa: E402


def load_paper_id_set(path: Path, *, column: str = "paper_id") -> frozenset[str]:
    """Load a set of paper IDs from a JSON list, newline text file, or Parquet column."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
        col = column if column in frame.columns else ("corpus_paper_id" if "corpus_paper_id" in frame.columns else None)
        if col is None:
            raise ValueError(f"Paper ID parquet {path} missing column {column!r}")
        return frozenset(str(val).strip() for val in frame[col].dropna().tolist() if str(val).strip())
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return frozenset(str(val).strip() for val in raw if val is not None and str(val).strip())
        if isinstance(raw, dict):
            return frozenset(str(val).strip() for val in raw if str(val).strip())
        raise ValueError(f"Expected JSON list or object of paper IDs in {path}")
    return frozenset(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def gate_public_dataset(
    *,
    dataset_name: str,
    signatures_json: Path,
    papers_json: Path,
    output_dir: Path,
    allowed_paper_ids: Collection[str],
    abstract_allowed_paper_ids: Collection[str],
    clusters_json: Path | None = None,
    scrub_emails: bool = True,
) -> dict[str, Any]:
    """Filter non-S2AG papers, redact unshareable abstracts, scrub emails, and write population_corrections.json.

    Runs once at intake before Arrow conversion or pair sampling so downstream artifacts
    never carry unshareable text or raw email addresses.
    """
    allowed_papers = {str(pid) for pid in allowed_paper_ids}
    abstract_allowed = {str(pid) for pid in abstract_allowed_paper_ids}
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    papers = json.loads(papers_json.read_text(encoding="utf-8"))
    signatures = json.loads(signatures_json.read_text(encoding="utf-8"))

    papers_before = len(papers)
    signatures_before = len(signatures)
    abstracts_present_before = 0
    abstracts_redacted = 0
    paper_email_fields_scrubbed = 0
    kept_papers: dict[str, Any] = {}
    dropped_paper_ids: list[str] = []

    for pid, raw_record in papers.items():
        pid_str = str(pid)
        if pid_str not in allowed_papers:
            dropped_paper_ids.append(pid_str)
            continue
        record = dict(raw_record)
        had_abstract = bool(record.get("abstract"))
        if had_abstract:
            abstracts_present_before += 1
        if pid_str not in abstract_allowed:
            if had_abstract:
                abstracts_redacted += 1
            record["abstract"] = None
        if scrub_emails:
            for field in ("title", "abstract"):
                val = record.get(field)
                if isinstance(val, str):
                    cleaned = scrub_text_emails(val)
                    if cleaned != val:
                        paper_email_fields_scrubbed += 1
                        record[field] = cleaned
        kept_papers[pid_str] = record

    kept_paper_set = set(kept_papers)
    kept_signatures: dict[str, Any] = {}
    dropped_signature_ids: list[str] = []
    signature_email_fields_redacted = 0

    for sid, raw_sig in signatures.items():
        sid_str = str(sid)
        sig_pid = str(raw_sig.get("paper_id", ""))
        if sig_pid not in kept_paper_set:
            dropped_signature_ids.append(sid_str)
            continue
        sig = dict(raw_sig)
        if scrub_emails and isinstance(sig.get("author_info"), dict):
            info = dict(sig["author_info"])
            for field in ("email", "emails"):
                if field in info:
                    before = info[field]
                    after = (
                        [hash_feature_email(v) for v in before]
                        if isinstance(before, list)
                        else hash_feature_email(before)
                    )
                    if after != before:
                        signature_email_fields_redacted += 1
                        info[field] = after
            for field in ("affiliations", "affiliation"):
                if field in info:
                    before = info[field]
                    if isinstance(before, list):
                        after = [scrub_text_emails(v) if isinstance(v, str) else v for v in before]
                    elif isinstance(before, str):
                        after = scrub_text_emails(before)
                    else:
                        continue
                    if after != before:
                        signature_email_fields_redacted += 1
                        info[field] = after
            sig["author_info"] = info
        kept_signatures[sid_str] = sig

    kept_sig_set = set(kept_signatures)
    out_papers = output_dir / f"{dataset_name}_papers.json"
    out_signatures = output_dir / f"{dataset_name}_signatures.json"
    _write_json(out_papers, kept_papers)
    _write_json(out_signatures, kept_signatures)

    outputs: dict[str, dict[str, Any]] = {
        "papers": {"path": out_papers.name, "sha256": sha256_file(out_papers)},
        "signatures": {"path": out_signatures.name, "sha256": sha256_file(out_signatures)},
    }
    clusters_before = 0
    clusters_after = 0
    if clusters_json is not None:
        raw_clusters = json.loads(clusters_json.read_text(encoding="utf-8"))
        clusters_before = len(raw_clusters)
        kept_clusters: dict[str, Any] = {}
        for cid, cluster in raw_clusters.items():
            members = [str(m) for m in cluster.get("signature_ids", []) if str(m) in kept_sig_set]
            if members:
                kept_clusters[str(cid)] = {**cluster, "signature_ids": members}
        clusters_after = len(kept_clusters)
        out_clusters = output_dir / f"{dataset_name}_clusters.json"
        _write_json(out_clusters, kept_clusters)
        outputs["clusters"] = {"path": out_clusters.name, "sha256": sha256_file(out_clusters)}

    report: dict[str, Any] = {
        "dataset": dataset_name,
        "papers_before": papers_before,
        "papers_after": len(kept_papers),
        "papers_dropped_non_s2ag": len(dropped_paper_ids),
        "signatures_before": signatures_before,
        "signatures_after": len(kept_signatures),
        "signatures_dropped_non_s2ag": len(dropped_signature_ids),
        "clusters_before": clusters_before,
        "clusters_after": clusters_after,
        "abstracts_present_before": abstracts_present_before,
        "abstracts_redacted": abstracts_redacted,
        "scrub_emails": bool(scrub_emails),
        "paper_email_fields_scrubbed": paper_email_fields_scrubbed,
        "signature_email_fields_redacted": signature_email_fields_redacted,
        "dropped_paper_ids_sample": sorted(dropped_paper_ids)[:20],
        "dropped_signature_ids_sample": sorted(dropped_signature_ids)[:20],
        "outputs": outputs,
    }
    _write_json(output_dir / "population_corrections.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--signatures-json", type=Path, required=True)
    parser.add_argument("--papers-json", type=Path, required=True)
    parser.add_argument("--clusters-json", type=Path, default=None)
    parser.add_argument("--allowed-papers", type=Path, required=True, help="JSON/parquet/txt of public S2AG paper IDs.")
    parser.add_argument(
        "--abstract-allowed-papers",
        type=Path,
        required=True,
        help="JSON/parquet/txt of paper IDs permitted to retain public abstracts.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-scrub-emails", action="store_true")
    args = parser.parse_args(argv)
    report = gate_public_dataset(
        dataset_name=args.dataset,
        signatures_json=args.signatures_json,
        papers_json=args.papers_json,
        clusters_json=args.clusters_json,
        output_dir=args.output_dir,
        allowed_paper_ids=load_paper_id_set(args.allowed_papers),
        abstract_allowed_paper_ids=load_paper_id_set(args.abstract_allowed_papers),
        scrub_emails=not args.no_scrub_emails,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
