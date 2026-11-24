"""Gate a public dataset snapshot by S2AG paper availability, abstract rights, and email redaction.

Implements the public data contract in ``docs/data.md``:
- Filters papers to production S2AG papers and removes dropped papers/signatures from
  ``signatures.json``, optional ``clusters.json``, ``specter2.pkl``, unknown-signature
  sidecars, and pair files.
- Replaces present restricted abstracts with the ``"Has Abstract"`` marker (preserving the
  ``has_abstract`` pairwise/incremental feature), and applies the same rule recursively to
  abstract correction-history fields.
- Hashes structured email fields and scrubs plaintext emails from free text at intake.
- Streams large JSON inputs and writes atomically into a fresh output directory so
  hardlinked upstream snapshots can never be mutated in place.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import tempfile
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

import ijson
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and._sha256 import sha256_file  # noqa: E402
from scripts.production.data.hash_release_emails import hash_feature_email, scrub_text_emails  # noqa: E402

REDACTED_ABSTRACT_MARKER = "Has Abstract"


def _normalize_paper_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or not value.is_integer():
            raise ValueError(f"Non-integer float paper_id: {value!r}")
        return str(int(value))
    text = str(value).strip()
    return text or None


def load_paper_id_set(path: Path, *, column: str = "paper_id") -> frozenset[str]:
    """Load a set of permitted paper IDs from a JSON list/bool-map, newline text file, or Parquet column.

    For JSON objects, values MUST be booleans and only keys mapped to ``True`` are admitted
    (preventing ``{"allowed": true, "restricted": false}`` from admitting ``"restricted"``).
    """
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        schema = pq.read_schema(path)
        col = column if column in schema.names else ("corpus_paper_id" if "corpus_paper_id" in schema.names else None)
        if col is None:
            raise ValueError(f"Paper ID parquet {path} missing column {column!r}")
        table = pq.read_table(path, columns=[col])
        ids: set[str] = set()
        for val in table.column(0).to_pylist():
            norm = _normalize_paper_id(val)
            if norm is not None:
                ids.add(norm)
        return frozenset(ids)
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return frozenset(norm for val in raw if (norm := _normalize_paper_id(val)) is not None)
        if isinstance(raw, dict):
            non_bool = [k for k, v in raw.items() if not isinstance(v, bool)]
            if non_bool:
                raise ValueError(
                    f"Permission map JSON {path} must map paper IDs to booleans; "
                    f"non-bool values for keys: {non_bool[:5]}"
                )
            return frozenset(
                norm for k, allowed in raw.items() if allowed is True and (norm := _normalize_paper_id(k)) is not None
            )
        raise ValueError(f"Expected JSON list or boolean permission map in {path}")
    return frozenset(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _atomic_write_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False)
    tmp = Path(handle.name)
    handle.close()
    try:
        if compact:
            text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        else:
            text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _redact_abstract_value(value: Any) -> tuple[Any, bool]:
    """Replace a present abstract string with ``REDACTED_ABSTRACT_MARKER``; preserve empty/None."""
    if isinstance(value, str) and value.strip():
        if value == REDACTED_ABSTRACT_MARKER:
            return value, False
        return REDACTED_ABSTRACT_MARKER, True
    return value, False


def _redact_abstract_history(obj: Any) -> tuple[Any, int]:
    """Recursively apply the abstract-sharing gate to correction-history fields in a paper record."""
    redacted_count = 0
    if isinstance(obj, dict):
        field_name = str(obj.get("field", "")).lower()
        is_abstract_entry = field_name == "abstract"
        out: dict[str, Any] = {}
        for key, value in obj.items():
            key_lower = str(key).lower()
            if "abstract" in key_lower:
                if isinstance(value, str):
                    new_val, changed = _redact_abstract_value(value)
                    out[key] = new_val
                    redacted_count += int(changed)
                elif isinstance(value, dict | list):
                    nested, cnt = _redact_abstract_history(value)
                    out[key] = nested
                    redacted_count += cnt
                else:
                    out[key] = value
            elif is_abstract_entry and key_lower in {
                "old",
                "new",
                "before",
                "after",
                "old_value",
                "new_value",
                "value",
                "previous",
            }:
                new_val, changed = _redact_abstract_value(value)
                out[key] = new_val
                redacted_count += int(changed)
            elif isinstance(value, dict | list):
                nested, cnt = _redact_abstract_history(value)
                out[key] = nested
                redacted_count += cnt
            else:
                out[key] = value
        return out, redacted_count
    if isinstance(obj, list):
        out_list: list[Any] = []
        for item in obj:
            if isinstance(item, str):
                new_val, changed = _redact_abstract_value(item)
                out_list.append(new_val)
                redacted_count += int(changed)
            elif isinstance(item, dict | list):
                nested, cnt = _redact_abstract_history(item)
                out_list.append(nested)
                redacted_count += cnt
            else:
                out_list.append(item)
        return out_list, redacted_count
    return obj, 0


def _filter_pair_file(source: Path, destination: Path, kept_signature_ids: set[str]) -> tuple[int, int]:
    before = 0
    after = 0
    if source.suffix.lower() == ".csv":
        handle = tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
        )
        tmp = Path(handle.name)
        handle.close()
        try:
            with (
                source.open("r", encoding="utf-8", newline="") as inp,
                tmp.open("w", encoding="utf-8", newline="") as out,
            ):
                reader = csv.DictReader(inp)
                if reader.fieldnames is None:
                    raise ValueError(f"Empty pair CSV: {source}")
                writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    before += 1
                    if (
                        str(row["signature_id_1"]) in kept_signature_ids
                        and str(row["signature_id_2"]) in kept_signature_ids
                    ):
                        writer.writerow(row)
                        after += 1
            tmp.replace(destination)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
    else:
        rows = json.loads(source.read_text(encoding="utf-8"))
        before = len(rows)
        kept = [
            row
            for row in rows
            if str(row["signature_id_1"]) in kept_signature_ids and str(row["signature_id_2"]) in kept_signature_ids
        ]
        after = len(kept)
        _atomic_write_json(destination, kept)
    return before, after


def gate_public_dataset(
    *,
    dataset_name: str,
    signatures_json: Path,
    papers_json: Path,
    output_dir: Path,
    allowed_paper_ids: Collection[str],
    abstract_allowed_paper_ids: Collection[str],
    clusters_json: Path | None = None,
    specter_pickle: Path | None = None,
    unknown_signatures_json: Path | None = None,
    pair_files: Sequence[Path] = (),
    scrub_emails: bool = True,
) -> dict[str, Any]:
    """Filter non-S2AG papers, redact restricted abstracts to ``"Has Abstract"``, scrub emails, and filter sidecars.

    Writes atomically into a fresh ``output_dir`` and records the gating audit in
    ``preparation_report.public_gate.json`` (avoiding collision with ``population_corrections.json``).
    """
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Public snapshot gate requires a fresh output directory: {output_dir}")
    input_paths = [
        p.resolve()
        for p in (signatures_json, papers_json, clusters_json, specter_pickle, unknown_signatures_json, *pair_files)
        if p is not None
    ]
    for src in input_paths:
        if output_dir == src.parent or output_dir.is_relative_to(src) or src.is_relative_to(output_dir):
            raise ValueError(f"Output directory {output_dir} overlaps input path {src}")

    allowed_papers = {str(pid) for pid in allowed_paper_ids}
    abstract_allowed = {str(pid) for pid in abstract_allowed_paper_ids}
    output_dir.mkdir(parents=True, exist_ok=False)

    papers_before = 0
    papers_after = 0
    abstracts_present_before = 0
    abstracts_redacted = 0
    abstract_history_fields_redacted = 0
    paper_email_fields_scrubbed = 0
    raw_paper_ids: set[str] = set()
    kept_paper_ids: set[str] = set()
    dropped_paper_ids: list[str] = []

    out_papers = output_dir / f"{dataset_name}_papers.json"
    tmp_papers = Path(tempfile.NamedTemporaryFile(dir=output_dir, prefix=".papers.", suffix=".tmp", delete=False).name)
    try:
        with papers_json.open("rb") as inp, tmp_papers.open("w", encoding="utf-8", newline="\n") as out:
            out.write("{")
            first = True
            for pid, raw_record in ijson.kvitems(inp, "", use_float=True):
                papers_before += 1
                pid_str = str(pid)
                raw_paper_ids.add(pid_str)
                if pid_str not in allowed_papers:
                    dropped_paper_ids.append(pid_str)
                    continue
                record = dict(raw_record)
                raw_abstract = record.get("abstract")
                had_abstract = isinstance(raw_abstract, str) and bool(raw_abstract.strip())
                if had_abstract:
                    abstracts_present_before += 1
                if pid_str not in abstract_allowed:
                    new_abs, changed = _redact_abstract_value(raw_abstract)
                    record["abstract"] = new_abs
                    abstracts_redacted += int(changed)
                    for key in list(record):
                        if key == "abstract":
                            continue
                        val = record[key]
                        if "abstract" in str(key).lower() or isinstance(val, dict | list):
                            cleaned_hist, hist_cnt = _redact_abstract_history(val)
                            if hist_cnt:
                                record[key] = cleaned_hist
                                abstract_history_fields_redacted += hist_cnt
                if scrub_emails:
                    for field in ("title", "abstract"):
                        val = record.get(field)
                        if isinstance(val, str):
                            cleaned = scrub_text_emails(val)
                            if cleaned != val:
                                paper_email_fields_scrubbed += 1
                                record[field] = cleaned
                if not first:
                    out.write(",")
                first = False
                out.write(json.dumps(pid_str) + ":" + json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                kept_paper_ids.add(pid_str)
                papers_after += 1
            out.write("}\n")
        tmp_papers.replace(out_papers)
    except Exception:
        tmp_papers.unlink(missing_ok=True)
        raise

    signatures_before = 0
    signatures_after = 0
    signatures_dropped_non_s2ag = 0
    signatures_dropped_missing_paper = 0
    signature_email_fields_redacted = 0
    kept_sig_ids: set[str] = set()
    dropped_signature_ids: list[str] = []

    out_signatures = output_dir / f"{dataset_name}_signatures.json"
    tmp_sigs = Path(
        tempfile.NamedTemporaryFile(dir=output_dir, prefix=".signatures.", suffix=".tmp", delete=False).name
    )
    try:
        with signatures_json.open("rb") as inp, tmp_sigs.open("w", encoding="utf-8", newline="\n") as out:
            out.write("{")
            first = True
            for sid, raw_sig in ijson.kvitems(inp, "", use_float=True):
                signatures_before += 1
                sid_str = str(sid)
                sig_pid = str(raw_sig.get("paper_id", ""))
                if sig_pid not in kept_paper_ids:
                    dropped_signature_ids.append(sid_str)
                    if sig_pid in raw_paper_ids:
                        signatures_dropped_non_s2ag += 1
                    else:
                        signatures_dropped_missing_paper += 1
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
                if not first:
                    out.write(",")
                first = False
                out.write(json.dumps(sid_str) + ":" + json.dumps(sig, ensure_ascii=False, separators=(",", ":")))
                kept_sig_ids.add(sid_str)
                signatures_after += 1
            out.write("}\n")
        tmp_sigs.replace(out_signatures)
    except Exception:
        tmp_sigs.unlink(missing_ok=True)
        raise

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
            members = [str(m) for m in cluster.get("signature_ids", []) if str(m) in kept_sig_ids]
            if members:
                kept_clusters[str(cid)] = {**cluster, "signature_ids": members}
        clusters_after = len(kept_clusters)
        out_clusters = output_dir / f"{dataset_name}_clusters.json"
        _atomic_write_json(out_clusters, kept_clusters)
        outputs["clusters"] = {"path": out_clusters.name, "sha256": sha256_file(out_clusters)}

    specter_before = 0
    specter_after = 0
    if specter_pickle is not None:
        with specter_pickle.open("rb") as inp:
            raw_specter = pickle.load(inp)
        if not isinstance(raw_specter, dict):
            raise ValueError(f"Expected dict in SPECTER2 pickle: {specter_pickle}")
        specter_before = len(raw_specter)
        kept_specter = {str(pid): vec for pid, vec in raw_specter.items() if str(pid) in kept_paper_ids}
        specter_after = len(kept_specter)
        out_specter = output_dir / specter_pickle.name
        tmp_pkl = Path(
            tempfile.NamedTemporaryFile(dir=output_dir, prefix=".specter.", suffix=".tmp", delete=False).name
        )
        try:
            with tmp_pkl.open("wb") as out:
                pickle.dump(kept_specter, out, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_pkl.replace(out_specter)
        except Exception:
            tmp_pkl.unlink(missing_ok=True)
            raise
        outputs["specter_embeddings"] = {"path": out_specter.name, "sha256": sha256_file(out_specter)}

    unknown_before = 0
    unknown_after = 0
    if unknown_signatures_json is not None:
        raw_unknown = json.loads(unknown_signatures_json.read_text(encoding="utf-8"))
        if not isinstance(raw_unknown, list):
            raise ValueError(f"Expected list in unknown-signature sidecar: {unknown_signatures_json}")
        unknown_before = len(raw_unknown)
        kept_unknown = [str(sid) for sid in raw_unknown if str(sid) in kept_sig_ids]
        unknown_after = len(kept_unknown)
        out_unknown = output_dir / unknown_signatures_json.name
        _atomic_write_json(out_unknown, kept_unknown)
        outputs["unknown_signatures"] = {"path": out_unknown.name, "sha256": sha256_file(out_unknown)}

    pair_reports: dict[str, dict[str, int]] = {}
    for pair_src in pair_files:
        pair_dst = output_dir / pair_src.name
        p_before, p_after = _filter_pair_file(pair_src, pair_dst, kept_sig_ids)
        pair_reports[pair_src.name] = {"before": p_before, "after": p_after}
        outputs[f"pair_file:{pair_src.name}"] = {"path": pair_dst.name, "sha256": sha256_file(pair_dst)}

    report: dict[str, Any] = {
        "dataset": dataset_name,
        "abstract_redaction_marker": REDACTED_ABSTRACT_MARKER,
        "papers_before": papers_before,
        "papers_after": papers_after,
        "papers_dropped_non_s2ag": len(dropped_paper_ids),
        "signatures_before": signatures_before,
        "signatures_after": signatures_after,
        "signatures_dropped_non_s2ag": signatures_dropped_non_s2ag,
        "signatures_dropped_missing_paper": signatures_dropped_missing_paper,
        "clusters_before": clusters_before,
        "clusters_after": clusters_after,
        "specter_before": specter_before,
        "specter_after": specter_after,
        "unknown_signatures_before": unknown_before,
        "unknown_signatures_after": unknown_after,
        "pair_files": pair_reports,
        "abstracts_present_before": abstracts_present_before,
        "abstracts_redacted": abstracts_redacted,
        "abstract_history_fields_redacted": abstract_history_fields_redacted,
        "scrub_emails": bool(scrub_emails),
        "paper_email_fields_scrubbed": paper_email_fields_scrubbed,
        "signature_email_fields_redacted": signature_email_fields_redacted,
        "dropped_paper_ids_sample": sorted(dropped_paper_ids)[:20],
        "dropped_signature_ids_sample": sorted(dropped_signature_ids)[:20],
        "outputs": outputs,
    }
    _atomic_write_json(output_dir / "preparation_report.public_gate.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--signatures-json", type=Path, required=True)
    parser.add_argument("--papers-json", type=Path, required=True)
    parser.add_argument("--clusters-json", type=Path, default=None)
    parser.add_argument("--specter-pickle", type=Path, default=None)
    parser.add_argument("--unknown-signatures-json", type=Path, default=None)
    parser.add_argument("--pair-file", type=Path, action="append", default=[])
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
        specter_pickle=args.specter_pickle,
        unknown_signatures_json=args.unknown_signatures_json,
        pair_files=args.pair_file,
        output_dir=args.output_dir,
        allowed_paper_ids=load_paper_id_set(args.allowed_papers),
        abstract_allowed_paper_ids=load_paper_id_set(args.abstract_allowed_papers),
        scrub_emails=not args.no_scrub_emails,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
