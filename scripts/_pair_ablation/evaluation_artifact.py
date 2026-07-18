"""Immutable pairwise-evaluation artifacts shared across training runs."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from scripts._pair_ablation.pair_sources import PAIR_COLUMNS, canonicalize_pairs
from scripts._pair_ablation.results import load_strict_json, strict_json_digest

EVALUATION_ARTIFACT_SCHEMA_VERSION = "s2and_pair_ablation_evaluation_v1"

_MANIFEST_KEYS = {
    "config",
    "config_digest",
    "content_digest",
    "domains",
    "input_digest",
    "pairs_sha256",
    "rows",
    "schema_version",
}
_DOMAIN_KEYS = {"auprc_oracle", "negatives", "pair_digest", "positives", "rows", "source_domain"}


@dataclass(frozen=True, slots=True)
class EvaluationArtifact:
    """Validated evaluation rows and their semantic identities."""

    pairs: pd.DataFrame
    manifest: dict[str, Any]
    pair_digest_by_domain: dict[str, str]
    oracle_by_domain: dict[str, str]
    content_digest: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    observed = set(value)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(f"{context} schema mismatch: missing={missing}, extra={extra}")


def _require_digest(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def evaluation_pair_digest(frame: pd.DataFrame, *, oracle_kind: str) -> str:
    """Digest exact sorted evaluation pairs plus their oracle semantics."""

    oracle = str(oracle_kind).strip()
    if not oracle:
        raise ValueError("oracle_kind must be non-empty")
    canonical = canonicalize_pairs(frame.loc[:, PAIR_COLUMNS]).sort_values(
        ["source_domain", "pair1", "pair2", "label"],
        kind="stable",
    )
    if canonical.empty:
        raise ValueError("evaluation pair digest requires at least one row")
    domains = canonical["source_domain"].astype(str).unique().tolist()
    if len(domains) != 1:
        raise ValueError(f"evaluation pair digest requires one source domain, observed={domains}")
    digest = hashlib.sha256(f"s2and-evaluation-pairs-v1\0{oracle}\0".encode())
    for source_domain, _family, pair1, pair2, label, _rule, _origin, _group in canonical.itertuples(
        index=False,
        name=None,
    ):
        digest.update(f"{source_domain}\0{pair1}\0{pair2}\0{int(label)}\n".encode())
    return digest.hexdigest()


def build_evaluation_manifest(
    pairs: pd.DataFrame,
    *,
    oracle_by_domain: Mapping[str, str],
    evaluation_config: Mapping[str, Any],
    input_digest: str,
    pairs_sha256: str,
) -> dict[str, Any]:
    """Build a semantic manifest for one already-persisted evaluation catalog."""

    _require_digest(input_digest, "input_digest")
    _require_digest(pairs_sha256, "pairs_sha256")
    canonical = canonicalize_pairs(pairs.loc[:, PAIR_COLUMNS])
    observed_domains = set(canonical["source_domain"].astype(str))
    expected_domains = set(oracle_by_domain)
    if observed_domains != expected_domains:
        raise ValueError(
            f"evaluation domains mismatch: missing={sorted(expected_domains - observed_domains)}, "
            f"extra={sorted(observed_domains - expected_domains)}"
        )
    domains = []
    for domain in sorted(expected_domains):
        frame = canonical.loc[canonical["source_domain"].eq(domain), PAIR_COLUMNS]
        labels = frame["label"].astype("int8")
        positives = int(labels.sum())
        rows = int(len(frame))
        if positives == 0 or positives == rows:
            raise ValueError(f"evaluation domain={domain!r} must contain both classes")
        oracle = str(oracle_by_domain[domain]).strip()
        if not oracle:
            raise ValueError(f"evaluation oracle for domain={domain!r} must be non-empty")
        domains.append(
            {
                "source_domain": domain,
                "auprc_oracle": oracle,
                "rows": rows,
                "positives": positives,
                "negatives": rows - positives,
                "pair_digest": evaluation_pair_digest(frame, oracle_kind=oracle),
            }
        )
    config = dict(evaluation_config)
    config_digest = strict_json_digest(config)
    content_digest = strict_json_digest(domains)
    return {
        "schema_version": EVALUATION_ARTIFACT_SCHEMA_VERSION,
        "config": config,
        "config_digest": config_digest,
        "input_digest": input_digest,
        "pairs_sha256": pairs_sha256,
        "rows": int(len(canonical)),
        "domains": domains,
        "content_digest": content_digest,
    }


def write_evaluation_artifact(
    artifact_dir: Path,
    pairs: pd.DataFrame,
    *,
    oracle_by_domain: Mapping[str, str],
    evaluation_config: Mapping[str, Any],
    input_digest: str,
) -> EvaluationArtifact:
    """Atomically persist and validate an immutable evaluation artifact."""

    canonical = canonicalize_pairs(pairs.loc[:, PAIR_COLUMNS])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = artifact_dir / "pairs.parquet"
    temporary_pairs = artifact_dir / "pairs.tmp.parquet"
    canonical.to_parquet(temporary_pairs, index=False)
    temporary_pairs.replace(pairs_path)
    manifest = build_evaluation_manifest(
        canonical,
        oracle_by_domain=oracle_by_domain,
        evaluation_config=evaluation_config,
        input_digest=input_digest,
        pairs_sha256=_sha256_file(pairs_path),
    )
    manifest_path = artifact_dir / "manifest.json"
    temporary_manifest = artifact_dir / "manifest.json.tmp"
    import json

    temporary_manifest.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return load_evaluation_artifact(
        artifact_dir,
        oracle_by_domain=oracle_by_domain,
        evaluation_config=evaluation_config,
        input_digest=input_digest,
    )


def load_evaluation_artifact(
    artifact_dir: Path,
    *,
    oracle_by_domain: Mapping[str, str],
    evaluation_config: Mapping[str, Any],
    input_digest: str,
) -> EvaluationArtifact:
    """Load and fully verify one evaluation catalog and manifest."""

    pairs_path = artifact_dir / "pairs.parquet"
    manifest_path = artifact_dir / "manifest.json"
    if not pairs_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete evaluation artifact: {artifact_dir}")
    manifest = load_strict_json(manifest_path)
    _require_exact_keys(manifest, _MANIFEST_KEYS, "evaluation manifest")
    if manifest["schema_version"] != EVALUATION_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported evaluation schema version: {manifest['schema_version']!r}")
    expected_config = dict(evaluation_config)
    if manifest["config"] != expected_config or manifest["config_digest"] != strict_json_digest(expected_config):
        raise ValueError("evaluation artifact configuration mismatch")
    if manifest["input_digest"] != input_digest:
        raise ValueError("evaluation artifact input identity mismatch")
    _require_digest(manifest["input_digest"], "evaluation manifest input_digest")
    if manifest["pairs_sha256"] != _sha256_file(pairs_path):
        raise ValueError("evaluation pair Parquet hash mismatch")
    pairs = canonicalize_pairs(pd.read_parquet(pairs_path).loc[:, PAIR_COLUMNS])
    rebuilt = build_evaluation_manifest(
        pairs,
        oracle_by_domain=oracle_by_domain,
        evaluation_config=expected_config,
        input_digest=input_digest,
        pairs_sha256=manifest["pairs_sha256"],
    )
    if manifest != rebuilt:
        raise ValueError("evaluation manifest does not match evaluation pair content")
    pair_digest_by_domain: dict[str, str] = {}
    observed_oracles: dict[str, str] = {}
    for record in manifest["domains"]:
        if not isinstance(record, dict):
            raise ValueError("evaluation manifest domain entries must be objects")
        _require_exact_keys(record, _DOMAIN_KEYS, "evaluation manifest domain")
        domain = str(record["source_domain"])
        if domain in pair_digest_by_domain:
            raise ValueError(f"duplicate evaluation manifest domain: {domain}")
        pair_digest_by_domain[domain] = _require_digest(record["pair_digest"], f"evaluation pair digest {domain}")
        observed_oracles[domain] = str(record["auprc_oracle"])
    return EvaluationArtifact(
        pairs=pairs,
        manifest=manifest,
        pair_digest_by_domain=pair_digest_by_domain,
        oracle_by_domain=observed_oracles,
        content_digest=_require_digest(manifest["content_digest"], "evaluation content_digest"),
    )
