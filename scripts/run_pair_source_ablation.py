"""Run the practice pair-source leave-one-domain-out ablation.

This experiment intentionally targets the extant pre-canonical Arrow data.  It
uses :mod:`scripts._pair_ablation.legacy_rust`, which is practice-only and
refuses the updated ``canonical_v2`` artifacts.  It does not publish a model or
weaken the maintained Arrow validators.  Repeat the winning recipe through the
maintained loader when the updated datasets and name-count generation arrive.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import MutableMapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and.consts import FEATURIZER_VERSION  # noqa: E402
from s2and.featurizer import (  # noqa: E402
    DEFAULT_FEATURE_GROUPS,
    DEFAULT_NAMELESS_FEATURE_GROUPS,
    FeaturizationInfo,
)
from s2and.model import Clusterer, FastCluster  # noqa: E402
from scripts._pair_ablation.b3_cache import (  # noqa: E402
    B3RawFeatureStore,
    b3_cache_builder_identity,
    build_or_load_b3_raw_feature_store,
    score_b3_raw_feature_store,
)
from scripts._pair_ablation.evaluation import (  # noqa: E402
    B3_MEMBER_IDENTITY_VERSION,
    B3DomainEvaluationPlans,
    GoldBlockData,
    b3_for_threshold,
    build_b3_evaluation_plans,
    build_block_linkages,
    load_gold_block_data,
    tune_b3_threshold,
)
from scripts._pair_ablation.evaluation_artifact import (  # noqa: E402
    EvaluationArtifact,
    load_evaluation_artifact,
    write_evaluation_artifact,
)
from scripts._pair_ablation.feature_artifact import (  # noqa: E402
    DomainFeatureStore,
    load_feature_store,
    pair_identity_digest,
    sha256_file,
    write_feature_store,
)
from scripts._pair_ablation.legacy_rust import (  # noqa: E402
    ArtifactDigest,
    build_legacy_rust_featurizer,
    current_artifact_identity,
    featurize_labeled_pairs,
    resolve_legacy_arrow_manifest,
    signature_ids_for_labeled_pairs,
)
from scripts._pair_ablation.linker_pairs import (  # noqa: E402
    BIG_BLOCK_ORCID_LABEL_RULE,
    LINKER_COMPONENT_PROXY_LABEL_RULE,
    PUBLIC_GOLD_LABEL_RULE,
    extract_linker_pair_catalog,
    linker_signature_input_paths,
)
from scripts._pair_ablation.linker_pairs import (  # noqa: E402
    PAIR_COLUMNS as LINKER_PAIR_COLUMNS,
)
from scripts._pair_ablation.modeling import (  # noqa: E402
    BALANCED_RANDOM_FAMILY,
    BASE_FAMILY,
    LINKER_BIG_POSITIVE_FAMILY,
    LINKER_FAMILIES,
    LINKER_PROXY_NEGATIVE_FAMILY,
    LINKER_PUBLIC_FAMILY,
    AblationArm,
    ablation_arm_registry,
    additive_linker_arm,
    averaged_positive_probability,
    catalog_for_arm,
    default_ablation_arms,
    load_pairwise_models,
    pair_catalog_diversity_diagnostics,
    pairwise_metrics,
    source_counts,
    train_pairwise_models,
)
from scripts._pair_ablation.pair_sources import (  # noqa: E402
    PAIR_COLUMNS,
    canonicalize_pairs,
    cap_pairs_per_domain,
    load_historical_augmented_pairs,
    load_medline_pairs,
    sample_within_blocks_balanced,
    sample_within_blocks_uniform,
)
from scripts._pair_ablation.ranking import (  # noqa: E402
    BASELINE_ARM,
    RANKING_INPUT_SCHEMA_VERSION,
    LoadedFold,
    load_ranking_input,
)
from scripts._pair_ablation.results import (  # noqa: E402
    FOLD_RESULT_SCHEMA_VERSION,
    FoldResultExpectation,
    load_fold_result,
    load_strict_json,
    recipe_id_for,
    strict_json_digest,
    write_fold_result,
)
from scripts._pair_ablation.run_identity import (  # noqa: E402
    RUN_MANIFEST_SCHEMA_VERSION,
    THREAD_ENVIRONMENT_KEYS,
    build_run_manifest,
    current_runtime_versions,
    load_run_manifest,
    rust_extension_binary_sha256,
)

logger = logging.getLogger("pair_source_ablation")

PUBLIC_DOMAINS = ("aminer", "arnetminer", "inspire", "kisti", "pubmed", "qian", "zbmath")
BIG_BLOCK_DOMAINS = ("a_khan", "a_silva", "h_wang", "j_smith", "s_gupta", "s_lee", "s_park")
PAIRWISE_ONLY_DOMAINS = ("medline",)
ALL_FOLD_DOMAINS = (*PUBLIC_DOMAINS, *PAIRWISE_ONLY_DOMAINS, *BIG_BLOCK_DOMAINS)
LINKER_PUBLIC_DOMAINS = tuple(domain for domain in PUBLIC_DOMAINS if domain != "aminer")

EVAL_GOLD_FAMILY = "evaluation_gold"
EVAL_FIXED_FAMILY = "evaluation_fixed"
EVAL_LINKER_PROXY_FAMILY = "evaluation_linker_component_proxy"
TRAINING_CATALOG_SCHEMA_VERSION = "s2and_pair_ablation_training_catalog_v1"
MODEL_CACHE_SCHEMA_VERSION = "s2and_pair_ablation_model_cache_v1"
_ADDITIVE_LINKER_ARM_PATTERN = re.compile(r"uniform_100k_plus_linker_(all13|big7)_([1-9][0-9]*)")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """The controls that determine catalogs, models, and metrics."""

    training_seed: int
    evaluation_seed: int
    n_jobs: int
    total_ram_gib: int
    uniform_pairs_per_domain: int
    name_pairs_per_domain: int
    balanced_medium_pairs_per_domain: int
    balanced_pool_pairs_per_domain: int
    linker_pairs_per_domain: int
    big_proxy_eval_pairs_per_class: int
    catalog_pool_cap_per_domain: int | None
    eval_pairs_per_domain: int
    threshold_pairs_per_domain: int
    estimator_scale: float
    b3_scope: str
    public_domains: tuple[str, ...]
    big_block_domains: tuple[str, ...]
    fold_domains: tuple[str, ...]
    arm_names: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_content_identity(
    raw_path: Path,
    digest_cache: MutableMapping[Path, ArtifactDigest],
) -> dict[str, Any]:
    """Return a strong content identity, reusing hashes for shared paths."""

    path = raw_path.resolve()
    cached = digest_cache.get(path)
    if cached is None:
        logger.info("hashing input artifact path=%s", path)
        if path.is_file():
            before = path.stat()
            sha256 = _sha256_file(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Input file changed while it was being hashed: {path}")
            cached = ArtifactDigest(path=path, kind="file", size_bytes=after.st_size, sha256=sha256)
        elif path.is_dir():
            hasher = hashlib.sha256(b"s2and-legacy-directory-sha256-v1\0")
            size_bytes = 0
            for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
                relative = child.relative_to(path).as_posix().encode("utf-8")
                child_size = child.stat().st_size
                hasher.update(relative)
                hasher.update(b"\0")
                hasher.update(str(child_size).encode("ascii"))
                hasher.update(b"\0")
                with child.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        hasher.update(chunk)
                hasher.update(b"\0")
                size_bytes += child_size
            cached = ArtifactDigest(
                path=path,
                kind="directory",
                size_bytes=size_bytes,
                sha256=hasher.hexdigest(),
            )
        else:
            raise FileNotFoundError(f"Cannot digest missing input artifact: {path}")
        digest_cache[path] = cached
    return {
        "path": str(cached.path),
        "kind": cached.kind,
        "size_bytes": cached.size_bytes,
        "sha256": cached.sha256,
    }


def _json_digest(value: Any) -> str:
    return strict_json_digest(value)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _purpose_seed(seed: int, *parts: str) -> int:
    """Derive a deterministic named sub-seed without magic arithmetic offsets."""

    payload = "\0".join((str(int(seed)), *map(str, parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def _save_array_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, values)
    temporary.replace(path)


def _git_identity() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(arguments, cwd=REPO_ROOT, text=True).strip()

    diff_binary = subprocess.check_output(("git", "diff", "--binary", "HEAD"), cwd=REPO_ROOT)
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "diff_binary_sha256": hashlib.sha256(diff_binary).hexdigest(),
        "status_short": run("git", "status", "--short"),
    }


def _implementation_identity() -> dict[str, str]:
    paths = {
        Path(__file__).resolve(),
        *((REPO_ROOT / "scripts" / "_pair_ablation").rglob("*.py")),
        *((REPO_ROOT / "s2and").rglob("*.py")),
    }
    return {str(path.relative_to(REPO_ROOT)): _sha256_file(path) for path in sorted(paths)}


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame without pandas' optional tabulate dependency."""

    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for values in frame.itertuples(index=False, name=None):
        rows.append(
            "| "
            + " | ".join(
                "" if pd.isna(value) else str(value).replace("|", "\\|").replace("\n", " ") for value in values
            )
            + " |"
        )
    return "\n".join((header, separator, *rows))


def _main_featurizer_info() -> FeaturizationInfo:
    return FeaturizationInfo(features_to_use=list(DEFAULT_FEATURE_GROUPS), featurizer_version=FEATURIZER_VERSION)


def _nameless_featurizer_info() -> FeaturizationInfo:
    return FeaturizationInfo(
        features_to_use=list(DEFAULT_NAMELESS_FEATURE_GROUPS),
        featurizer_version=FEATURIZER_VERSION,
    )


def _standard_manifest(data_root: Path, domain: str) -> Path:
    return data_root / domain / "manifest.json"


def _domain_manifest(data_root: Path, linker_bundle_root: Path, domain: str) -> Path:
    if domain in BIG_BLOCK_DOMAINS:
        return linker_bundle_root / "datasets" / domain / "manifest.json"
    return _standard_manifest(data_root, domain)


def _gold_path(data_root: Path, domain: str) -> Path:
    return data_root / domain / f"{domain}_clusters.json"


def _bundle_file(bundle_root: Path, raw_path: object, *, context: str) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        raise ValueError(f"{context} must be bundle-relative: {path}")
    resolved = (bundle_root / path).resolve()
    if not resolved.is_relative_to(bundle_root):
        raise ValueError(f"{context} escapes the linker bundle: {path}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing declared {context}: {resolved}")
    return resolved


def _catalog_input_paths(
    *,
    data_root: Path,
    backup_data_root: Path,
    linker_bundle_root: Path,
    config: ExperimentConfig,
) -> dict[str, Path]:
    """Resolve every file whose contents determine the persisted pair catalog."""

    paths: dict[str, Path] = {}
    for domain in config.public_domains:
        paths[f"public.{domain}.signatures"] = data_root / domain / "signatures.arrow"
        paths[f"public.{domain}.clusters"] = _gold_path(data_root, domain)
    for split in ("train", "test"):
        paths[f"medline.{split}_pairs"] = backup_data_root / "medline" / f"{split}_pairs.csv"
    for split in ("train", "val", "test"):
        paths[f"augmented.{split}_pairs"] = backup_data_root / "augmented" / f"{split}_pairs.csv"

    selected_linker_domains = {
        *(domain for domain in config.public_domains if domain in LINKER_PUBLIC_DOMAINS),
        *config.big_block_domains,
    }
    if selected_linker_domains:
        bundle_path = (linker_bundle_root / "bundle.json").resolve()
        if not bundle_path.is_file():
            raise FileNotFoundError(f"Missing linker bundle manifest: {bundle_path}")
        paths["linker.bundle"] = bundle_path
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        try:
            label_assets = payload["assets"]["featureless_rows"]["files"]
            component_assets = payload["assets"]["candidate_members"]["datasets"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Linker bundle does not declare label and candidate-member assets") from exc
        if not isinstance(label_assets, dict) or not isinstance(component_assets, dict):
            raise ValueError("Linker bundle label and candidate-member assets must be mappings")
        for key, raw_path in sorted(label_assets.items()):
            paths[f"linker.labels.{key}"] = _bundle_file(
                linker_bundle_root,
                raw_path,
                context=f"featureless_rows.files[{key!r}]",
            )
        missing_components = sorted(selected_linker_domains.difference(component_assets))
        if missing_components:
            raise ValueError(f"Linker bundle lacks candidate-member assets for: {missing_components}")
        for domain in sorted(selected_linker_domains):
            paths[f"linker.components.{domain}"] = _bundle_file(
                linker_bundle_root,
                component_assets[domain],
                context=f"candidate_members.datasets[{domain!r}]",
            )
        for key, path in linker_signature_input_paths(
            bundle_path,
            public_datasets={domain for domain in selected_linker_domains if domain in LINKER_PUBLIC_DOMAINS},
            big_block_datasets=config.big_block_domains,
        ).items():
            paths[f"linker.{key}"] = path
    return {key: path.resolve() for key, path in paths.items()}


def build_input_identity(
    *,
    data_root: Path,
    backup_data_root: Path,
    linker_bundle_root: Path,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Hash all label and feature inputs once so every checkpoint is content-bound."""

    digest_cache: dict[Path, ArtifactDigest] = {}
    catalog_files = {
        key: _path_content_identity(path, digest_cache)
        for key, path in sorted(
            _catalog_input_paths(
                data_root=data_root,
                backup_data_root=backup_data_root,
                linker_bundle_root=linker_bundle_root,
                config=config,
            ).items()
        )
    }
    feature_artifacts = {}
    feature_domains = (*config.public_domains, "medline", *config.big_block_domains)
    for domain in feature_domains:
        artifacts = resolve_legacy_arrow_manifest(_domain_manifest(data_root, linker_bundle_root, domain))
        feature_artifacts[domain] = current_artifact_identity(
            artifacts,
            include_path_digests=True,
            digest_cache=digest_cache,
        )
    return {
        "roots": {
            "data_root": str(data_root),
            "backup_data_root": str(backup_data_root),
            "linker_bundle_root": str(linker_bundle_root),
        },
        "catalog_files": catalog_files,
        "feature_artifacts": feature_artifacts,
    }


def _training_gold_catalogs(
    gold: GoldBlockData,
    *,
    uniform_pairs: int,
    balanced_pool_pairs: int,
    training_seed: int,
) -> tuple[pd.DataFrame, ...]:
    """Build training-only gold pair sources with named deterministic seeds."""

    if balanced_pool_pairs <= 0 or balanced_pool_pairs % 2:
        raise ValueError("balanced_pool_pairs must be a positive even 50/50 pool size")
    pool_seed = _purpose_seed(training_seed, "gold_uniform_pool", gold.dataset)
    pool = sample_within_blocks_uniform(
        gold.blocks,
        gold.cluster_by_signature,
        uniform_pairs,
        random_seed=pool_seed,
        source_domain=gold.dataset,
        source_family="uniform_pool",
    )
    base = pool.copy()
    base["source_family"] = BASE_FAMILY
    base["origin"] = f"virtual_uniform_base:{gold.dataset}:seed={pool_seed}"
    balanced = sample_within_blocks_balanced(
        gold.blocks,
        gold.cluster_by_signature,
        positive_size=balanced_pool_pairs // 2,
        negative_size=balanced_pool_pairs // 2,
        random_seed=_purpose_seed(training_seed, "gold_balanced_pool", gold.dataset),
        source_domain=gold.dataset,
        source_family=BALANCED_RANDOM_FAMILY,
    )
    return (
        canonicalize_pairs(base.loc[:, PAIR_COLUMNS]),
        canonicalize_pairs(balanced.loc[:, PAIR_COLUMNS]),
    )


def _public_evaluation_pairs(
    gold: GoldBlockData,
    *,
    eval_pairs: int,
    evaluation_seed: int,
) -> pd.DataFrame:
    """Build immutable public pair evaluation rows from the evaluation seed."""

    return canonicalize_pairs(
        sample_within_blocks_uniform(
            gold.blocks,
            gold.cluster_by_signature,
            eval_pairs,
            random_seed=_purpose_seed(evaluation_seed, "public_pair_evaluation", gold.dataset),
            source_domain=gold.dataset,
            source_family=EVAL_GOLD_FAMILY,
        ).loc[:, PAIR_COLUMNS]
    )


def _fixed_binary_evaluation_cap(
    frame: pd.DataFrame,
    *,
    cap_per_class: int,
    evaluation_seed: int,
) -> pd.DataFrame:
    """Apply the same per-class evaluation cap without majority backfill."""

    frames = []
    for label in (0, 1):
        subset = frame.loc[frame["label"].eq(label), PAIR_COLUMNS]
        if subset.empty:
            continue
        frames.append(
            cap_pairs_per_domain(
                subset,
                cap_per_class,
                random_seed=_purpose_seed(evaluation_seed, "big_proxy_evaluation", f"label={label}"),
                sampling="query_uniform",
            )
        )
    if not frames:
        return frame.loc[[], PAIR_COLUMNS].copy()
    selected = canonicalize_pairs(pd.concat(frames, ignore_index=True))
    selected["source_family"] = EVAL_LINKER_PROXY_FAMILY
    return canonicalize_pairs(selected.loc[:, PAIR_COLUMNS])


def _cap_training_pool(
    frame: pd.DataFrame,
    *,
    config: ExperimentConfig,
    purpose: str,
) -> pd.DataFrame:
    """Bound non-gold source pools for smoke/preflight runs only."""

    canonical = canonicalize_pairs(frame.loc[:, PAIR_COLUMNS])
    if config.catalog_pool_cap_per_domain is None:
        return canonical
    return cap_pairs_per_domain(
        canonical,
        config.catalog_pool_cap_per_domain,
        random_seed=_purpose_seed(config.training_seed, "catalog_pool_cap", purpose),
        sampling="query_uniform",
    )


def _evaluation_artifact_identity(
    input_identity: dict[str, Any],
    config: ExperimentConfig,
) -> tuple[dict[str, str], dict[str, Any], str]:
    oracle_by_domain = {
        domain: (
            "gold_cluster_pairs"
            if domain in config.public_domains
            else "fixed_pair_labels"
            if domain == "medline"
            else "linker_component_proxy"
        )
        for domain in config.fold_domains
    }
    evaluation_config = {
        "evaluation_seed": config.evaluation_seed,
        "eval_pairs_per_domain": config.eval_pairs_per_domain,
        "big_proxy_eval_pairs_per_class": config.big_proxy_eval_pairs_per_class,
        "fold_domains": list(config.fold_domains),
    }
    relevant_files = {
        key: value for key, value in input_identity["catalog_files"].items() if not str(key).startswith("augmented.")
    }
    relevant_features = {domain: input_identity["feature_artifacts"][domain] for domain in config.fold_domains}
    input_digest = _json_digest(
        {
            "catalog_files": relevant_files,
            "feature_artifacts": relevant_features,
        }
    )
    return oracle_by_domain, evaluation_config, input_digest


def build_pair_catalogs(
    *,
    data_root: Path,
    backup_data_root: Path,
    linker_bundle_root: Path,
    output_dir: Path,
    config: ExperimentConfig,
    input_identity: dict[str, Any],
    resume: bool,
) -> tuple[pd.DataFrame, EvaluationArtifact]:
    """Build separate training and immutable evaluation pair artifacts."""

    catalog_path = output_dir / "catalog" / "training_pairs.parquet"
    metadata_path = output_dir / "catalog" / "metadata.json"
    evaluation_dir = output_dir / "evaluation"
    diagnostic_paths = tuple(
        output_dir / "catalog" / name
        for name in (
            "diversity_diagnostics.json",
            "domain_family.csv",
            "label_rules.csv",
            "reference_overlap.csv",
        )
    )
    catalog_input_digest = _json_digest(input_identity["catalog_files"])
    oracle_by_domain, evaluation_config, evaluation_input_digest = _evaluation_artifact_identity(
        input_identity,
        config,
    )
    if resume and catalog_path.exists() and metadata_path.exists() and (evaluation_dir / "manifest.json").exists():
        metadata = load_strict_json(metadata_path)
        if (
            metadata.get("schema_version") == TRAINING_CATALOG_SCHEMA_VERSION
            and metadata.get("catalog_config_digest") == _json_digest(asdict(config))
            and metadata.get("catalog_input_digest") == catalog_input_digest
            and metadata.get("catalog_sha256") == sha256_file(catalog_path)
            and all(path.is_file() for path in diagnostic_paths)
        ):
            training = pd.read_parquet(catalog_path)
            evaluation = load_evaluation_artifact(
                evaluation_dir,
                oracle_by_domain=oracle_by_domain,
                evaluation_config=evaluation_config,
                input_digest=evaluation_input_digest,
            )
            return training, evaluation
        raise RuntimeError("Existing pair catalog was built from a different configuration or input generation")

    training_frames: list[pd.DataFrame] = []
    evaluation_frames: list[pd.DataFrame] = []
    for domain in config.public_domains:
        logger.info("loading gold block metadata domain=%s", domain)
        gold = load_gold_block_data(
            domain,
            data_root / domain / "signatures.arrow",
            _gold_path(data_root, domain),
        )
        training_frames.extend(
            _training_gold_catalogs(
                gold,
                uniform_pairs=config.uniform_pairs_per_domain,
                balanced_pool_pairs=config.balanced_pool_pairs_per_domain,
                training_seed=config.training_seed,
            )
        )
        if domain in config.fold_domains:
            evaluation_frames.append(
                _public_evaluation_pairs(
                    gold,
                    eval_pairs=config.eval_pairs_per_domain,
                    evaluation_seed=config.evaluation_seed,
                )
            )

    if "medline" in config.fold_domains or config.arm_names:
        medline = load_medline_pairs(backup_data_root / "medline")
        training_frames.append(_cap_training_pool(medline, config=config, purpose="medline"))
        if "medline" in config.fold_domains:
            medline_eval = medline.copy()
            medline_eval["source_family"] = EVAL_FIXED_FAMILY
            evaluation_frames.append(canonicalize_pairs(medline_eval.loc[:, PAIR_COLUMNS]))

    augmented = load_historical_augmented_pairs(backup_data_root / "augmented")
    # ORCID is inconsistent and Medline is exactly duplicated by its direct
    # fixed source.  Public-domain augmentation remains a distinct ablation.
    augmented = augmented.loc[augmented["source_domain"].isin(config.public_domains), PAIR_COLUMNS]
    if not augmented.empty:
        training_frames.append(_cap_training_pool(augmented, config=config, purpose="augmented"))

    selected_linker_public = tuple(domain for domain in config.public_domains if domain in LINKER_PUBLIC_DOMAINS)
    if selected_linker_public or config.big_block_domains:
        logger.info("extracting linker-derived pairs")
        linker = extract_linker_pair_catalog(
            linker_bundle_root / "bundle.json",
            public_gold_cluster_paths={domain: _gold_path(data_root, domain) for domain in selected_linker_public},
            big_block_datasets=config.big_block_domains,
            proxy_negatives_per_query=2,
            proxy_negatives_per_domain=None,
            seed=_purpose_seed(config.evaluation_seed, "linker_proxy_candidate_pool"),
        )
        strict = linker.strict.loc[:, LINKER_PAIR_COLUMNS].copy()
        public_strict = strict.loc[strict["label_rule"].eq(PUBLIC_GOLD_LABEL_RULE), PAIR_COLUMNS].copy()
        big_positive = strict.loc[strict["label_rule"].eq(BIG_BLOCK_ORCID_LABEL_RULE), PAIR_COLUMNS].copy()
        if not public_strict.empty:
            public_strict["source_family"] = LINKER_PUBLIC_FAMILY
            training_frames.append(_cap_training_pool(public_strict, config=config, purpose="linker_public"))
        if not big_positive.empty:
            big_positive["source_family"] = LINKER_BIG_POSITIVE_FAMILY
            big_positive = canonicalize_pairs(big_positive)
            training_frames.append(_cap_training_pool(big_positive, config=config, purpose="linker_big_positive"))

        proxy = linker.linker_component_proxy.loc[:, LINKER_PAIR_COLUMNS].copy()
        proxy = proxy.loc[proxy["label_rule"].eq(LINKER_COMPONENT_PROXY_LABEL_RULE), PAIR_COLUMNS]
        if not proxy.empty:
            proxy["source_family"] = LINKER_PROXY_NEGATIVE_FAMILY
            proxy = canonicalize_pairs(proxy)
            training_frames.append(_cap_training_pool(proxy, config=config, purpose="linker_proxy_negative"))

        if not big_positive.empty and not proxy.empty:
            proxy_eval = pd.concat([big_positive, proxy], ignore_index=True)
            proxy_eval = proxy_eval.loc[proxy_eval["source_domain"].isin(config.fold_domains), PAIR_COLUMNS]
            if not proxy_eval.empty:
                evaluation_frames.append(
                    _fixed_binary_evaluation_cap(
                        proxy_eval,
                        cap_per_class=config.big_proxy_eval_pairs_per_class,
                        evaluation_seed=config.evaluation_seed,
                    )
                )

    if not training_frames or not evaluation_frames:
        raise RuntimeError("Pair catalog construction produced no training or evaluation rows")
    # Preserve source-family identity in the training catalog. Canonicalize each
    # family separately, then prove that overlapping families never disagree.
    family_frames = [canonicalize_pairs(frame.loc[:, PAIR_COLUMNS]) for frame in training_frames if not frame.empty]
    training_catalog = pd.concat(family_frames, ignore_index=True)
    conflicts = (
        training_catalog.groupby(["source_domain", "pair1", "pair2"], sort=False, observed=True)["label"]
        .nunique()
        .gt(1)
    )
    if bool(conflicts.any()):
        examples = conflicts.index[conflicts].tolist()[:5]
        raise ValueError(f"Pair sources disagree on final labels for canonical pair keys: {examples}")
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    training_catalog.to_parquet(catalog_path, index=False)
    diversity = pair_catalog_diversity_diagnostics(training_catalog, reference_family=BASE_FAMILY)
    _write_json_atomic(catalog_path.parent / "diversity_diagnostics.json", diversity)
    for table_name in ("domain_family", "label_rules", "reference_overlap"):
        pd.DataFrame(diversity[table_name]).to_csv(catalog_path.parent / f"{table_name}.csv", index=False)
    metadata = {
        "schema_version": TRAINING_CATALOG_SCHEMA_VERSION,
        "catalog_config_digest": _json_digest(asdict(config)),
        "catalog_input_digest": catalog_input_digest,
        "catalog_inputs": input_identity["catalog_files"],
        "catalog_path": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "rows": int(len(training_catalog)),
        "source_counts": source_counts(training_catalog),
        "diversity_diagnostics_path": str(catalog_path.parent / "diversity_diagnostics.json"),
    }
    _write_json_atomic(metadata_path, metadata)
    evaluation_catalog = canonicalize_pairs(
        pd.concat([frame.loc[:, PAIR_COLUMNS] for frame in evaluation_frames if not frame.empty], ignore_index=True)
    )
    evaluation = write_evaluation_artifact(
        evaluation_dir,
        evaluation_catalog,
        oracle_by_domain=oracle_by_domain,
        evaluation_config=evaluation_config,
        input_digest=evaluation_input_digest,
    )
    return training_catalog, evaluation


def load_reused_pair_artifacts(
    *,
    artifact_source_dir: Path,
    output_dir: Path,
    config: ExperimentConfig,
    input_identity: dict[str, Any],
) -> tuple[pd.DataFrame, EvaluationArtifact, dict[str, DomainFeatureStore]]:
    """Load a prior seed-matched catalog and feature stores after strict checks."""

    source_dir = artifact_source_dir.resolve()
    source_manifest_path = source_dir / "run_manifest.json"
    source_manifest = load_run_manifest(source_manifest_path)
    source_config = dict(source_manifest["config"])
    current_config = asdict(config)
    source_config.pop("arm_names", None)
    current_config.pop("arm_names", None)
    if strict_json_digest(source_config) != strict_json_digest(current_config):
        raise ValueError("reused artifact run configuration differs outside arm_names")
    if source_manifest["input_identity"] != input_identity:
        raise ValueError("reused artifact input identity differs from the current raw inputs")

    catalog_path = source_dir / "catalog" / "training_pairs.parquet"
    catalog_metadata_path = source_dir / "catalog" / "metadata.json"
    catalog_metadata = load_strict_json(catalog_metadata_path)
    if (
        not catalog_path.is_file()
        or catalog_metadata.get("catalog_sha256") != sha256_file(catalog_path)
        or catalog_metadata.get("rows") is None
    ):
        raise ValueError("reused training catalog is missing or does not match its metadata")
    training_catalog = pd.read_parquet(catalog_path)
    if tuple(training_catalog.columns) != PAIR_COLUMNS:
        raise ValueError(
            "reused training catalog schema mismatch: "
            f"expected={PAIR_COLUMNS}, observed={tuple(training_catalog.columns)}"
        )
    if int(catalog_metadata["rows"]) != len(training_catalog):
        raise ValueError("reused training catalog row count does not match its metadata")
    if catalog_metadata.get("source_counts") != source_counts(training_catalog):
        raise ValueError("reused training catalog source counts do not match its metadata")

    oracle_by_domain, evaluation_config, evaluation_input_digest = _evaluation_artifact_identity(
        input_identity,
        config,
    )
    evaluation = load_evaluation_artifact(
        source_dir / "evaluation",
        oracle_by_domain=oracle_by_domain,
        evaluation_config=evaluation_config,
        input_digest=evaluation_input_digest,
    )
    stores = load_feature_stores(
        source_dir,
        training_catalog=training_catalog,
        evaluation_catalog=evaluation.pairs,
        input_identity=input_identity,
    )
    feature_manifest_sha256 = {
        domain: sha256_file(source_dir / "features" / domain / "manifest.json")
        for domain in sorted(stores)
    }
    verification = {
        "schema_version": "s2and_pair_ablation_artifact_reuse_verification_v1",
        "artifact_source_dir": str(source_dir),
        "source_run_manifest_path": str(source_manifest_path.resolve()),
        "source_run_manifest_sha256": sha256_file(source_manifest_path),
        "source_run_id": source_manifest["run_id"],
        "training_seed": config.training_seed,
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256": catalog_metadata["catalog_sha256"],
        "catalog_rows": len(training_catalog),
        "evaluation_manifest_path": str((source_dir / "evaluation" / "manifest.json").resolve()),
        "evaluation_manifest_sha256": sha256_file(source_dir / "evaluation" / "manifest.json"),
        "feature_manifest_sha256": feature_manifest_sha256,
    }
    _write_json_atomic(output_dir / "artifact_reuse_verification.json", verification)
    return training_catalog, evaluation, stores


def validated_reused_b3_builder_identity(
    *,
    artifact_source_dir: Path,
    output_dir: Path,
    current_implementation_sha256: dict[str, str],
    current_runtime_versions: dict[str, str],
    current_rust_version: str,
    current_rust_extension_sha256: str,
) -> str:
    """Reuse the raw B3 cache only when every builder dependency is unchanged."""

    source_manifest_path = artifact_source_dir.resolve() / "run_manifest.json"
    source_manifest = load_run_manifest(source_manifest_path)
    allowed_non_builder_changes = {
        "scripts/run_pair_source_ablation.py",
        "scripts/_pair_ablation/modeling.py",
    }
    source_implementation = source_manifest["implementation_sha256"]
    if set(source_implementation) != set(current_implementation_sha256):
        raise ValueError("cannot reuse B3 builder identity after implementation file-set changes")
    changed_paths_raw = {
        path
        for path in source_implementation
        if source_implementation[path] != current_implementation_sha256[path]
    }
    changed_paths = {path.replace("\\", "/") for path in changed_paths_raw}
    unexpected_changes = sorted(changed_paths.difference(allowed_non_builder_changes))
    if unexpected_changes:
        raise ValueError(
            "cannot reuse B3 builder identity because builder dependencies changed: "
            f"{unexpected_changes}"
        )
    if source_manifest["runtime_versions"] != current_runtime_versions:
        raise ValueError("cannot reuse B3 builder identity after runtime-version changes")
    if source_manifest["rust_version"] != current_rust_version:
        raise ValueError("cannot reuse B3 builder identity after Rust version changes")
    if source_manifest["rust_extension_sha256"] != current_rust_extension_sha256:
        raise ValueError("cannot reuse B3 builder identity after Rust extension changes")
    if source_manifest["featurizer_version"] != FEATURIZER_VERSION:
        raise ValueError("cannot reuse B3 builder identity after featurizer-version changes")

    builder_identity = b3_cache_builder_identity(
        implementation_sha256=source_implementation,
        runtime_versions=source_manifest["runtime_versions"],
    )
    verification = {
        "schema_version": "s2and_pair_ablation_b3_builder_reuse_verification_v1",
        "artifact_source_run_manifest_path": str(source_manifest_path),
        "artifact_source_run_manifest_sha256": sha256_file(source_manifest_path),
        "allowed_non_builder_changes": sorted(allowed_non_builder_changes),
        "observed_changed_paths": sorted(changed_paths),
        "cache_builder_identity": builder_identity,
        "runtime_versions": current_runtime_versions,
        "rust_version": current_rust_version,
        "rust_extension_sha256": current_rust_extension_sha256,
        "featurizer_version": FEATURIZER_VERSION,
    }
    _write_json_atomic(output_dir / "b3_builder_reuse_verification.json", verification)
    return builder_identity


def build_feature_stores(
    training_catalog: pd.DataFrame,
    evaluation_catalog: pd.DataFrame,
    *,
    data_root: Path,
    linker_bundle_root: Path,
    output_dir: Path,
    config: ExperimentConfig,
    input_identity: dict[str, Any],
    resume: bool,
) -> None:
    """Featurize every unique catalog pair exactly once with Rust."""

    unique = canonicalize_pairs(
        pd.concat(
            [
                training_catalog.loc[:, PAIR_COLUMNS],
                evaluation_catalog.loc[:, PAIR_COLUMNS],
            ],
            ignore_index=True,
        )
    )
    main_info = _main_featurizer_info()
    nameless_info = _nameless_featurizer_info()
    expected_main_indices = sorted(
        {
            index
            for feature_name in main_info.features_to_use
            for index in main_info.feature_group_to_index[feature_name]
        }
    )
    expected_nameless_indices = sorted(
        {
            index
            for feature_name in nameless_info.features_to_use
            for index in nameless_info.feature_group_to_index[feature_name]
        }
    )
    for domain, raw_domain_pairs in unique.groupby("source_domain", sort=True):
        domain = str(domain)
        domain_pairs = canonicalize_pairs(raw_domain_pairs.loc[:, PAIR_COLUMNS])
        store_dir = output_dir / "features" / domain
        pair_digest = pair_identity_digest(domain_pairs)
        artifacts = resolve_legacy_arrow_manifest(_domain_manifest(data_root, linker_bundle_root, domain))
        try:
            artifact_identity = input_identity["feature_artifacts"][domain]
        except KeyError as exc:
            raise ValueError(f"Input identity is missing feature artifacts for domain={domain!r}") from exc
        artifact_identity_digest = _json_digest(artifact_identity)
        if resume and (store_dir / "manifest.json").exists():
            load_feature_store(
                store_dir,
                expected_domain=domain,
                expected_pair_digest=pair_digest,
                expected_artifact_identity_digest=artifact_identity_digest,
                expected_main_feature_indices=expected_main_indices,
                expected_nameless_feature_indices=expected_nameless_indices,
            )
            logger.info("feature store already complete domain=%s rows=%d", domain, len(domain_pairs))
            continue

        tuples = tuple(domain_pairs[["pair1", "pair2", "label"]].itertuples(index=False, name=None))
        signature_ids = signature_ids_for_labeled_pairs(tuples)
        logger.info(
            "building Rust featurizer domain=%s signatures=%d pairs=%d",
            domain,
            len(signature_ids),
            len(tuples),
        )
        rust_featurizer = build_legacy_rust_featurizer(
            artifacts,
            n_jobs=config.n_jobs,
            signature_ids=signature_ids,
        )
        matrices = featurize_labeled_pairs(
            rust_featurizer,
            tuples,
            featurization_info=main_info,
            nameless_featurization_info=nameless_info,
            n_jobs=config.n_jobs,
        )
        if matrices.nameless is None:
            raise RuntimeError("Nameless featurization unexpectedly returned None")
        if (
            list(matrices.main_feature_indices) != expected_main_indices
            or list(matrices.nameless_feature_indices or ()) != expected_nameless_indices
        ):
            raise RuntimeError(f"Rust feature projection mismatch for domain={domain!r}")
        write_feature_store(
            store_dir,
            domain=domain,
            pairs=domain_pairs,
            main=matrices.main,
            nameless=matrices.nameless,
            labels=matrices.labels,
            artifact_identity_digest=artifact_identity_digest,
            artifact_manifest_sha256=artifacts.manifest_sha256,
            main_feature_indices=expected_main_indices,
            nameless_feature_indices=expected_nameless_indices,
        )
        del matrices, rust_featurizer
        gc.collect()


def load_feature_stores(
    output_dir: Path,
    *,
    training_catalog: pd.DataFrame,
    evaluation_catalog: pd.DataFrame,
    input_identity: dict[str, Any],
) -> dict[str, DomainFeatureStore]:
    stores: dict[str, DomainFeatureStore] = {}
    feature_root = output_dir / "features"
    unique = canonicalize_pairs(
        pd.concat(
            [training_catalog.loc[:, PAIR_COLUMNS], evaluation_catalog.loc[:, PAIR_COLUMNS]],
            ignore_index=True,
        )
    )
    main_info = _main_featurizer_info()
    nameless_info = _nameless_featurizer_info()
    expected_main_indices = sorted(
        {
            index
            for feature_name in main_info.features_to_use
            for index in main_info.feature_group_to_index[feature_name]
        }
    )
    expected_nameless_indices = sorted(
        {
            index
            for feature_name in nameless_info.features_to_use
            for index in nameless_info.feature_group_to_index[feature_name]
        }
    )
    expected_domains = set(unique["source_domain"].astype(str))
    observed_domains = {path.name for path in feature_root.iterdir() if path.is_dir()}
    if observed_domains != expected_domains:
        raise ValueError(
            f"feature-store domains mismatch: missing={sorted(expected_domains - observed_domains)}, "
            f"extra={sorted(observed_domains - expected_domains)}"
        )
    for domain, domain_pairs in unique.groupby("source_domain", sort=True):
        domain = str(domain)
        try:
            artifact_identity = input_identity["feature_artifacts"][domain]
        except KeyError as exc:
            raise ValueError(f"Input identity is missing feature artifacts for domain={domain!r}") from exc
        store = load_feature_store(
            feature_root / domain,
            expected_domain=domain,
            expected_pair_digest=pair_identity_digest(domain_pairs),
            expected_artifact_identity_digest=_json_digest(artifact_identity),
            expected_main_feature_indices=expected_main_indices,
            expected_nameless_feature_indices=expected_nameless_indices,
        )
        stores[domain] = store
    return stores


def arrays_for_catalog(
    selected: pd.DataFrame,
    stores: dict[str, DomainFeatureStore],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    canonical = canonicalize_pairs(selected.loc[:, PAIR_COLUMNS]).sort_values(
        ["source_domain", "pair1", "pair2", "label"],
        kind="stable",
    )
    main_parts = []
    nameless_parts = []
    label_parts = []
    for domain, frame in canonical.groupby("source_domain", sort=True):
        domain = str(domain)
        try:
            store = stores[domain]
        except KeyError as exc:
            raise ValueError(f"No feature store for source_domain={domain!r}") from exc
        indices = []
        expected_labels = []
        for _domain, _family, pair1, pair2, label, _rule, _origin, _group in frame.itertuples(index=False, name=None):
            key = (str(pair1), str(pair2))
            try:
                index = store.row_by_pair[key]
            except KeyError as exc:
                raise ValueError(f"Feature store domain={domain!r} is missing pair={key}") from exc
            indices.append(index)
            expected_labels.append(int(label))
        index_array = np.asarray(indices, dtype=np.int64)
        observed_labels = np.asarray(store.labels[index_array], dtype=np.int8)
        if not np.array_equal(observed_labels, np.asarray(expected_labels, dtype=np.int8)):
            raise ValueError(f"Feature-store label mismatch for source_domain={domain!r}")
        main_parts.append(np.asarray(store.main[index_array]))
        nameless_parts.append(np.asarray(store.nameless[index_array]))
        label_parts.append(observed_labels)
    return np.vstack(main_parts), np.vstack(nameless_parts), np.hstack(label_parts)


def evaluation_catalog(
    artifact: EvaluationArtifact,
    domain: str,
) -> tuple[pd.DataFrame, str, str]:
    """Return one domain's immutable pair evaluation rows and identities."""

    try:
        oracle = artifact.oracle_by_domain[domain]
        pair_digest = artifact.pair_digest_by_domain[domain]
    except KeyError as exc:
        raise ValueError(f"Unknown evaluation domain: {domain}") from exc
    selected = artifact.pairs.loc[artifact.pairs["source_domain"].eq(domain), PAIR_COLUMNS]
    if selected.empty:
        raise ValueError(f"No evaluation pairs for domain={domain!r}")
    return canonicalize_pairs(selected), oracle, pair_digest


def _load_public_gold(
    data_root: Path,
    public_domains: tuple[str, ...],
) -> dict[str, GoldBlockData]:
    return {
        domain: load_gold_block_data(
            domain,
            data_root / domain / "signatures.arrow",
            _gold_path(data_root, domain),
        )
        for domain in public_domains
    }


def _clusterer_for_models(models: Any, *, n_jobs: int) -> Clusterer:
    return Clusterer(
        _main_featurizer_info(),
        models.main,
        cluster_model=FastCluster(linkage="average"),
        n_jobs=n_jobs,
        use_cache=False,
        nameless_classifier=models.nameless,
        nameless_featurizer_info=_nameless_featurizer_info(),
    )


def _b3_metrics_for_fold(
    *,
    held_out_domain: str,
    clusterer: Clusterer,
    plans: dict[str, B3DomainEvaluationPlans],
    full_rust_featurizers: dict[str, Any],
    b3_cache_root: Path,
    input_identity: dict[str, Any],
    rust_version: str,
    rust_extension_sha256: str,
    cache_builder_identity: str,
    validated_b3_stores: MutableMapping[str, B3RawFeatureStore],
    config: ExperimentConfig,
) -> tuple[dict[str, Any], str, list[str]]:
    thresholds = np.linspace(0.3, 0.9, 61).tolist()
    calibration_domains = sorted(domain for domain in plans if domain != held_out_domain)
    fold_digest = _b3_fold_digest(plans, held_out_domain=held_out_domain, b3_scope=config.b3_scope)
    rust_featurizer_identity = {
        "adapter": "practice_only_legacy_arrow_rust_v1",
        "preprocess": False,
        "name_tuples": "filtered",
    }
    calibration_linkages = {}
    calibration_blocks = {}
    calibration_gold = {}
    cache_digests: list[str] = []
    for domain in calibration_domains:
        plan = plans[domain].calibration
        store = build_or_load_b3_raw_feature_store(
            b3_cache_root,
            plan=plan,
            rust_featurizer=full_rust_featurizers[domain],
            feature_artifact_identity=input_identity["feature_artifacts"][domain],
            rust_featurizer_identity=rust_featurizer_identity,
            clusterer=clusterer,
            rust_version=rust_version,
            rust_extension_sha256=rust_extension_sha256,
            cache_builder_identity=cache_builder_identity,
            validated_stores=validated_b3_stores,
        )
        distances = score_b3_raw_feature_store(
            store,
            clusterer=clusterer,
            total_ram_bytes=config.total_ram_gib * 1024**3,
        )
        cache_digests.append(store.cache_digest)
        selected_blocks = plan.blocks_dict()
        calibration_linkages[domain] = build_block_linkages(selected_blocks, distances)
        calibration_blocks[domain] = selected_blocks
        calibration_gold[domain] = plan.gold_dict()

    threshold, calibration_metrics = tune_b3_threshold(
        calibration_linkages,
        calibration_blocks,
        calibration_gold,
        thresholds,
    )
    heldout_plan = plans[held_out_domain].heldout
    heldout_store = build_or_load_b3_raw_feature_store(
        b3_cache_root,
        plan=heldout_plan,
        rust_featurizer=full_rust_featurizers[held_out_domain],
        feature_artifact_identity=input_identity["feature_artifacts"][held_out_domain],
        rust_featurizer_identity=rust_featurizer_identity,
        clusterer=clusterer,
        rust_version=rust_version,
        rust_extension_sha256=rust_extension_sha256,
        cache_builder_identity=cache_builder_identity,
        validated_stores=validated_b3_stores,
    )
    distances = score_b3_raw_feature_store(
        heldout_store,
        clusterer=clusterer,
        total_ram_bytes=config.total_ram_gib * 1024**3,
    )
    cache_digests.append(heldout_store.cache_digest)
    heldout_blocks = heldout_plan.blocks_dict()
    heldout_linkages = {held_out_domain: build_block_linkages(heldout_blocks, distances)}
    precision, recall, f1 = b3_for_threshold(
        heldout_linkages,
        {held_out_domain: heldout_blocks},
        {held_out_domain: heldout_plan.gold_dict()},
        threshold,
    )
    return (
        {
            "scope": config.b3_scope,
            "threshold": float(threshold),
            "threshold_calibration": calibration_metrics,
            "heldout_blocks": int(len(heldout_blocks)),
            "heldout_signatures": int(sum(len(values) for values in heldout_blocks.values())),
            "heldout_pairs": int(sum(len(values) * (len(values) - 1) // 2 for values in heldout_blocks.values())),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "scoring_backend": (
                "rust_pair_features_and_rust_lightgbm_cached_raw_features:" f"{B3_MEMBER_IDENTITY_VERSION}"
            ),
        },
        fold_digest,
        cache_digests,
    )


def _b3_fold_digest(
    plans: dict[str, B3DomainEvaluationPlans],
    *,
    held_out_domain: str,
    b3_scope: str,
) -> str:
    thresholds = np.linspace(0.3, 0.9, 61).tolist()
    calibration_domains = sorted(domain for domain in plans if domain != held_out_domain)
    return _json_digest(
        {
            "schema": "pair_ablation_b3_fold_v2",
            "held_out_domain": held_out_domain,
            "heldout_plan_digest": plans[held_out_domain].heldout.plan_digest,
            "calibration_plan_digests": {
                domain: plans[domain].calibration.plan_digest for domain in calibration_domains
            },
            "thresholds": thresholds,
            "scope": b3_scope,
            "member_identity_version": B3_MEMBER_IDENTITY_VERSION,
        }
    )


def verify_frozen_baseline(
    *,
    ranking_input_path: Path,
    output_dir: Path,
    training_catalog: pd.DataFrame,
    evaluation_artifact: EvaluationArtifact,
    b3_plans: dict[str, B3DomainEvaluationPlans],
    config: ExperimentConfig,
) -> dict[str, LoadedFold]:
    """Bind the current base and evaluations to the completed frozen baseline."""

    manifest_path = ranking_input_path.resolve()
    folds = load_ranking_input(manifest_path)
    baseline_folds = [
        fold
        for fold in folds
        if fold.payload["arm"] == BASELINE_ARM and fold.payload["training_seed"] == config.training_seed
    ]
    by_domain = {fold.payload["held_out_domain"]: fold for fold in baseline_folds}
    if len(by_domain) != len(baseline_folds) or set(by_domain) != set(config.fold_domains):
        raise ValueError(
            "frozen baseline ranking input does not contain exactly one matching baseline fold per domain: "
            f"expected={sorted(config.fold_domains)}, observed={sorted(by_domain)}"
        )

    baseline_arm = _select_arms((BASELINE_ARM,))[0]
    fold_records = []
    for domain in config.fold_domains:
        frozen = by_domain[domain]
        selected, _audit = catalog_for_arm(
            training_catalog,
            baseline_arm,
            held_out_domain=domain,
            random_seed=config.training_seed,
            linker_pairs_per_domain=config.linker_pairs_per_domain,
        )
        base_digest = pair_identity_digest(selected)
        if base_digest != frozen.payload["training_pair_digest"]:
            raise ValueError(
                "current uniform base differs from its frozen completed result: "
                f"seed={config.training_seed}, heldout={domain}, "
                f"current={base_digest}, frozen={frozen.payload['training_pair_digest']}"
            )
        _evaluation, oracle_kind, evaluation_pair_digest = evaluation_catalog(evaluation_artifact, domain)
        if oracle_kind != frozen.payload["pairwise"]["oracle_kind"]:
            raise ValueError(f"frozen baseline oracle kind differs for heldout={domain}")
        if evaluation_pair_digest != frozen.payload["evaluation_pair_digest"]:
            raise ValueError(f"frozen baseline pair evaluation identity differs for heldout={domain}")
        b3_evaluation_digest = (
            _b3_fold_digest(b3_plans, held_out_domain=domain, b3_scope=config.b3_scope)
            if domain in config.public_domains
            else None
        )
        if b3_evaluation_digest != frozen.payload["b3_evaluation_digest"]:
            raise ValueError(f"frozen baseline B3 evaluation identity differs for heldout={domain}")
        fold_records.append(
            {
                "held_out_domain": domain,
                "result_path": str(frozen.path),
                "result_sha256": frozen.sha256,
                "run_manifest_path": str(frozen.run_manifest_path),
                "run_manifest_sha256": frozen.run_manifest_sha256,
                "training_pair_digest": base_digest,
                "evaluation_pair_digest": evaluation_pair_digest,
                "b3_evaluation_digest": b3_evaluation_digest,
            }
        )

    verification = {
        "schema_version": "s2and_pair_ablation_frozen_baseline_verification_v1",
        "ranking_input_path": str(manifest_path),
        "ranking_input_sha256": sha256_file(manifest_path),
        "training_seed": config.training_seed,
        "evaluation_seed": config.evaluation_seed,
        "baseline_arm": BASELINE_ARM,
        "folds": fold_records,
    }
    _write_json_atomic(output_dir / "frozen_baseline_verification.json", verification)
    return by_domain


def verify_additive_recipe_assemblies(
    *,
    output_dir: Path,
    training_catalog: pd.DataFrame,
    arms: tuple[AblationArm, ...],
    frozen_baseline: dict[str, LoadedFold],
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Assemble every additive fold and prove base preservation before training."""

    records = []
    linker_keys: dict[tuple[str, str], set[tuple[str, str, str]]] = {}
    for arm in arms:
        additive = arm.additive_linker_recipe
        if additive is None:
            continue
        for domain in config.fold_domains:
            selected, audit = catalog_for_arm(
                training_catalog,
                arm,
                held_out_domain=domain,
                random_seed=config.training_seed,
                linker_pairs_per_domain=config.linker_pairs_per_domain,
            )
            frozen = frozen_baseline[domain]
            if audit["base_pair_digest"] != frozen.payload["training_pair_digest"]:
                raise AssertionError(f"preflight additive base digest mismatch arm={arm.name}, heldout={domain}")
            if len(selected) != audit["base_rows_after_lodo"] + audit["linker_selected_rows"]:
                raise AssertionError(f"preflight additive row reconciliation failed arm={arm.name}, heldout={domain}")
            selected_linker = selected.loc[selected["source_family"].isin(LINKER_FAMILIES), PAIR_COLUMNS]
            linker_keys[(arm.name, domain)] = {
                (str(source_domain), str(pair1), str(pair2))
                for source_domain, pair1, pair2 in selected_linker[
                    ["source_domain", "pair1", "pair2"]
                ].itertuples(index=False, name=None)
            }
            records.append(
                {
                    "arm": arm.name,
                    "held_out_domain": domain,
                    "source_set": additive.source_set,
                    "linker_cap_per_domain": additive.linker_pairs_per_domain,
                    "base_pair_digest": audit["base_pair_digest"],
                    "base_rows": audit["base_rows_after_lodo"],
                    "linker_rows": audit["linker_selected_rows"],
                    "final_rows": audit["final_rows"],
                    "linker_share": audit["linker_selected_rows"] / audit["final_rows"],
                    "linker_base_overlap_rows": audit["linker_base_overlap_rows"],
                }
            )

    by_source_set: dict[str, list[AblationArm]] = {}
    for arm in arms:
        if arm.additive_linker_recipe is not None:
            by_source_set.setdefault(arm.additive_linker_recipe.source_set, []).append(arm)
    for source_set, source_arms in by_source_set.items():
        ordered = sorted(source_arms, key=lambda arm: arm.additive_linker_recipe.linker_pairs_per_domain)  # type: ignore[union-attr]
        for lower, upper in zip(ordered, ordered[1:], strict=False):
            for domain in config.fold_domains:
                if not linker_keys[(lower.name, domain)].issubset(linker_keys[(upper.name, domain)]):
                    raise AssertionError(
                        "additive linker doses are not nested: "
                        f"source_set={source_set}, lower={lower.name}, upper={upper.name}, heldout={domain}"
                    )

    payload = {
        "schema_version": "s2and_pair_ablation_additive_recipe_preflight_v1",
        "training_seed": config.training_seed,
        "records": records,
    }
    _write_json_atomic(output_dir / "additive_recipe_preflight.json", payload)
    return payload


def _balanced_gold_pairs_for_arm(arm: AblationArm, config: ExperimentConfig) -> int | None:
    recipe = arm.exact_budget_recipe
    if recipe is None or recipe.balanced_gold_dose is None:
        return None
    return {
        "low": config.name_pairs_per_domain,
        "medium": config.balanced_medium_pairs_per_domain,
        "max": config.balanced_pool_pairs_per_domain,
    }[recipe.balanced_gold_dose]


def _arm_requires_recipe_audit(arm: AblationArm) -> bool:
    return arm.exact_budget_recipe is not None or arm.additive_linker_recipe is not None


def _recipe_metadata(arm: AblationArm, config: ExperimentConfig) -> dict[str, Any]:
    base_sampler = "uniform_100k"
    source_caps: dict[str, int] = {"uniform_pairs_per_domain": config.uniform_pairs_per_domain}
    if config.catalog_pool_cap_per_domain is not None:
        source_caps["catalog_pool_cap_per_domain"] = config.catalog_pool_cap_per_domain
    balanced_gold_pairs = _balanced_gold_pairs_for_arm(arm, config)
    if balanced_gold_pairs is not None:
        source_caps["balanced_pairs_per_domain"] = balanced_gold_pairs
        source_caps["balanced_pool_pairs_per_domain"] = config.balanced_pool_pairs_per_domain
    linker_families = (LINKER_PUBLIC_FAMILY, LINKER_BIG_POSITIVE_FAMILY, LINKER_PROXY_NEGATIVE_FAMILY)
    if any(family in arm.source_families for family in linker_families):
        source_caps["linker_pairs_per_domain"] = (
            arm.exact_budget_recipe.linker_pairs_per_domain
            if arm.exact_budget_recipe is not None and arm.exact_budget_recipe.linker_pairs_per_domain is not None
            else config.linker_pairs_per_domain
        )
    if arm.additive_linker_recipe is not None:
        additive = arm.additive_linker_recipe
        auxiliaries = [f"balanced_linker_{additive.source_set}"]
        source_caps["linker_pairs_per_domain"] = additive.linker_pairs_per_domain
        balancing = "linker_per_domain_shared_min_binary_no_backfill"
        budget_policy = "additive_to_unchanged_uniform_after_lodo"
        fixed_budget = False
        assembly_version = "additive_linker_lodo_v1"
    elif arm.exact_budget_recipe is None:
        auxiliaries = sorted(arm.source_families.difference({BASE_FAMILY}))
        balancing = "none"
        budget_policy = "additive"
        fixed_budget = False
        assembly_version = "additive_lodo_v1"
    else:
        auxiliaries = list(arm.exact_budget_recipe.auxiliary_families)
        if arm.exact_budget_recipe.balanced_linker:
            auxiliaries.append("balanced_linker")
        if arm.exact_budget_recipe.capped_proxy_negative:
            auxiliaries.append("capped_proxy_negative")
        balancing_policies = []
        if arm.exact_budget_recipe.balanced_gold_dose is not None:
            balancing_policies.append("balanced_gold_per_domain_label_hash_prefix_no_backfill")
        if arm.exact_budget_recipe.balanced_linker:
            balancing_policies.append("linker_per_domain_shared_min_binary_no_backfill")
        elif arm.exact_budget_recipe.capped_proxy_negative:
            balancing_policies.append("proxy_per_domain_deterministic_cap_negative_only_no_backfill")
        if not balancing_policies:
            balancing_policies.append("explicit_auxiliary_sources")
        balancing = "+".join(balancing_policies)
        budget_policy = "exact_uniform_after_lodo"
        fixed_budget = True
        assembly_version = "exact_budget_v1"
    return {
        "arm": arm.name,
        "assembly_version": assembly_version,
        "auxiliary_sources": auxiliaries,
        "balancing": balancing,
        "base_sampler": base_sampler,
        "budget_policy": budget_policy,
        "complexity_rank": len(auxiliaries),
        "fixed_budget": fixed_budget,
        "source_caps": source_caps,
    }


def _result_path(output_dir: Path, arm: str, held_out_domain: str) -> Path:
    return output_dir / "results" / arm / f"{held_out_domain}.json"


def _load_cached_models(
    cache_dir: Path,
    *,
    run_id: str,
    training_pair_digest: str,
    training_rows: int,
    n_jobs: int,
) -> tuple[Any, dict[str, Any]] | None:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = load_strict_json(manifest_path)
    if set(manifest) != {"schema_version", "run_id", "training_pair_digest", "training_rows", "models"}:
        raise RuntimeError(f"Cached model manifest has an invalid schema: {manifest_path}")
    if manifest.get("schema_version") != MODEL_CACHE_SCHEMA_VERSION:
        raise RuntimeError(f"Cached model manifest has an unsupported version: {manifest_path}")
    if (
        manifest.get("run_id") != run_id
        or manifest.get("training_pair_digest") != training_pair_digest
        or manifest.get("training_rows") != training_rows
    ):
        raise RuntimeError(f"Cached models belong to a different run or training set: {cache_dir}")
    metadata = manifest.get("models")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Cached model manifest has no model metadata: {manifest_path}")
    for name in ("main", "nameless"):
        model_path = cache_dir / f"{name}.lgb"
        try:
            expected_sha = metadata[name]["model_sha256"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Cached model manifest is malformed: {manifest_path}") from exc
        if Path(metadata[name].get("model_path", "")).resolve() != model_path.resolve():
            raise RuntimeError(f"Cached model path does not match its manifest: {model_path}")
        if not model_path.is_file() or _sha256_file(model_path) != expected_sha:
            raise RuntimeError(f"Cached model content does not match its manifest: {model_path}")
    return load_pairwise_models(cache_dir, n_jobs=n_jobs), metadata


def run_ablation(
    training_catalog: pd.DataFrame,
    evaluation_artifact: EvaluationArtifact,
    *,
    data_root: Path,
    linker_bundle_root: Path,
    donor_model_dir: Path,
    output_dir: Path,
    config: ExperimentConfig,
    arms: tuple[AblationArm, ...],
    run_id: str,
    input_identity: dict[str, Any],
    b3_cache_root: Path,
    rust_version: str,
    rust_extension_sha256: str,
    cache_builder_identity: str,
    resume: bool,
    feature_stores: dict[str, DomainFeatureStore] | None = None,
    frozen_baseline: dict[str, LoadedFold] | None = None,
    prepared_b3_plans: dict[str, B3DomainEvaluationPlans] | None = None,
) -> None:
    """Train, score, and checkpoint every requested arm/fold."""

    stores = (
        feature_stores
        if feature_stores is not None
        else load_feature_stores(
            output_dir,
            training_catalog=training_catalog,
            evaluation_catalog=evaluation_artifact.pairs,
            input_identity=input_identity,
        )
    )
    public_gold = _load_public_gold(data_root, config.public_domains)
    b3_plans = (
        prepared_b3_plans
        if prepared_b3_plans is not None
        else build_b3_evaluation_plans(
            public_gold,
            evaluation_seed=config.evaluation_seed,
            threshold_pairs_per_domain=config.threshold_pairs_per_domain,
            b3_scope=config.b3_scope,
        )
    )
    full_rust_featurizers: dict[str, Any] = {}
    validated_b3_stores: dict[str, B3RawFeatureStore] = {}

    def public_featurizer(domain: str) -> Any:
        if domain not in full_rust_featurizers:
            artifacts = resolve_legacy_arrow_manifest(_standard_manifest(data_root, domain))
            logger.info("building full-domain Rust featurizer for B3 domain=%s", domain)
            full_rust_featurizers[domain] = build_legacy_rust_featurizer(
                artifacts,
                n_jobs=config.n_jobs,
                signature_ids=None,
            )
        return full_rust_featurizers[domain]

    for arm in arms:
        recipe = _recipe_metadata(arm, config)
        recipe_id = recipe_id_for(recipe)
        for held_out_domain in config.fold_domains:
            result_path = _result_path(output_dir, arm.name, held_out_domain)
            selected, assembly_audit = catalog_for_arm(
                training_catalog,
                arm,
                held_out_domain=held_out_domain,
                random_seed=config.training_seed,
                linker_pairs_per_domain=config.linker_pairs_per_domain,
                balanced_pairs_per_domain=_balanced_gold_pairs_for_arm(arm, config),
                balanced_pool_pairs_per_domain=config.balanced_pool_pairs_per_domain,
            )
            if arm.additive_linker_recipe is not None and frozen_baseline is not None:
                frozen = frozen_baseline[held_out_domain]
                if assembly_audit["base_pair_digest"] != frozen.payload["training_pair_digest"]:
                    raise AssertionError(
                        "additive linker base digest differs from the frozen production base: "
                        f"arm={arm.name}, heldout={held_out_domain}"
                    )
                assembly_audit = {
                    **assembly_audit,
                    "frozen_baseline_result_path": str(frozen.path),
                    "frozen_baseline_result_sha256": frozen.sha256,
                    "frozen_baseline_run_manifest_path": str(frozen.run_manifest_path),
                    "frozen_baseline_run_manifest_sha256": frozen.run_manifest_sha256,
                }
            training_pair_digest = pair_identity_digest(selected)
            evaluation, oracle_kind, evaluation_pair_digest = evaluation_catalog(
                evaluation_artifact,
                held_out_domain,
            )
            b3_evaluation_digest = (
                _b3_fold_digest(
                    b3_plans,
                    held_out_domain=held_out_domain,
                    b3_scope=config.b3_scope,
                )
                if held_out_domain in public_gold
                else None
            )
            expected_result = FoldResultExpectation(
                run_id=run_id,
                arm=arm.name,
                source_families=tuple(sorted(arm.source_families)),
                held_out_domain=held_out_domain,
                training_seed=config.training_seed,
                evaluation_seed=config.evaluation_seed,
                recipe=recipe,
                training_pair_digest=training_pair_digest,
                evaluation_pair_digest=evaluation_pair_digest,
                b3_evaluation_digest=b3_evaluation_digest,
                oracle_kind=oracle_kind,
                b3_scope=config.b3_scope if held_out_domain in public_gold else None,
                requires_recipe_audit=_arm_requires_recipe_audit(arm),
            )
            if resume and result_path.exists():
                load_fold_result(result_path, expected=expected_result)
                logger.info("result already complete arm=%s heldout=%s", arm.name, held_out_domain)
                continue
            started = time.perf_counter()
            logger.info("training arm=%s heldout=%s", arm.name, held_out_domain)
            model_dir = output_dir / "models" / "by_training_pair_digest" / training_pair_digest
            cached = _load_cached_models(
                model_dir,
                run_id=run_id,
                training_pair_digest=training_pair_digest,
                training_rows=len(selected),
                n_jobs=config.n_jobs,
            )
            model_cache_hit = cached is not None
            main_train = nameless_train = y_train = None
            if cached is None:
                main_train, nameless_train, y_train = arrays_for_catalog(selected, stores)
                models = train_pairwise_models(
                    main_train,
                    nameless_train,
                    y_train,
                    main_featurizer_info=_main_featurizer_info(),
                    nameless_featurizer_info=_nameless_featurizer_info(),
                    donor_model_dir=donor_model_dir,
                    output_dir=model_dir,
                    n_jobs=config.n_jobs,
                    random_seed=config.training_seed,
                    estimator_scale=config.estimator_scale,
                )
                model_metadata = models.metadata
                _write_json_atomic(
                    model_dir / "manifest.json",
                    {
                        "schema_version": MODEL_CACHE_SCHEMA_VERSION,
                        "run_id": run_id,
                        "training_pair_digest": training_pair_digest,
                        "training_rows": int(len(selected)),
                        "models": model_metadata,
                    },
                )
            else:
                models, model_metadata = cached
            main_eval, nameless_eval, y_eval = arrays_for_catalog(evaluation, stores)
            probability = averaged_positive_probability(models, main_eval, nameless_eval)
            metrics = pairwise_metrics(y_eval, probability, oracle_kind=oracle_kind)

            b3_metrics = None
            b3_cache_digests: list[str] = []
            if held_out_domain in public_gold:
                # Calibration must never touch the held-out domain.
                for domain in public_gold:
                    if domain != held_out_domain:
                        public_featurizer(domain)
                public_featurizer(held_out_domain)
                clusterer = _clusterer_for_models(models, n_jobs=config.n_jobs)
                b3_metrics, observed_b3_digest, b3_cache_digests = _b3_metrics_for_fold(
                    held_out_domain=held_out_domain,
                    clusterer=clusterer,
                    plans=b3_plans,
                    full_rust_featurizers=full_rust_featurizers,
                    b3_cache_root=b3_cache_root,
                    input_identity=input_identity,
                    rust_version=rust_version,
                    rust_extension_sha256=rust_extension_sha256,
                    cache_builder_identity=cache_builder_identity,
                    validated_b3_stores=validated_b3_stores,
                    config=config,
                )
                if observed_b3_digest != b3_evaluation_digest:
                    raise AssertionError("B3 evaluation identity changed during fold execution")

            result = {
                "schema_version": FOLD_RESULT_SCHEMA_VERSION,
                "run_id": run_id,
                "recipe_id": recipe_id,
                "recipe": recipe,
                "arm": arm.name,
                "source_families": sorted(arm.source_families),
                "held_out_domain": held_out_domain,
                "training_seed": config.training_seed,
                "evaluation_seed": config.evaluation_seed,
                "evaluation_pair_digest": evaluation_pair_digest,
                "b3_evaluation_digest": b3_evaluation_digest,
                "training_pair_digest": training_pair_digest,
                "model_cache_hit": model_cache_hit,
                "training_rows": int(len(selected)),
                "training_positives": int(selected["label"].sum()),
                "training_negatives": int(len(selected) - selected["label"].sum()),
                "training_source_counts": source_counts(selected),
                "pair_recipe_assembly": (assembly_audit if _arm_requires_recipe_audit(arm) else None),
                "pairwise": metrics,
                "b3": b3_metrics,
                "b3_cache_digests": b3_cache_digests,
                "models": model_metadata,
                "elapsed_seconds": float(time.perf_counter() - started),
            }
            write_fold_result(result_path, result, expected=expected_result)
            print(
                json.dumps(
                    {
                        "event": "fold_complete",
                        "arm": arm.name,
                        "held_out_domain": held_out_domain,
                        "auroc": metrics["auroc"],
                        "auprc": metrics["auprc"],
                        "b3_f1": None if b3_metrics is None else b3_metrics["f1"],
                        "elapsed_seconds": result["elapsed_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del main_train, nameless_train, y_train, main_eval, nameless_eval, y_eval, probability, models
            gc.collect()


def summarize_results(
    *,
    output_dir: Path,
    arms: tuple[AblationArm, ...],
    config: ExperimentConfig,
    run_id: str,
    training_catalog: pd.DataFrame,
    evaluation_artifact: EvaluationArtifact,
    b3_plans: dict[str, B3DomainEvaluationPlans],
) -> dict[str, Any]:
    rows = []
    missing = []
    for arm in arms:
        recipe = _recipe_metadata(arm, config)
        for domain in config.fold_domains:
            path = _result_path(output_dir, arm.name, domain)
            if not path.exists():
                missing.append({"arm": arm.name, "held_out_domain": domain})
                continue
            selected, _audit = catalog_for_arm(
                training_catalog,
                arm,
                held_out_domain=domain,
                random_seed=config.training_seed,
                linker_pairs_per_domain=config.linker_pairs_per_domain,
                balanced_pairs_per_domain=_balanced_gold_pairs_for_arm(arm, config),
                balanced_pool_pairs_per_domain=config.balanced_pool_pairs_per_domain,
            )
            _evaluation, oracle_kind, evaluation_pair_digest = evaluation_catalog(evaluation_artifact, domain)
            b3_evaluation_digest = (
                _b3_fold_digest(b3_plans, held_out_domain=domain, b3_scope=config.b3_scope)
                if domain in config.public_domains
                else None
            )
            result = load_fold_result(
                path,
                expected=FoldResultExpectation(
                    run_id=run_id,
                    arm=arm.name,
                    source_families=tuple(sorted(arm.source_families)),
                    held_out_domain=domain,
                    training_seed=config.training_seed,
                    evaluation_seed=config.evaluation_seed,
                    recipe=recipe,
                    training_pair_digest=pair_identity_digest(selected),
                    evaluation_pair_digest=evaluation_pair_digest,
                    b3_evaluation_digest=b3_evaluation_digest,
                    oracle_kind=oracle_kind,
                    b3_scope=config.b3_scope if domain in config.public_domains else None,
                    requires_recipe_audit=_arm_requires_recipe_audit(arm),
                ),
            )
            pair = result["pairwise"]
            b3 = result.get("b3")
            rows.append(
                {
                    "arm": arm.name,
                    "held_out_domain": domain,
                    "oracle_kind": pair["oracle_kind"],
                    "pair_rows": pair["rows"],
                    "prevalence": pair["prevalence"],
                    "auroc": pair["auroc"],
                    "auprc": pair["auprc"],
                    "b3_f1": None if b3 is None else b3["f1"],
                    "b3_precision": None if b3 is None else b3["precision"],
                    "b3_recall": None if b3 is None else b3["recall"],
                    "training_rows": result["training_rows"],
                    "elapsed_seconds": result["elapsed_seconds"],
                }
            )
    metrics = pd.DataFrame(rows)
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(summary_dir / "heldout_metrics.csv", index=False)

    aggregates = []
    gold_domains = set(config.public_domains) | {"medline"}
    for arm in arms:
        arm_rows = metrics.loc[metrics["arm"].eq(arm.name)]
        gold = arm_rows.loc[arm_rows["held_out_domain"].isin(gold_domains)]
        public = arm_rows.loc[arm_rows["held_out_domain"].isin(config.public_domains)]
        medline = arm_rows.loc[arm_rows["held_out_domain"].eq("medline")]
        proxy = arm_rows.loc[arm_rows["oracle_kind"].eq("linker_component_proxy")]
        b3 = arm_rows.loc[arm_rows["b3_f1"].notna()]
        aggregates.append(
            {
                "arm": arm.name,
                "completed_folds": int(len(arm_rows)),
                "mean_gold_auroc": None if gold.empty else float(gold["auroc"].mean()),
                "mean_gold_auprc": None if gold.empty else float(gold["auprc"].mean()),
                "worst_gold_auprc": None if gold.empty else float(gold["auprc"].min()),
                "mean_public_auroc": None if public.empty else float(public["auroc"].mean()),
                "mean_public_auprc": None if public.empty else float(public["auprc"].mean()),
                "worst_public_auprc": None if public.empty else float(public["auprc"].min()),
                "medline_auroc": None if medline.empty else float(medline.iloc[0]["auroc"]),
                "medline_auprc": None if medline.empty else float(medline.iloc[0]["auprc"]),
                "mean_proxy_auroc": None if proxy.empty else float(proxy["auroc"].mean()),
                "mean_proxy_auprc": None if proxy.empty else float(proxy["auprc"].mean()),
                "mean_b3_f1": None if b3.empty else float(b3["b3_f1"].mean()),
                "worst_b3_f1": None if b3.empty else float(b3["b3_f1"].min()),
            }
        )
    aggregate_frame = pd.DataFrame(aggregates)
    aggregate_frame.to_csv(summary_dir / "arm_summary.csv", index=False)
    complete = not missing

    payload = {
        "run_id": run_id,
        "complete": complete,
        "study_stage": "descriptive_single_seed_screening",
        "missing": missing,
        "arms": aggregates,
        "confirmation_required": (
            "Treat this as screening. Confirm the baseline and top recipes over at least three sampling seeds, "
            "then compare paired held-out-domain deltas and a second linker dose before selecting a release recipe."
        ),
        "proxy_warning": (
            "Big-block AUROC/AUPRC uses exact-ORCID positives plus label-0 component-member negative proxies; "
            "it is not gold and is not pooled with public/Medline pair metrics."
        ),
    }
    _write_json_atomic(summary_dir / "summary.json", payload)
    lines = [
        "# Pair-source ablation summary",
        "",
        f"- complete: `{complete}`",
        f"- run id: `{run_id}`",
        "",
        "This summary is descriptive only. The separate paired multi-seed ranker is the sole release-selection "
        "authority.",
        "",
        "Medline is reported separately from the public-domain macro metrics. Big-block pair metrics are a "
        "component-label proxy, not gold, and are excluded from ranking.",
        "",
        _markdown_table(aggregate_frame),
        "",
    ]
    (summary_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def write_run_ranking_input(
    *,
    output_dir: Path,
    arms: tuple[AblationArm, ...],
    config: ExperimentConfig,
    run_id: str,
    training_catalog: pd.DataFrame,
    evaluation_artifact: EvaluationArtifact,
    b3_plans: dict[str, B3DomainEvaluationPlans],
) -> Path:
    """Write strict independent expectations/hashes for this run's final ranker."""

    run_manifest_path = (output_dir / "run_manifest.json").resolve()
    run_manifest = load_run_manifest(run_manifest_path)
    if run_manifest["run_id"] != run_id:
        raise ValueError(
            f"Ranking input run_id does not match {run_manifest_path}: "
            f"expected={run_id}, observed={run_manifest['run_id']}"
        )
    run_manifest_sha256 = sha256_file(run_manifest_path)
    folds: list[dict[str, Any]] = []
    for arm in arms:
        recipe = _recipe_metadata(arm, config)
        for domain in config.fold_domains:
            result_path = _result_path(output_dir, arm.name, domain).resolve()
            if not result_path.is_file():
                raise FileNotFoundError(f"Cannot build ranking input; fold is missing: {result_path}")
            selected, _audit = catalog_for_arm(
                training_catalog,
                arm,
                held_out_domain=domain,
                random_seed=config.training_seed,
                linker_pairs_per_domain=config.linker_pairs_per_domain,
                balanced_pairs_per_domain=_balanced_gold_pairs_for_arm(arm, config),
                balanced_pool_pairs_per_domain=config.balanced_pool_pairs_per_domain,
            )
            _evaluation, oracle_kind, evaluation_pair_digest = evaluation_catalog(evaluation_artifact, domain)
            b3_evaluation_digest = (
                _b3_fold_digest(b3_plans, held_out_domain=domain, b3_scope=config.b3_scope)
                if domain in config.public_domains
                else None
            )
            expectation = {
                "run_id": run_id,
                "arm": arm.name,
                "source_families": sorted(arm.source_families),
                "held_out_domain": domain,
                "training_seed": config.training_seed,
                "evaluation_seed": config.evaluation_seed,
                "recipe": recipe,
                "training_pair_digest": pair_identity_digest(selected),
                "evaluation_pair_digest": evaluation_pair_digest,
                "b3_evaluation_digest": b3_evaluation_digest,
                "oracle_kind": oracle_kind,
                "b3_scope": config.b3_scope if domain in config.public_domains else None,
                "requires_recipe_audit": _arm_requires_recipe_audit(arm),
            }
            folds.append(
                {
                    "path": str(result_path),
                    "result_sha256": sha256_file(result_path),
                    "run_manifest_path": str(run_manifest_path),
                    "run_manifest_sha256": run_manifest_sha256,
                    "expected": expectation,
                }
            )
    path = output_dir / "ranking_input.json"
    _write_json_atomic(
        path,
        {
            "schema_version": RANKING_INPUT_SCHEMA_VERSION,
            "folds": folds,
        },
    )
    return path


def _persist_summary_with_ranking_input(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    ranking_input_path: Path,
) -> dict[str, Any]:
    """Persist and return the same completed summary payload printed by the CLI."""

    updated = {**summary, "ranking_input_path": str(ranking_input_path)}
    _write_json_atomic(output_dir / "summary" / "summary.json", updated)
    return updated


def _select_arms(names: tuple[str, ...]) -> tuple[AblationArm, ...]:
    by_name = {arm.name: arm for arm in ablation_arm_registry()}
    selected: list[AblationArm] = []
    unknown: list[str] = []
    for name in names:
        registered = by_name.get(name)
        if registered is not None:
            selected.append(registered)
            continue
        match = _ADDITIVE_LINKER_ARM_PATTERN.fullmatch(name)
        if match is None:
            unknown.append(name)
            continue
        source_set, raw_dose = match.groups()
        selected.append(additive_linker_arm(source_set, int(raw_dose)))  # type: ignore[arg-type]
    if unknown:
        raise ValueError(f"Unknown ablation arms: {sorted(unknown)}")
    return tuple(selected)


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if args.smoke:
        public = tuple(args.public_domains or ("pubmed", "qian"))
        big = tuple(args.big_block_domains or ("h_wang",))
        folds = tuple(args.fold_domains or (*public, "medline", *big))
        arm_names = tuple(args.arms or (arm.name for arm in default_ablation_arms()))
        return ExperimentConfig(
            training_seed=args.training_seed,
            evaluation_seed=args.evaluation_seed,
            n_jobs=args.n_jobs,
            total_ram_gib=args.total_ram_gib,
            uniform_pairs_per_domain=min(args.uniform_pairs_per_domain, 200),
            name_pairs_per_domain=min(args.name_pairs_per_domain, 40),
            balanced_medium_pairs_per_domain=min(args.balanced_medium_pairs_per_domain, 100),
            balanced_pool_pairs_per_domain=min(args.balanced_pool_pairs_per_domain, 200),
            linker_pairs_per_domain=min(args.linker_pairs_per_domain, 100),
            big_proxy_eval_pairs_per_class=min(args.big_proxy_eval_pairs_per_class, 100),
            catalog_pool_cap_per_domain=(
                50 if args.catalog_pool_cap_per_domain is None else min(args.catalog_pool_cap_per_domain, 50)
            ),
            eval_pairs_per_domain=min(args.eval_pairs_per_domain, 200),
            threshold_pairs_per_domain=min(args.threshold_pairs_per_domain, 200),
            estimator_scale=min(args.estimator_scale, 0.002),
            b3_scope="test",
            public_domains=public,
            big_block_domains=big,
            fold_domains=folds,
            arm_names=arm_names,
        )
    public = tuple(args.public_domains or PUBLIC_DOMAINS)
    big = tuple(args.big_block_domains or BIG_BLOCK_DOMAINS)
    folds = tuple(args.fold_domains or (*public, "medline", *big))
    arm_names = tuple(args.arms or (arm.name for arm in default_ablation_arms()))
    return ExperimentConfig(
        training_seed=args.training_seed,
        evaluation_seed=args.evaluation_seed,
        n_jobs=args.n_jobs,
        total_ram_gib=args.total_ram_gib,
        uniform_pairs_per_domain=args.uniform_pairs_per_domain,
        name_pairs_per_domain=args.name_pairs_per_domain,
        balanced_medium_pairs_per_domain=args.balanced_medium_pairs_per_domain,
        balanced_pool_pairs_per_domain=args.balanced_pool_pairs_per_domain,
        linker_pairs_per_domain=args.linker_pairs_per_domain,
        big_proxy_eval_pairs_per_class=args.big_proxy_eval_pairs_per_class,
        catalog_pool_cap_per_domain=args.catalog_pool_cap_per_domain,
        eval_pairs_per_domain=args.eval_pairs_per_domain,
        threshold_pairs_per_domain=args.threshold_pairs_per_domain,
        estimator_scale=args.estimator_scale,
        b3_scope=args.b3_scope,
        public_domains=public,
        big_block_domains=big,
        fold_domains=folds,
        arm_names=arm_names,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("catalog", "features", "run", "summarize", "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "s2and" / "data")
    parser.add_argument("--backup-data-root", type=Path, default=REPO_ROOT / "s2and" / "data-backup")
    parser.add_argument(
        "--linker-bundle-root",
        type=Path,
        default=REPO_ROOT / "s2and" / "data" / "s2and_and_big_blocks_linker_dataset_20260525",
    )
    parser.add_argument(
        "--donor-model-dir",
        type=Path,
        default=REPO_ROOT / "s2and" / "data" / "production_model_v1.21" / "pairwise",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "scratch" / "pair_source_ablation_20260710")
    parser.add_argument("--training-seed", type=int, default=1111)
    parser.add_argument("--evaluation-seed", type=int, default=1111)
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--total-ram-gib", type=int, default=200)
    parser.add_argument("--uniform-pairs-per-domain", type=int, default=100_000)
    parser.add_argument("--name-pairs-per-domain", type=int, default=10_000)
    parser.add_argument("--balanced-medium-pairs-per-domain", type=int, default=50_000)
    parser.add_argument("--balanced-pool-pairs-per-domain", type=int, default=100_000)
    parser.add_argument("--linker-pairs-per-domain", type=int, default=10_000)
    parser.add_argument("--big-proxy-eval-pairs-per-class", type=int, default=10_000)
    parser.add_argument("--catalog-pool-cap-per-domain", type=int, default=None)
    parser.add_argument("--eval-pairs-per-domain", type=int, default=100_000)
    parser.add_argument("--threshold-pairs-per-domain", type=int, default=100_000)
    parser.add_argument("--estimator-scale", type=float, default=1.0)
    parser.add_argument("--b3-scope", choices=("test", "full"), default="test")
    parser.add_argument("--public-domains", nargs="*", default=None)
    parser.add_argument("--big-block-domains", nargs="*", default=None)
    parser.add_argument("--fold-domains", nargs="*", default=None)
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--b3-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--reuse-artifacts-from",
        type=Path,
        default=None,
        help="Reuse and strictly validate a seed-matched catalog/evaluation/features directory.",
    )
    parser.add_argument(
        "--frozen-baseline-ranking-input",
        type=Path,
        default=None,
        help="Completed ranking input whose uniform_100k folds define the immutable additive base.",
    )
    parser.add_argument(
        "--expected-comparison-identity-sha256",
        type=str,
        default=None,
        help="Fail unless the prospective full-run comparison identity equals this SHA-256.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate manifest, reused artifacts, and frozen baseline without training folds.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-full", action="store_true")
    return parser


def _validate_config(config: ExperimentConfig) -> None:
    if config.n_jobs <= 0 or config.total_ram_gib <= 0:
        raise ValueError("n_jobs and total_ram_gib must be positive")
    for field in (
        "uniform_pairs_per_domain",
        "name_pairs_per_domain",
        "balanced_medium_pairs_per_domain",
        "balanced_pool_pairs_per_domain",
        "linker_pairs_per_domain",
        "big_proxy_eval_pairs_per_class",
        "eval_pairs_per_domain",
        "threshold_pairs_per_domain",
    ):
        if getattr(config, field) <= 0:
            raise ValueError(f"{field} must be positive")
    balanced_doses = (
        config.name_pairs_per_domain,
        config.balanced_medium_pairs_per_domain,
        config.balanced_pool_pairs_per_domain,
    )
    if not balanced_doses[0] < balanced_doses[1] < balanced_doses[2]:
        raise ValueError("balanced gold doses must satisfy low < medium < max pool")
    if any(dose % 2 for dose in balanced_doses):
        raise ValueError("balanced gold doses must be even for exact 50/50 class quotas")
    if config.balanced_pool_pairs_per_domain > config.uniform_pairs_per_domain:
        raise ValueError("balanced max pool must fit the B-only uniform per-domain budget")
    if config.estimator_scale <= 0:
        raise ValueError("estimator_scale must be positive")
    if config.training_seed < 0 or config.evaluation_seed < 0:
        raise ValueError("training_seed and evaluation_seed must be non-negative")
    if config.catalog_pool_cap_per_domain is not None and config.catalog_pool_cap_per_domain <= 0:
        raise ValueError("catalog_pool_cap_per_domain must be positive when provided")
    if config.b3_scope not in {"test", "full"}:
        raise ValueError("b3_scope must be 'test' or 'full'")
    for field in ("public_domains", "big_block_domains", "fold_domains", "arm_names"):
        values = getattr(config, field)
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{field} must be non-empty and contain no duplicates")
    if not set(config.public_domains).issubset(PUBLIC_DOMAINS):
        raise ValueError("public_domains contains an unsupported domain")
    if not set(config.big_block_domains).issubset(BIG_BLOCK_DOMAINS):
        raise ValueError("big_block_domains contains an unsupported domain")
    available_folds = set(config.public_domains) | {"medline"} | set(config.big_block_domains)
    if not set(config.fold_domains).issubset(available_folds):
        raise ValueError("fold_domains must be selected public, Medline, or selected big-block domains")
    if set(config.fold_domains).intersection(config.public_domains) and len(config.public_domains) < 2:
        raise ValueError("B3 evaluation requires at least two selected public domains for held-out calibration")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("s2and").setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    if not args.smoke and not args.run_full:
        raise SystemExit("Full ablation is expensive; pass --run-full explicitly (or use --smoke).")
    config = _config_from_args(args)
    _validate_config(config)
    arms = _select_arms(config.arm_names)
    additive_arms = tuple(arm for arm in arms if arm.additive_linker_recipe is not None)
    if additive_arms and not args.smoke and args.frozen_baseline_ranking_input is None:
        raise SystemExit("Full additive linker arms require --frozen-baseline-ranking-input.")
    if args.expected_comparison_identity_sha256 is not None and re.fullmatch(
        r"[0-9a-f]{64}",
        args.expected_comparison_identity_sha256,
    ) is None:
        raise SystemExit("--expected-comparison-identity-sha256 must be 64 lowercase hexadecimal characters.")
    data_root = args.data_root.resolve()
    backup_data_root = args.backup_data_root.resolve()
    linker_bundle_root = args.linker_bundle_root.resolve()
    donor_model_dir = args.donor_model_dir.resolve()
    output_dir = args.output_dir.resolve()
    b3_cache_root = (
        args.b3_cache_dir.resolve()
        if args.b3_cache_dir is not None
        else (output_dir.parent / "pair_source_ablation_b3_raw_cache").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    input_identity = build_input_identity(
        data_root=data_root,
        backup_data_root=backup_data_root,
        linker_bundle_root=linker_bundle_root,
        config=config,
    )
    rust_version = str(__import__("s2and_rust").__version__)
    rust_extension_sha256 = rust_extension_binary_sha256()
    implementation_sha256 = _implementation_identity()
    runtime_versions = current_runtime_versions()
    git_identity = _git_identity()
    b3_builder_identity = (
        validated_reused_b3_builder_identity(
            artifact_source_dir=args.reuse_artifacts_from,
            output_dir=output_dir,
            current_implementation_sha256=implementation_sha256,
            current_runtime_versions=runtime_versions,
            current_rust_version=rust_version,
            current_rust_extension_sha256=rust_extension_sha256,
        )
        if args.reuse_artifacts_from is not None
        else b3_cache_builder_identity(
            implementation_sha256=implementation_sha256,
            runtime_versions=runtime_versions,
        )
    )
    run_input_identity = {
        **input_identity,
        "b3_cache_builder_identity": b3_builder_identity,
    }
    recipes = [
        {"recipe_id": recipe_id_for(recipe), "recipe": recipe}
        for recipe in (_recipe_metadata(arm, config) for arm in arms)
    ]
    run_payload = build_run_manifest(
        {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "config": asdict(config),
            "recipes": recipes,
            "input_identity": run_input_identity,
            "git": git_identity,
            "implementation_sha256": implementation_sha256,
            "featurizer_version": FEATURIZER_VERSION,
            "rust_version": rust_version,
            "rust_extension_sha256": rust_extension_sha256,
            "runtime_versions": runtime_versions,
            "donor_model_sha256": {
                name: _sha256_file(donor_model_dir / f"{name}.lgb") for name in ("main", "nameless")
            },
            "adapter": "practice_only_legacy_arrow_rust_v1",
            "warning": "Never promote these legacy-artifact models. Repeat the winner on canonical_v2.",
            "thread_environment": {key: os.environ.get(key) for key in THREAD_ENVIRONMENT_KEYS},
        }
    )
    run_id = run_payload["run_id"]
    comparison_identity_sha256 = run_payload["comparison_identity"]["sha256"]
    if (
        args.expected_comparison_identity_sha256 is not None
        and comparison_identity_sha256 != args.expected_comparison_identity_sha256
    ):
        raise RuntimeError(
            "Prospective pair-ablation comparison identity mismatch: "
            f"expected={args.expected_comparison_identity_sha256}, observed={comparison_identity_sha256}"
        )
    run_manifest_path = output_dir / "run_manifest.json"
    if run_manifest_path.exists():
        existing = load_run_manifest(run_manifest_path)
        if existing["run_id"] != run_id:
            raise RuntimeError(f"Output directory belongs to a different run: {output_dir}")
    else:
        _write_json_atomic(run_manifest_path, run_payload)

    feature_stores: dict[str, DomainFeatureStore] | None = None
    if args.reuse_artifacts_from is not None:
        training_catalog, evaluation_artifact, feature_stores = load_reused_pair_artifacts(
            artifact_source_dir=args.reuse_artifacts_from,
            output_dir=output_dir,
            config=config,
            input_identity=input_identity,
        )
    else:
        catalog_path = output_dir / "catalog" / "training_pairs.parquet"
        evaluation_path = output_dir / "evaluation" / "pairs.parquet"
        if args.phase in {"catalog", "all"}:
            training_catalog, evaluation_artifact = build_pair_catalogs(
                data_root=data_root,
                backup_data_root=backup_data_root,
                linker_bundle_root=linker_bundle_root,
                output_dir=output_dir,
                config=config,
                input_identity=input_identity,
                resume=args.resume,
            )
        else:
            if not catalog_path.exists() or not evaluation_path.exists():
                raise FileNotFoundError(
                    f"Pair catalogs have not been built: training={catalog_path}, evaluation={evaluation_path}"
                )
            training_catalog, evaluation_artifact = build_pair_catalogs(
                data_root=data_root,
                backup_data_root=backup_data_root,
                linker_bundle_root=linker_bundle_root,
                output_dir=output_dir,
                config=config,
                input_identity=input_identity,
                resume=True,
            )

    if args.reuse_artifacts_from is None and args.phase in {"features", "all"}:
        build_feature_stores(
            training_catalog,
            evaluation_artifact.pairs,
            data_root=data_root,
            linker_bundle_root=linker_bundle_root,
            output_dir=output_dir,
            config=config,
            input_identity=input_identity,
            resume=args.resume,
        )

    public_gold = _load_public_gold(data_root, config.public_domains)
    b3_plans = build_b3_evaluation_plans(
        public_gold,
        evaluation_seed=config.evaluation_seed,
        threshold_pairs_per_domain=config.threshold_pairs_per_domain,
        b3_scope=config.b3_scope,
    )
    frozen_baseline = None
    if args.frozen_baseline_ranking_input is not None:
        frozen_baseline = verify_frozen_baseline(
            ranking_input_path=args.frozen_baseline_ranking_input,
            output_dir=output_dir,
            training_catalog=training_catalog,
            evaluation_artifact=evaluation_artifact,
            b3_plans=b3_plans,
            config=config,
        )
    if args.preflight_only:
        additive_preflight = None
        if additive_arms:
            if frozen_baseline is None:
                raise AssertionError("full additive preflight requires a verified frozen baseline")
            additive_preflight = verify_additive_recipe_assemblies(
                output_dir=output_dir,
                training_catalog=training_catalog,
                arms=arms,
                frozen_baseline=frozen_baseline,
                config=config,
            )
        print(
            json.dumps(
                {
                    "additive_fold_assemblies_verified": (
                        0 if additive_preflight is None else len(additive_preflight["records"])
                    ),
                    "comparison_identity_sha256": comparison_identity_sha256,
                    "event": "preflight_complete",
                    "frozen_baseline_verified": frozen_baseline is not None,
                    "reused_artifacts_verified": feature_stores is not None,
                    "run_id": run_id,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return

    if args.phase in {"run", "all"}:
        run_ablation(
            training_catalog,
            evaluation_artifact,
            data_root=data_root,
            linker_bundle_root=linker_bundle_root,
            donor_model_dir=donor_model_dir,
            output_dir=output_dir,
            config=config,
            arms=arms,
            run_id=run_id,
            input_identity=input_identity,
            b3_cache_root=b3_cache_root,
            rust_version=rust_version,
            rust_extension_sha256=rust_extension_sha256,
            cache_builder_identity=b3_builder_identity,
            resume=args.resume,
            feature_stores=feature_stores,
            frozen_baseline=frozen_baseline,
            prepared_b3_plans=b3_plans,
        )
    if args.phase in {"summarize", "all"}:
        summary = summarize_results(
            output_dir=output_dir,
            arms=arms,
            config=config,
            run_id=run_id,
            training_catalog=training_catalog,
            evaluation_artifact=evaluation_artifact,
            b3_plans=b3_plans,
        )
        if summary["complete"]:
            ranking_input_path = write_run_ranking_input(
                output_dir=output_dir,
                arms=arms,
                config=config,
                run_id=run_id,
                training_catalog=training_catalog,
                evaluation_artifact=evaluation_artifact,
                b3_plans=b3_plans,
            )
            summary = _persist_summary_with_ranking_input(
                output_dir=output_dir,
                summary=summary,
                ranking_input_path=ranking_input_path,
            )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
