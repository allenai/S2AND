"""Strict loader for the small, immutable pair-ablation input layout."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from s2and.consts import FEATURIZER_VERSION, NORMALIZATION_VERSION
from s2and.featurizer import DEFAULT_FEATURE_GROUPS, DEFAULT_NAMELESS_FEATURE_GROUPS, FeaturizationInfo
from scripts._pair_ablation.study import ALL_DOMAINS, GOLD_DOMAINS, validate_catalog

_EVALUATION_FILES = ("main.npy", "nameless.npy", "labels.npy")
_B3_FILES = (
    "main.npy",
    "nameless.npy",
    "staged_labels.npy",
    "pair_offsets.npy",
    "signature_offsets.npy",
    "signature_ids.npy",
    "gold_cluster_ids.npy",
)
_FEATURE_SCHEMA = {
    "featurizer_version": FEATURIZER_VERSION,
    "normalization_version": NORMALIZATION_VERSION,
    "main_feature_groups": list(DEFAULT_FEATURE_GROUPS),
    "nameless_feature_groups": list(DEFAULT_NAMELESS_FEATURE_GROUPS),
}


def _feature_width(groups: list[str]) -> int:
    info = FeaturizationInfo(features_to_use=groups, featurizer_version=FEATURIZER_VERSION)
    return len(info.lightgbm_monotone_constraints.split(","))


_MAIN_WIDTH = _feature_width(list(DEFAULT_FEATURE_GROUPS))
_NAMELESS_WIDTH = _feature_width(list(DEFAULT_NAMELESS_FEATURE_GROUPS))


@dataclass(frozen=True, slots=True)
class PairEvaluation:
    """Pair-labeled evaluation matrices for one domain."""

    main: np.ndarray
    nameless: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True, slots=True)
class B3Evaluation:
    """Cluster-gold rows in SciPy condensed order within each signature block."""

    main: np.ndarray
    nameless: np.ndarray
    staged_labels: np.ndarray
    pair_offsets: np.ndarray
    signature_offsets: np.ndarray
    signature_ids: np.ndarray
    gold_cluster_ids: np.ndarray


@dataclass(frozen=True, slots=True)
class PreparedStudy:
    """All immutable inputs needed to run the pair ablation."""

    root: Path
    catalog: pd.DataFrame
    training_main: np.ndarray
    training_nameless: np.ndarray
    evaluation: dict[str, PairEvaluation]
    b3: dict[str, B3Evaluation]
    prepared_digest: str


def _load_npy(path: Path) -> np.ndarray:
    if not path.is_file():
        raise ValueError(f"missing prepared array: {path}")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _has_infinity(values: np.ndarray) -> bool:
    flat = values.reshape(-1)
    for start in range(0, len(flat), 1_000_000):
        if np.isinf(flat[start : start + 1_000_000]).any():
            return True
    return False


def _matrix(path: Path, *, rows: int | None = None, width: int | None = None) -> np.ndarray:
    values = _load_npy(path)
    if (
        values.ndim != 2
        or values.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or values.shape[1] == 0
        or not values.flags.c_contiguous
    ):
        raise ValueError(f"feature matrix must be a nonempty-width C-contiguous float32/float64 array: {path}")
    if rows is not None and values.shape[0] != rows:
        raise ValueError(f"feature row count mismatch: {path}")
    if width is not None and values.shape[1] != width:
        raise ValueError(f"feature width mismatch: {path}")
    if _has_infinity(values):
        raise ValueError(f"feature matrix contains infinity: {path}")
    return values


def _binary_labels(path: Path, *, rows: int) -> np.ndarray:
    values = _load_npy(path)
    if values.ndim != 1 or values.shape[0] != rows or values.dtype.kind not in "iu":
        raise ValueError(f"labels must be a one-dimensional integral array: {path}")
    if not values.size or not np.isin(values, (0, 1)).all():
        raise ValueError(f"labels must contain exact binary values: {path}")
    if not np.any(values == 0) or not np.any(values == 1):
        raise ValueError(f"evaluation labels must contain both classes: {path}")
    return values


def _staged_labels(path: Path, *, rows: int) -> np.ndarray:
    values = _load_npy(path)
    if values.ndim != 1 or values.shape[0] != rows or values.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError(f"staged labels must be a float32/float64 vector: {path}")
    if _has_infinity(values):
        raise ValueError(f"staged labels contain infinity: {path}")
    return values


def _ids(path: Path) -> np.ndarray:
    values = _load_npy(path)
    if values.ndim != 1 or values.dtype.kind != "U":
        raise ValueError(f"IDs must be a one-dimensional Unicode array: {path}")
    if any(not str(value) for value in values):
        raise ValueError(f"IDs must be nonempty: {path}")
    return values


def _offsets(path: Path, *, end: int, strict: bool) -> np.ndarray:
    values = _load_npy(path)
    malformed = (
        values.ndim != 1
        or values.dtype != np.dtype("int64")
        or len(values) < 2
        or np.any(values < 0)
        or np.any(values > end)
    )
    if not malformed:
        differences = np.diff(values)
        malformed = (
            int(values[0]) != 0 or int(values[-1]) != end or np.any(differences <= 0 if strict else differences < 0)
        )
    if malformed:
        qualifier = "strictly increase" if strict else "never decrease"
        raise ValueError(f"offsets must {qualifier} from zero to {end}: {path}")
    return values


def _subdirectories(root: Path) -> set[str]:
    return {path.name for path in root.iterdir() if path.is_dir()} if root.is_dir() else set()


def _validate_feature_schema(path: Path) -> None:
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read prepared feature schema: {path}") from exc
    if observed != _FEATURE_SCHEMA:
        raise ValueError(f"prepared feature schema does not match current S2AND defaults: {path}")


def _digest(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256(b"pair-ablation-prepared-v1\0")
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def load_prepared(root: str | Path) -> PreparedStudy:
    """Load and validate a prepared study without manifests or compatibility modes."""

    root = Path(root)
    catalog_path = root / "training" / "catalog.parquet"
    if not catalog_path.is_file():
        raise ValueError(f"missing prepared catalog: {catalog_path}")
    catalog = validate_catalog(pd.read_parquet(catalog_path))
    if catalog.empty:
        raise ValueError("prepared training catalog is empty")
    schema_path = root / "training" / "feature_schema.json"
    _validate_feature_schema(schema_path)
    training_main_path = root / "training" / "main.npy"
    training_nameless_path = root / "training" / "nameless.npy"
    training_main = _matrix(training_main_path, rows=len(catalog), width=_MAIN_WIDTH)
    training_nameless = _matrix(training_nameless_path, rows=len(catalog), width=_NAMELESS_WIDTH)
    main_width, nameless_width = training_main.shape[1], training_nameless.shape[1]
    required = [catalog_path, schema_path, training_main_path, training_nameless_path]

    evaluation_root = root / "evaluation"
    evaluation_domains = _subdirectories(evaluation_root)
    unknown = sorted(evaluation_domains.difference(ALL_DOMAINS))
    if not evaluation_domains or unknown:
        raise ValueError(f"evaluation must be a nonempty known-domain subset; unknown={unknown}")
    evaluation: dict[str, PairEvaluation] = {}
    for domain in sorted(evaluation_domains):
        domain_root = evaluation_root / domain
        paths = [domain_root / filename for filename in _EVALUATION_FILES]
        main = _matrix(paths[0], width=main_width)
        nameless = _matrix(paths[1], rows=main.shape[0], width=nameless_width)
        labels = _binary_labels(paths[2], rows=main.shape[0])
        evaluation[domain] = PairEvaluation(main, nameless, labels)
        required.extend(paths)

    expected_b3 = evaluation_domains.intersection(GOLD_DOMAINS)
    b3_root = root / "b3"
    actual_b3 = _subdirectories(b3_root)
    if actual_b3 != expected_b3:
        raise ValueError(
            f"B3 directories must equal loaded gold evaluation domains: "
            f"expected={sorted(expected_b3)}, actual={sorted(actual_b3)}"
        )
    b3: dict[str, B3Evaluation] = {}
    for domain in sorted(expected_b3):
        paths = [b3_root / domain / filename for filename in _B3_FILES]
        main = _matrix(paths[0], width=main_width)
        nameless = _matrix(paths[1], rows=main.shape[0], width=nameless_width)
        staged = _staged_labels(paths[2], rows=main.shape[0])
        signature_ids = _ids(paths[5])
        gold_ids = _ids(paths[6])
        if len(signature_ids) != len(gold_ids) or len(set(signature_ids.tolist())) != len(signature_ids):
            raise ValueError(f"B3 signature IDs must be unique and aligned with gold IDs: {domain}")
        pair_offsets = _offsets(paths[3], end=main.shape[0], strict=False)
        signature_offsets = _offsets(paths[4], end=len(signature_ids), strict=True)
        if len(pair_offsets) != len(signature_offsets):
            raise ValueError(f"B3 offset arrays must describe the same blocks: {domain}")
        signature_counts = np.diff(signature_offsets)
        if not np.array_equal(np.diff(pair_offsets), signature_counts * (signature_counts - 1) // 2):
            raise ValueError(f"B3 block pair counts do not equal nC2: {domain}")
        b3[domain] = B3Evaluation(
            main,
            nameless,
            staged,
            pair_offsets,
            signature_offsets,
            signature_ids,
            gold_ids,
        )
        required.extend(paths)

    return PreparedStudy(
        root,
        catalog,
        training_main,
        training_nameless,
        evaluation,
        b3,
        _digest(root, required),
    )
