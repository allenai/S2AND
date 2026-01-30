from typing import Optional, Any
import os
import weakref
import threading
import logging

from s2and.consts import CACHE_ROOT, FEATURIZER_VERSION, CLUSTER_SEEDS_LOOKUP, LARGE_DISTANCE
from s2and.data import ANDData


# Treat extension as Any for typing; it is optional.
_s2and_rust: Optional[Any]
try:
    import s2and_rust as _s2and_rust  # type: ignore
except Exception:
    _s2and_rust = None
s2and_rust: Optional[Any] = _s2and_rust

logger = logging.getLogger("s2and")

_RUST_FEATURIZER_CACHE: "weakref.WeakKeyDictionary[ANDData, object]" = weakref.WeakKeyDictionary()
_RUST_FEATURIZER_CACHE_LOCK = threading.Lock()
RUST_FEATURIZER_CACHE_VERSION = 2

_ENV_TRUE_VALUES = {"1", "true", "yes"}


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in _ENV_TRUE_VALUES


def _rust_prod_mode(dataset: Optional[ANDData] = None) -> bool:
    if "S2AND_RUST_PROD_MODE" in os.environ:
        return _env_flag("S2AND_RUST_PROD_MODE", "0")
    return bool(dataset is not None and getattr(dataset, "mode", "") == "inference")


def _should_write_rust_cache(write_cache: Optional[bool], dataset: Optional[ANDData] = None) -> bool:
    if _rust_prod_mode(dataset):
        return False
    if write_cache is not None:
        return write_cache
    return _env_flag("S2AND_RUST_FEATURIZER_DISK_CACHE_WRITE", "0")


def _rust_cache_path(dataset: ANDData) -> str:
    cache_dir = os.environ.get("S2AND_RUST_FEATURIZER_CACHE_DIR", "")
    if not cache_dir:
        cache_dir = os.path.join(str(CACHE_ROOT), "rust_featurizer")
    os.makedirs(cache_dir, exist_ok=True)
    skip_fasttext = _env_flag("S2AND_SKIP_FASTTEXT", "")
    rust_version = getattr(s2and_rust, "__version__", None) if s2and_rust is not None else None
    rust_version = rust_version or str(RUST_FEATURIZER_CACHE_VERSION)
    key = (
        f"{dataset.name}_v{FEATURIZER_VERSION}_rv{rust_version}"
        f"_s{len(dataset.signatures)}_p{len(dataset.papers)}"
        f"_n{len(dataset.name_tuples)}_r{int(dataset.compute_reference_features)}"
        f"_pre{int(getattr(dataset, 'preprocess', True))}_sf{int(skip_fasttext)}"
    )
    return os.path.join(cache_dir, f"{key}.bin")


def _get_rust_featurizer(dataset: ANDData, write_cache: Optional[bool] = None) -> Any:
    if s2and_rust is None:
        raise RuntimeError("s2and_rust extension not built. Build with: " "maturin develop -m s2and_rust/Cargo.toml")
    featurizer = _RUST_FEATURIZER_CACHE.get(dataset)
    if featurizer is not None:
        return featurizer

    with _RUST_FEATURIZER_CACHE_LOCK:
        featurizer = _RUST_FEATURIZER_CACHE.get(dataset)
        if featurizer is not None:
            return featurizer

        use_disk_cache = _env_flag("S2AND_RUST_FEATURIZER_DISK_CACHE", "1") and not _rust_prod_mode(dataset)
        cache_path = _rust_cache_path(dataset) if use_disk_cache else None
        if use_disk_cache and cache_path and os.path.exists(cache_path):
            try:
                featurizer = s2and_rust.RustFeaturizer.load(cache_path)
                # Ensure cluster seeds reflect the current dataset, even if the cache is reused.
                featurizer.update_cluster_seeds(dataset.cluster_seeds_require, dataset.cluster_seeds_disallow)
            except Exception as e:  # pragma: no cover - disk cache is best-effort
                logger.warning(f"Failed to load Rust featurizer cache at {cache_path}: {e}")
                featurizer = None
        if featurizer is None:
            featurizer = s2and_rust.RustFeaturizer.from_dataset(
                dataset,
                CLUSTER_SEEDS_LOOKUP["require"],
                CLUSTER_SEEDS_LOOKUP["disallow"],
            )
            if use_disk_cache and cache_path and _should_write_rust_cache(write_cache, dataset):
                try:
                    featurizer.save(cache_path)
                except Exception as e:  # pragma: no cover - disk cache is best-effort
                    logger.warning(f"Failed to save Rust featurizer cache at {cache_path}: {e}")
        _RUST_FEATURIZER_CACHE[dataset] = featurizer

    return featurizer


def warm_rust_featurizer(dataset: ANDData) -> None:
    """Preload the Rust featurizer into memory for low-latency inference."""
    _get_rust_featurizer(dataset)


def update_rust_cluster_seeds(dataset: ANDData) -> None:
    featurizer = _get_rust_featurizer(dataset)
    featurizer.update_cluster_seeds(dataset.cluster_seeds_require, dataset.cluster_seeds_disallow)


def get_constraint_rust(
    dataset: ANDData,
    sig_id_1: str,
    sig_id_2: str,
    low_value: float = 0.0,
    high_value: float = LARGE_DISTANCE,
    dont_merge_cluster_seeds: bool = True,
    incremental_dont_use_cluster_seeds: bool = False,
    featurizer: Optional[Any] = None,
):
    if s2and_rust is None:
        raise RuntimeError("s2and_rust extension not built. Build with: " "maturin develop -m s2and_rust/Cargo.toml")
    if featurizer is None:
        featurizer = _get_rust_featurizer(dataset)
    return featurizer.get_constraint(
        sig_id_1,
        sig_id_2,
        low_value,
        high_value,
        dont_merge_cluster_seeds,
        incremental_dont_use_cluster_seeds,
    )


def featurize_pair_rust(dataset: ANDData, sig_id_1: str, sig_id_2: str):
    featurizer = _get_rust_featurizer(dataset)
    return featurizer.featurize_pair(sig_id_1, sig_id_2)
