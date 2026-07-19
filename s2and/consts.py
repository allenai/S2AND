import json
import logging
import os
import threading
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("s2and")

try:
    PROJECT_ROOT_PATH = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
except NameError:
    PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.getcwd()))

# Package-level data directory (s2and/data/) ships with pip install
_PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
_PACKAGE_DATA_DIR = os.path.join(_PACKAGE_DIR, "data")

CONFIG_LOCATION_ENV = "S2AND_PATH_CONFIG"
CONFIG_LOCATION = os.path.join(_PACKAGE_DATA_DIR, "path_config.json")
_MAIN_DATA_DIR_PLACEHOLDER = "absolute path of wherever you downloaded the data to"
_CONFIG: dict[str, Any] | None = None
_CONFIG_LOCK = threading.Lock()


def _resolved_config_location() -> str:
    env_override = os.environ.get(CONFIG_LOCATION_ENV)
    if env_override:
        return env_override
    return CONFIG_LOCATION


def _load_config() -> dict[str, Any]:
    config_location = _resolved_config_location()
    try:
        with open(config_location, encoding="utf-8") as json_file:
            raw_config = json.load(json_file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find S2AND path config at {config_location!r}. "
            f"Set {CONFIG_LOCATION_ENV} or ensure s2and/data/path_config.json exists in the package."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in S2AND path config {config_location!r}: {exc.msg}") from exc

    if not isinstance(raw_config, dict):
        raise ValueError(f"Invalid S2AND path config at {config_location!r}: expected a JSON object.")

    config = dict(raw_config)
    main_data_dir = config.get("main_data_dir")
    if main_data_dir is None:
        raise ValueError(f"Invalid S2AND path config at {config_location!r}: missing 'main_data_dir'.")

    if main_data_dir == _MAIN_DATA_DIR_PLACEHOLDER:
        logger.warning(
            "You haven't set `main_data_dir` in s2and/data/path_config.json! "
            "Using s2and/data/ as default data directory."
        )
        main_data_dir = _PACKAGE_DATA_DIR

    resolved_main_data_dir = os.path.abspath(str(main_data_dir))
    if not os.path.exists(resolved_main_data_dir):
        raise FileNotFoundError(
            "The `main_data_dir` specified in path_config.json doesn't exist: " f"{resolved_main_data_dir!r}."
        )
    config["main_data_dir"] = resolved_main_data_dir
    return config


def _get_config() -> dict[str, Any]:
    global _CONFIG
    if _CONFIG is None:
        with _CONFIG_LOCK:
            if _CONFIG is None:
                _CONFIG = _load_config()
    return _CONFIG


class _LazyConfig(MutableMapping[str, Any]):
    def _state(self) -> dict[str, Any]:
        return _get_config()

    def __getitem__(self, key: str) -> Any:
        return self._state()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._state()[key] = value

    def __delitem__(self, key: str) -> None:
        del self._state()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._state())

    def __len__(self) -> int:
        return len(self._state())


class _LazyDataPath(os.PathLike[str]):
    def __init__(self, filename: str, *, fallback_url: str | None = None):
        self._filename = filename
        self._fallback_url = fallback_url

    def _resolve(self) -> str:
        main_data_dir = str(CONFIG["main_data_dir"])
        candidate = os.path.join(main_data_dir, self._filename)
        if self._fallback_url and not os.path.exists(candidate):
            return self._fallback_url
        return candidate

    def __fspath__(self) -> str:
        return self._resolve()

    def __str__(self) -> str:
        return self._resolve()

    def __repr__(self) -> str:
        return f"_LazyDataPath(filename={self._filename!r}, fallback_url={self._fallback_url!r})"


# Lazily-loaded path config to avoid import-time file I/O.
CONFIG: MutableMapping[str, Any] = _LazyConfig()

# Lazily-resolved artifact paths
NAME_COUNTS_INDEX_PATH = _LazyDataPath("name_counts_index")

# feature caching related consts
CACHE_ROOT = Path(os.getenv("S2AND_CACHE", str(Path.home() / ".s2and"))).resolve()
"""
Incrementation history
1 - initial version
2 - changed to SPECTERv2, subblocking etc
3 - name-count semantics contract and inference compatibility gating
4 - correctness-pass feature-value fixes (self-cite same-paper guard, email
    missing-suffix, equal_middle multi-token initials, empty-surname name
    counts, unconditional reference-list features, decoupled ngram short-token
    filter, whitespace-only equal)
5 - language detection: is_reliable requires the detector to return a known,
    reliable language. Affects the language reliability, predicted-language
    equality, and english_or_unknown_count features. (Sinonym normalization
    path removed; no feature-value effect.)
6 - reference features removed entirely: the six reference_features columns
    (references_authors_overlap, references_titles_overlap,
    references_venues_overlap, references_author_blocks_jaccard,
    references_self_citation, references_overlap) no longer exist and the
    feature vector shrank from 39 to 33 columns. Models trained on the old
    layout are incompatible.
7 - Rust language-detection text gate aligned to Python str.isalpha semantics
    (general category L* only). Rust's char::is_alphabetic additionally counted
    Other_Alphabetic (e.g. Indic combining vowel signs) and Nl characters,
    which could flip the zero-alpha early exit and the >0.9 uppercase-ratio
    lowercasing branch on such titles, diverging Rust language detection from
    Python. Python feature values are unchanged; Rust-side values change only
    for titles containing those characters near the gate boundaries.
8 - canonical_v2 name normalization cutover (docs/normalization_migration_blocked.md).
    Signature first/middle/last fields are canonicalized with one routine
    (apostrophe-like marks deleted in all fields, uniform dash separators,
    dash-bound given-name compounds kept as spaced tokens, spaced canonical
    surnames with particles preserved). Count keys use the gated canonical
    fields (no compact-join shims; missing components are NaN, and the
    "legacy_full_first_token" last_first_initial semantics is retired).
    Name-tuple compatibility probing (joined/first-token forms) and the
    subblocking ASCII/non-ASCII dash spill repair are removed. Every name
    field value and count feature can change for names with dashes,
    apostrophe-like marks, compound surnames, or title prefixes.
9 - fastText removed from language detection; CLD2 is now the single detector.
    The old boolean `language_reliability_count` pair feature is replaced by
    `language_reliability_min`, the minimum CLD2 reliable-confidence score for
    the two papers. Models trained on the old language-feature policy are
    incompatible.
10 - canonical runtime/parity corrections: titles preserve identifying digits,
     CLD2 is explicitly called in plain-text mode, malformed emails are missing
     evidence, query-author text is assembled only from canonical fields, and
     incremental six-decimal values use ties-to-even rounding.
"""
FEATURIZER_VERSION = 10

# Name-normalization contract (docs/normalization_migration_blocked.md, OD4
# single-mode cutover). Code, models, and data artifacts must all declare this
# exact policy; rollback deploys the previous package and artifact set together.
NORMALIZATION_VERSION = "canonical_v2"

# important constant values
NUMPY_NAN = np.nan
DEFAULT_CHUNK_SIZE = 100
LARGE_DISTANCE = 1e4
LARGE_INTEGER = 10 * LARGE_DISTANCE
CLUSTER_SEEDS_LOOKUP = {"require": 0, "disallow": LARGE_DISTANCE}
SPECTER_DIM = 768
