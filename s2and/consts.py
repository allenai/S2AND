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

CONFIG_LOCATION_ENV = "S2AND_PATH_CONFIG"
CONFIG_LOCATION = os.path.join(PROJECT_ROOT_PATH, "data", "path_config.json")
_MAIN_DATA_DIR_PLACEHOLDER = "absolute path of wherever you downloaded the data to"
_NAME_COUNTS_FALLBACK_URL = "https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/s2and-release/name_counts.pickle"
_FASTTEXT_FALLBACK_URL = "https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/s2and-release/lid.176.bin"
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
            f"Set {CONFIG_LOCATION_ENV} or create data/path_config.json."
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
            "You haven't set `main_data_dir` in data/path_config.json! Using data/ as default data directory."
        )
        main_data_dir = os.path.join(PROJECT_ROOT_PATH, "data")

    resolved_main_data_dir = os.path.abspath(str(main_data_dir))
    if not os.path.exists(resolved_main_data_dir):
        raise FileNotFoundError(
            "The `main_data_dir` specified in data/path_config.json doesn't exist: " f"{resolved_main_data_dir!r}."
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
NAME_COUNTS_PATH = _LazyDataPath("name_counts.pickle", fallback_url=_NAME_COUNTS_FALLBACK_URL)
FASTTEXT_PATH = _LazyDataPath("lid.176.bin", fallback_url=_FASTTEXT_FALLBACK_URL)

# feature caching related consts
CACHE_ROOT = Path(os.getenv("S2AND_CACHE", str(Path.home() / ".s2and"))).resolve()
"""
Incrementation history
1 - initial version
2 - changed to SPECTERv2, subblocking etc
3 - name-count semantics contract and inference compatibility gating
"""
FEATURIZER_VERSION = 3

# important constant values
NUMPY_NAN = np.nan
DEFAULT_CHUNK_SIZE = 100
LARGE_DISTANCE = 1e4
LARGE_INTEGER = 10 * LARGE_DISTANCE
CLUSTER_SEEDS_LOOKUP = {"require": 0, "disallow": LARGE_DISTANCE}
SPECTER_DIM = 768
