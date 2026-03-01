from __future__ import annotations

import os

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}


def parse_bool_env(name: str, *, default: bool = False, strict: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return bool(default)
    normalized = raw_value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    if strict:
        raise ValueError(f"Invalid {name}={raw_value!r}; expected one of 0/1/true/false/yes/no/on/off.")
    return bool(default)
