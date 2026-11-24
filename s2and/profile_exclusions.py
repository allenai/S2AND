"""Compatibility imports. The code lives in `s2and.corrections.profile_exclusions`."""

from __future__ import annotations

import logging as logging
from collections import defaultdict as defaultdict
from collections.abc import Callable as Callable
from collections.abc import Iterable as Iterable
from collections.abc import Mapping as Mapping
from collections.abc import Sequence as Sequence
from typing import Any as Any

import numpy as np  # noqa: F401

from s2and.consts import LARGE_DISTANCE as LARGE_DISTANCE
from s2and.corrections.profile_exclusions import MAX_EVICTION_ROUNDS as MAX_EVICTION_ROUNDS
from s2and.corrections.profile_exclusions import ProfileExclusions as ProfileExclusions
from s2and.corrections.profile_exclusions import _required_groups as _required_groups
from s2and.corrections.profile_exclusions import _set_condensed_pairs as _set_condensed_pairs
from s2and.corrections.profile_exclusions import _violating_pairs as _violating_pairs
from s2and.corrections.profile_exclusions import _violations as _violations
from s2and.corrections.profile_exclusions import (
    enforce_profile_exclusions_in_block as enforce_profile_exclusions_in_block,
)
from s2and.corrections.profile_exclusions import (
    evict_profile_exclusion_violations as evict_profile_exclusion_violations,
)
from s2and.corrections.profile_exclusions import logger as logger
from s2and.corrections.profile_exclusions import normalize_profile_exclusions as normalize_profile_exclusions
from s2and.corrections.profile_exclusions import validate_profile_exclusions as validate_profile_exclusions
