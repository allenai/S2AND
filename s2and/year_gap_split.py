"""Compatibility imports. The code lives in `s2and.corrections.year_gap_split`."""

from __future__ import annotations

import logging as logging
from collections.abc import Mapping as Mapping
from collections.abc import Sequence as Sequence

import numpy as np  # noqa: F401

from s2and.consts import PLACEHOLDER_YEARS as PLACEHOLDER_YEARS
from s2and.consts import YEAR_DIFF_CAP as YEAR_DIFF_CAP
from s2and.consts import normalize_paper_year as normalize_paper_year
from s2and.corrections.year_gap_split import SMALL_SURNAME_BLOCK as SMALL_SURNAME_BLOCK
from s2and.corrections.year_gap_split import _has_close_pair as _has_close_pair
from s2and.corrections.year_gap_split import _mean_cross_code as _mean_cross_code
from s2and.corrections.year_gap_split import _parts as _parts
from s2and.corrections.year_gap_split import _place_undated as _place_undated
from s2and.corrections.year_gap_split import _shared as _shared
from s2and.corrections.year_gap_split import logger as logger
from s2and.corrections.year_gap_split import split_year_gaps as split_year_gaps
from s2and.corrections.year_gap_split import split_years as split_years
from s2and.pair_codes import code_tiles as code_tiles
