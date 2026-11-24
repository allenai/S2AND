"""Compatibility imports for cluster corrections."""

from __future__ import annotations

import logging as logging
import math as math
from collections import Counter as Counter
from collections.abc import Callable as Callable
from collections.abc import Mapping as Mapping
from collections.abc import Sequence as Sequence
from collections.abc import Set as Set
from dataclasses import dataclass as dataclass
from dataclasses import field as field

import numpy as np  # noqa: F401

from s2and.corrections.name_conflict_repair import BONUS_MAX_PLAIN_AVERAGE as BONUS_MAX_PLAIN_AVERAGE
from s2and.corrections.name_conflict_repair import BONUS_MIN_CLOSE_SHARE as BONUS_MIN_CLOSE_SHARE
from s2and.corrections.name_conflict_repair import BONUS_MIN_SIDE_MENTIONS as BONUS_MIN_SIDE_MENTIONS
from s2and.corrections.name_conflict_repair import DEFAULT_MAX_AVERAGE_DISTANCE as DEFAULT_MAX_AVERAGE_DISTANCE
from s2and.corrections.name_conflict_repair import MAX_ODD_MENTIONS as MAX_ODD_MENTIONS
from s2and.corrections.name_conflict_repair import MIN_BLOCK_SIZE as MIN_BLOCK_SIZE
from s2and.corrections.name_conflict_repair import MIN_SIDE as MIN_SIDE
from s2and.corrections.name_conflict_repair import REPAIR_ROUNDS as REPAIR_ROUNDS
from s2and.corrections.name_conflict_repair import NameConflictRepair as NameConflictRepair
from s2and.corrections.name_conflict_repair import _AsideRule as _AsideRule
from s2and.corrections.name_conflict_repair import _height_bound as _height_bound
from s2and.corrections.name_conflict_repair import _impossible_pairs as _impossible_pairs
from s2and.corrections.name_conflict_repair import _merge_disjoint as _merge_disjoint
from s2and.corrections.name_conflict_repair import _name_rules_explain_every_pair as _name_rules_explain_every_pair
from s2and.corrections.name_conflict_repair import _pair_evidence as _pair_evidence
from s2and.corrections.name_conflict_repair import _PairEvidence as _PairEvidence
from s2and.corrections.name_conflict_repair import _passes_merge_guards as _passes_merge_guards
from s2and.corrections.name_conflict_repair import _reading_merge_bound as _reading_merge_bound
from s2and.corrections.name_conflict_repair import _reading_round as _reading_round
from s2and.corrections.name_conflict_repair import _repair_round as _repair_round
from s2and.corrections.name_conflict_repair import _smallest_covers as _smallest_covers
from s2and.corrections.name_conflict_repair import has_full_first_name as has_full_first_name
from s2and.corrections.name_conflict_repair import leading_orcids as leading_orcids
from s2and.corrections.name_conflict_repair import logger as logger
from s2and.corrections.name_conflict_repair import repair_close_code as repair_close_code
from s2and.corrections.name_conflict_repair import repair_name_conflicts_in_block as repair_name_conflicts_in_block
from s2and.pair_codes import FARTHEST_CODE as FARTHEST_CODE
from s2and.pair_codes import HARD_CODE as HARD_CODE
from s2and.pair_codes import SCALE as SCALE
from s2and.pair_codes import certain_close_code as certain_close_code
from s2and.pair_codes import code_tiles as code_tiles
from s2and.pair_codes import codes_against as codes_against
from s2and.runtime import load_s2and_rust_extension as load_s2and_rust_extension
