"""Compatibility imports for cluster corrections."""

from __future__ import annotations

import logging as logging
from collections import Counter as Counter
from collections import defaultdict as defaultdict
from collections.abc import Callable as Callable
from collections.abc import Mapping as Mapping
from collections.abc import Sequence as Sequence
from collections.abc import Set as Set

from s2and.corrections.orcid_join import INITIAL_ONLY_MIN_COVERAGE as INITIAL_ONLY_MIN_COVERAGE
from s2and.corrections.orcid_join import SUPPORTED_FORM_SHARE as SUPPORTED_FORM_SHARE
from s2and.corrections.orcid_join import _can_join as _can_join
from s2and.corrections.orcid_join import _dominant_first_word as _dominant_first_word
from s2and.corrections.orcid_join import _first_word as _first_word
from s2and.corrections.orcid_join import _names_conflict_across as _names_conflict_across
from s2and.corrections.orcid_join import _supported_forms as _supported_forms
from s2and.corrections.orcid_join import _supported_orcid as _supported_orcid
from s2and.corrections.orcid_join import join_orcids_across_subblocks as join_orcids_across_subblocks
from s2and.corrections.orcid_join import logger as logger
from s2and.name_conflict_repair import leading_orcids as leading_orcids
from s2and.text import CanonicalNameParts as CanonicalNameParts
from s2and.text import canonical_names_conflict as canonical_names_conflict
