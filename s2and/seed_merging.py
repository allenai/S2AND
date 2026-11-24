"""Compatibility imports for cluster corrections."""

from collections import defaultdict as defaultdict
from collections.abc import Iterable as Iterable
from collections.abc import Mapping as Mapping
from collections.abc import Sequence as Sequence
from collections.abc import Set as Set
from typing import TypeVar as TypeVar

import numpy as np  # noqa: F401

from s2and.consts import LARGE_DISTANCE as LARGE_DISTANCE
from s2and.corrections.seed_merging import _Component as _Component
from s2and.corrections.seed_merging import filter_generated_seed_negatives as filter_generated_seed_negatives
from s2and.corrections.seed_merging import merge_seed_labels as merge_seed_labels
from s2and.corrections.seed_merging import pack_seed_components as pack_seed_components
from s2and.corrections.seed_merging import restore_seed_membership as restore_seed_membership
from s2and.corrections.seed_merging import seed_disallow_adjacency as seed_disallow_adjacency
