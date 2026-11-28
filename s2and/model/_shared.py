"""Hold helpers that several `s2and.model` modules share.

Keep these here so sibling modules do not import them from `incremental`.
"""

from __future__ import annotations

import copy
import gc
import logging
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from s2and.model_pairwise import clone_cluster_model, intify

if TYPE_CHECKING:
    from .incremental import _AttachmentBlockState

logger = logging.getLogger("s2and")


# Carry bulk semantics through public attachment overrides that delegate to
# super(), without mutating a shared Clusterer or changing public arguments.
_ARROW_BULK_DISTANCE_OVERRIDES: ContextVar[bool] = ContextVar("s2and_arrow_bulk_distance_overrides", default=False)


# Subblocked initial-only attachment sets the caller-required seed members that
# its name checks skip. None keeps the public incremental behavior unchanged.
_ARROW_ATTACH_NAME_CHECK_EXEMPT: ContextVar[frozenset[str] | None] = ContextVar(
    "s2and_arrow_attach_name_check_exempt", default=None
)


# Subblocked attachment passes share native state over the original block: one
# featurizer that each pass views, and one planner that each pass reloads instead
# of rebuilding. None builds per pass.
_ARROW_ATTACH_BLOCK_STATE: ContextVar[_AttachmentBlockState | None] = ContextVar(
    "s2and_arrow_attach_block_state", default=None
)


_P = ParamSpec("_P")


_R = TypeVar("_R")


def _gc_paused(method: Callable[_P, _R]) -> Callable[_P, _R]:
    """Pause Python's cyclic garbage collector while ``method`` runs.

    Predictions allocate millions of short-lived containers, and each full
    collection walks every live object; that took 1-2% of large requests.
    Reference counting still frees acyclic objects at once, and the collector
    resumes afterwards for cycles. A nested call leaves the pause to its caller.
    """

    @wraps(method)
    def paused(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if not gc.isenabled():
            return method(*args, **kwargs)
        gc.disable()
        try:
            return method(*args, **kwargs)
        finally:
            gc.enable()

    return paused


def _clusterer_with_prediction_params(clusterer: Any, params: dict[str, Any] | None) -> Any:
    """Bind explicit clustering overrides to a shallow, request-owned receiver.

    Pairwise models and native resources remain shared. Nested prediction calls
    reuse this receiver, while each block still clones its clustering estimator.
    """
    if not params or params is getattr(clusterer, "_s2and_prediction_params_source", None):
        return clusterer
    if params is getattr(clusterer, "_s2and_prediction_params", None):
        return clusterer
    request = copy.copy(clusterer)
    request.cluster_model = clone_cluster_model(clusterer.cluster_model)
    effective_params = {key: intify(value) for key, value in params.items()}
    request.cluster_model.set_params(**effective_params)
    request._s2and_prediction_params_source = params
    request._s2and_prediction_params = effective_params
    request._s2and_prediction_cache_owner = getattr(clusterer, "_s2and_prediction_cache_owner", clusterer)
    return request


def _prediction_cluster_model_params(clusterer: Any, params: dict[str, Any] | None) -> dict[str, Any] | None:
    """Retain explicit overrides in nested calls, including input ownership."""
    return params if params is not None else getattr(clusterer, "_s2and_prediction_params", None)
