from __future__ import annotations

import logging
import pickle
import warnings
from collections.abc import Iterator
from os import PathLike
from typing import Any, BinaryIO

import numpy as np
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("s2and")

PickleSource = str | PathLike[str] | BinaryIO


def _iter_object_graph(root: Any) -> Iterator[Any]:
    stack = [root]
    seen: set[int] = set()
    while stack:
        obj = stack.pop()
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        yield obj

        if isinstance(obj, dict):
            stack.extend(obj.values())
            continue

        if isinstance(obj, list | tuple | set | frozenset):
            stack.extend(obj)
            continue

        obj_dict = getattr(obj, "__dict__", None)
        if isinstance(obj_dict, dict):
            stack.extend(obj_dict.values())


def _refresh_compatible_label_encoders(root: Any) -> int:
    refreshed_count = 0
    for obj in _iter_object_graph(root):
        label_encoder = getattr(obj, "_le", None)
        classes = getattr(obj, "_classes", None)
        if not isinstance(label_encoder, LabelEncoder) or classes is None:
            continue
        if not hasattr(label_encoder, "classes_"):
            continue

        classes_array = np.asarray(classes)
        encoder_classes_array = np.asarray(label_encoder.classes_)
        if not np.array_equal(classes_array, encoder_classes_array):
            continue

        refreshed = LabelEncoder()
        refreshed.classes_ = classes_array.copy()
        obj._le = refreshed
        refreshed_count += 1
    return refreshed_count


def _replay_warning(warning_message: warnings.WarningMessage) -> None:
    warnings.warn(
        message=warning_message.message,
        category=warning_message.category,
        stacklevel=2,
    )


def load_pickle_with_verified_label_encoder_compat(
    source: PickleSource, *, suppress_safe_labelencoder_warning: bool = True
) -> Any:
    """
    Load a pickle and suppress sklearn LabelEncoder version warnings only when
    `_le.classes_` exactly matches sibling `_classes`.
    """
    should_close = not hasattr(source, "read")
    file_obj: BinaryIO
    if should_close:
        file_obj = open(source, "rb")  # type: ignore[arg-type]
    else:
        file_obj = source  # type: ignore[assignment]

    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", InconsistentVersionWarning)
            loaded = pickle.load(file_obj)
    finally:
        if should_close:
            file_obj.close()

    inconsistent_warnings = [
        warning_message
        for warning_message in caught_warnings
        if isinstance(warning_message.message, InconsistentVersionWarning)
    ]

    suppressed_warning_ids: set[int] = set()
    if suppress_safe_labelencoder_warning and inconsistent_warnings:
        label_encoder_warnings = [
            warning_message
            for warning_message in inconsistent_warnings
            if getattr(warning_message.message, "estimator_name", None) == "LabelEncoder"
        ]
        non_label_encoder_warnings = [
            warning_message
            for warning_message in inconsistent_warnings
            if getattr(warning_message.message, "estimator_name", None) != "LabelEncoder"
        ]
        if label_encoder_warnings and not non_label_encoder_warnings:
            refreshed_count = _refresh_compatible_label_encoders(loaded)
            if refreshed_count >= len(label_encoder_warnings):
                suppressed_warning_ids = {id(warning_message) for warning_message in label_encoder_warnings}
                logger.debug(
                    "Suppressed %d verified-safe LabelEncoder sklearn version warning(s).",
                    len(label_encoder_warnings),
                )

    for warning_message in caught_warnings:
        if id(warning_message) in suppressed_warning_ids:
            continue
        _replay_warning(warning_message)

    return loaded
