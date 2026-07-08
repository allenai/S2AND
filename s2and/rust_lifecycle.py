from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from s2and.runtime import Backend

RustLifecycleMode = Literal[
    "python_only",
    "rust_inference",
    "rust_arrow_training",
]

_SKIP_PYTHON_PAPER_PREPROCESS_MODES: frozenset[RustLifecycleMode] = frozenset(
    {
        "rust_arrow_training",
    }
)
_DEFER_SIGNATURE_NGRAM_MODES: frozenset[RustLifecycleMode] = frozenset(
    {
        "rust_inference",
        "rust_arrow_training",
    }
)
_DEFER_SIGNATURE_FIELD_MODES: frozenset[RustLifecycleMode] = frozenset(
    {
        "rust_arrow_training",
    }
)


@dataclass(frozen=True)
class RustLifecyclePolicy:
    """Frozen Rust lifecycle decision for a dataset."""

    mode: RustLifecycleMode

    @property
    def skip_python_paper_preprocess(self) -> bool:
        """Return whether Python paper preprocessing is deferred to Rust."""

        return self.mode in _SKIP_PYTHON_PAPER_PREPROCESS_MODES

    @property
    def defer_signature_ngrams_to_rust(self) -> bool:
        """Return whether signature n-gram computation is deferred to Rust."""

        return self.mode in _DEFER_SIGNATURE_NGRAM_MODES

    @property
    def defer_signature_fields_to_rust(self) -> bool:
        """Return whether normalized signature fields are deferred to Rust."""

        return self.mode in _DEFER_SIGNATURE_FIELD_MODES


PYTHON_ONLY_POLICY = RustLifecyclePolicy(mode="python_only")


def _is_inference_mode(mode: str) -> bool:
    return mode.strip().lower() == "inference"


def build_rust_lifecycle_policy(
    *,
    backend: Backend,
    mode: str,
    preprocess: bool,
    arrow_featurization: bool = False,
) -> RustLifecyclePolicy:
    """Decide which dataset-build work Python may defer to Rust.

    ``arrow_featurization`` marks datasets whose Rust featurizer is built from
    Arrow IPC artifacts (``RustFeaturizer.from_arrow_paths``); Rust reads text,
    n-grams, and name counts from the Arrow bundle, so Python-side paper
    preprocessing and signature n-gram/field materialization are skipped.
    Rust-backend datasets without Arrow featurizer paths featurize in Python
    (there is no Python-object ingestion door in Rust), so they get the full
    Python preprocessing lifecycle.
    """

    if backend == "python":
        return PYTHON_ONLY_POLICY

    if arrow_featurization:
        return RustLifecyclePolicy(mode="rust_arrow_training")

    if _is_inference_mode(mode) and preprocess:
        # Rust prediction reads Arrow artifacts; Python fallbacks materialize
        # signature n-grams lazily (featurizer._ensure_python_pair_signature_ngrams).
        return RustLifecyclePolicy(mode="rust_inference")

    return PYTHON_ONLY_POLICY
