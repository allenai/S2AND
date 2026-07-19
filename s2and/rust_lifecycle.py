from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RustLifecycleMode = Literal[
    "python_only",
    "rust_arrow_training",
]


@dataclass(frozen=True)
class RustLifecyclePolicy:
    """Frozen Rust lifecycle decision for a dataset."""

    mode: RustLifecycleMode

    @property
    def skip_python_paper_preprocess(self) -> bool:
        """Return whether Python paper preprocessing is deferred to Rust."""

        return self.mode == "rust_arrow_training"

    @property
    def defer_signature_ngrams_to_rust(self) -> bool:
        """Return whether signature n-gram computation is deferred to Rust."""

        return self.mode == "rust_arrow_training"

    @property
    def defer_signature_fields_to_rust(self) -> bool:
        """Return whether normalized signature fields are deferred to Rust."""

        return self.mode == "rust_arrow_training"


PYTHON_ONLY_POLICY = RustLifecyclePolicy(mode="python_only")
RUST_ARROW_TRAINING_POLICY = RustLifecyclePolicy(mode="rust_arrow_training")
