"""Exact model-to-artifact identity for global name-count features."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from s2and._sha256 import is_lowercase_sha256

_FEATURE_CONTRACT_FIELD = "name_counts_manifest_sha256"


def _nonempty_string(value: Any, *, field: str, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} requires nonempty string {field}")
    return value


def _sha256(value: Any, *, field: str, context: str) -> str:
    digest = _nonempty_string(value, field=field, context=context)
    if not is_lowercase_sha256(digest):
        raise ValueError(f"{context} requires lowercase SHA-256 {field}")
    return digest


@dataclass(frozen=True, slots=True)
class NameCountsBinding:
    """The native index manifest identity persisted in a model contract."""

    manifest_sha256: str

    @classmethod
    def from_feature_contract(
        cls,
        feature_contract: Mapping[str, Any] | None,
        *,
        context: str,
    ) -> NameCountsBinding:
        """Read one complete binding without copying the feature-contract mapping."""

        if not isinstance(feature_contract, Mapping):
            raise ValueError(f"{context} requires a feature_contract mapping with a name-count identity")
        if _FEATURE_CONTRACT_FIELD not in feature_contract:
            raise ValueError(f"{context} requires {_FEATURE_CONTRACT_FIELD}")
        return cls(
            manifest_sha256=_sha256(
                feature_contract[_FEATURE_CONTRACT_FIELD],
                field=_FEATURE_CONTRACT_FIELD,
                context=context,
            ),
        )

    @classmethod
    def from_manifest_sha256(cls, manifest_sha256: Any, *, context: str) -> NameCountsBinding:
        """Read one direct identity from an already-opened name-count manifest."""

        return cls(
            manifest_sha256=_sha256(
                manifest_sha256,
                field=_FEATURE_CONTRACT_FIELD,
                context=context,
            ),
        )

    @classmethod
    def from_rust_featurizer(cls, rust_featurizer: Any, *, context: str) -> NameCountsBinding:
        """Read the runtime-only binding retained by an Arrow-built Rust featurizer."""

        manifest_sha256 = getattr(rust_featurizer, "name_counts_manifest_sha256", None)
        if manifest_sha256 is None:
            raise ValueError(f"{context} requires a Rust featurizer with a verified name-count manifest")
        return cls(
            manifest_sha256=_sha256(
                manifest_sha256,
                field=_FEATURE_CONTRACT_FIELD,
                context=context,
            ),
        )

    def feature_contract_fields(self) -> dict[str, str]:
        """Return the exact fields persisted in a trained model contract."""

        return {_FEATURE_CONTRACT_FIELD: self.manifest_sha256}

    def require_matches(self, observed: NameCountsBinding, *, context: str, source: str) -> None:
        """Reject a model/artifact generation mismatch before feature computation."""

        if observed != self:
            raise ValueError(
                f"{context} name-count binding mismatch for {source}: expected={self!r} observed={observed!r}"
            )
