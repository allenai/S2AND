"""Exact model-to-artifact identity for global name-count features."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_FEATURE_CONTRACT_FIELDS = (
    "name_counts_generation_id",
    "name_counts_pickle_sha256",
    "name_counts_source_snapshot_id",
    "name_counts_selected_rows_sha256",
)


def _nonempty_string(value: Any, *, field: str, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} requires nonempty string {field}")
    return value


def _sha256(value: Any, *, field: str, context: str) -> str:
    digest = _nonempty_string(value, field=field, context=context)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{context} requires lowercase SHA-256 {field}")
    return digest


@dataclass(frozen=True, slots=True)
class NameCountsBinding:
    """The four provenance values persisted in a trained feature contract."""

    generation_id: str
    pickle_sha256: str
    source_snapshot_id: str
    selected_rows_sha256: str

    @classmethod
    def from_feature_contract(
        cls,
        feature_contract: Mapping[str, Any] | None,
        *,
        context: str,
        required: bool,
    ) -> NameCountsBinding | None:
        """Read one complete binding without copying the feature-contract mapping."""

        if not isinstance(feature_contract, Mapping):
            if required:
                raise ValueError(f"{context} requires a feature_contract mapping with name-count provenance")
            return None
        present_fields = tuple(field for field in _FEATURE_CONTRACT_FIELDS if field in feature_contract)
        if not present_fields:
            if required:
                raise ValueError(f"{context} requires name-count provenance fields {list(_FEATURE_CONTRACT_FIELDS)!r}")
            return None
        if len(present_fields) != len(_FEATURE_CONTRACT_FIELDS):
            missing_fields = [field for field in _FEATURE_CONTRACT_FIELDS if field not in feature_contract]
            raise ValueError(f"{context} has partial name-count provenance; missing={missing_fields!r}")
        return cls(
            generation_id=_nonempty_string(
                feature_contract["name_counts_generation_id"],
                field="name_counts_generation_id",
                context=context,
            ),
            pickle_sha256=_sha256(
                feature_contract["name_counts_pickle_sha256"],
                field="name_counts_pickle_sha256",
                context=context,
            ),
            source_snapshot_id=_nonempty_string(
                feature_contract["name_counts_source_snapshot_id"],
                field="name_counts_source_snapshot_id",
                context=context,
            ),
            selected_rows_sha256=_sha256(
                feature_contract["name_counts_selected_rows_sha256"],
                field="name_counts_selected_rows_sha256",
                context=context,
            ),
        )

    @classmethod
    def from_provenance(cls, provenance: Any, *, context: str) -> NameCountsBinding:
        """Read the corresponding identity directly from verified source provenance."""

        if not isinstance(provenance, Mapping) or provenance.get("schema_version") != "name_counts_provenance_v1":
            raise ValueError(f"{context} requires name_counts_provenance_v1 provenance")
        return cls(
            generation_id=_nonempty_string(
                provenance.get("generation_id"),
                field="generation_id",
                context=context,
            ),
            pickle_sha256=_sha256(
                provenance.get("pickle_sha256"),
                field="pickle_sha256",
                context=context,
            ),
            source_snapshot_id=_nonempty_string(
                provenance.get("source_snapshot_id"),
                field="source_snapshot_id",
                context=context,
            ),
            selected_rows_sha256=_sha256(
                provenance.get("selected_rows_sha256"),
                field="selected_rows_sha256",
                context=context,
            ),
        )

    @classmethod
    def from_arrow_name_counts_index(cls, index_dir: str | Path, *, context: str) -> NameCountsBinding:
        """Read the binding from a validated Arrow name-count index manifest."""

        manifest_path = Path(index_dir) / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("schema_version") != "name_counts_index_v1":
            raise ValueError(f"{context} requires a name_counts_index_v1 manifest: {manifest_path}")
        return cls.from_provenance(
            payload.get("source_provenance"),
            context=f"{context} source_provenance",
        )

    @classmethod
    def from_rust_featurizer(cls, rust_featurizer: Any, *, context: str) -> NameCountsBinding:
        """Read the runtime-only binding retained by an Arrow-built Rust featurizer."""

        binding = getattr(rust_featurizer, "name_counts_provenance_binding", None)
        if binding is None:
            raise ValueError(
                f"{context} requires a Rust featurizer with verified name-count provenance; "
                "rebuild it from a provenance-bearing name_counts_index"
            )
        if not isinstance(binding, tuple) or len(binding) != 4:
            raise ValueError(f"{context} Rust featurizer returned an invalid name-count provenance binding")
        generation_id, pickle_sha256, source_snapshot_id, selected_rows_sha256 = binding
        return cls(
            generation_id=_nonempty_string(
                generation_id,
                field="generation_id",
                context=context,
            ),
            pickle_sha256=_sha256(
                pickle_sha256,
                field="pickle_sha256",
                context=context,
            ),
            source_snapshot_id=_nonempty_string(
                source_snapshot_id,
                field="source_snapshot_id",
                context=context,
            ),
            selected_rows_sha256=_sha256(
                selected_rows_sha256,
                field="selected_rows_sha256",
                context=context,
            ),
        )

    def require_matches(self, observed: NameCountsBinding, *, context: str, source: str) -> None:
        """Reject a model/artifact generation mismatch before feature computation."""

        if observed != self:
            raise ValueError(
                f"{context} name-count binding mismatch for {source}: expected={self!r} observed={observed!r}"
            )
