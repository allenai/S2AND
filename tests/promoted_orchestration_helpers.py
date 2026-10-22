"""Input builders for promoted-linker orchestration tests."""

import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import s2and.incremental_linking.production as production_module
import s2and.model as model_module
from s2and.arrow_inputs import ArrowDataset
from s2and.incremental_linking.feature_block import (
    read_cluster_seed_disallows_arrow,
    read_incremental_query_signatures_arrow,
)
from s2and.incremental_linking.retrieval import RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS
from tests.helpers import write_minimal_arrow_prediction_bundle


def _minimal_arrow_dataset(tmp_path: Path) -> ArrowDataset:
    write_minimal_arrow_prediction_bundle(tmp_path)
    return ArrowDataset.open(tmp_path)


def _patch_fake_raw_arrow_planner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    captured: dict[str, Any] | None = None,
) -> None:
    """Install a planner-shaped Rust fake for promoted raw Arrow orchestration tests."""

    class FakePlanner:
        def __init__(self, resource: object, query_signature_ids: list[str], **_kwargs: object):
            self._resource = resource
            self._query_signature_ids = tuple(query_signature_ids)
            self._name_counts_index = object()
            if captured is not None:
                captured.setdefault("planner_inits", []).append(self._query_signature_ids)
                captured.setdefault("planner_refs", []).append(weakref.ref(self))
                captured["planner_name_counts_index"] = self._name_counts_index

        @classmethod
        def from_query_signatures(
            cls,
            resource: object,
            query_signatures_path: str,
            **kwargs: object,
        ) -> "FakePlanner":
            rows = read_incremental_query_signatures_arrow(Path(query_signatures_path))
            return cls(resource, [row.signature_id for row in rows], **kwargs)

        @classmethod
        def from_auto_queries(
            cls,
            resource: object,
            _cluster_seeds_path: str,
            **kwargs: object,
        ) -> "FakePlanner":
            if captured is not None:
                disallow_path = kwargs.get("cluster_seed_disallows_path")
                captured.setdefault("planner_disallows", []).append(
                    set() if disallow_path is None else set(read_cluster_seed_disallows_arrow(Path(str(disallow_path))))
                )
            return cls(resource, [], **kwargs)

        def build_telemetry(self):
            return {
                "query_signature_count": len(self._query_signature_ids),
                "signature_count": len(self._query_signature_ids),
            }

        def name_counts_index(self) -> object:
            if captured is not None:
                captured["planner_name_counts_index_calls"] = (
                    int(captured.get("planner_name_counts_index_calls", 0)) + 1
                )
            return self._name_counts_index

        def build_featurizer(self, *_args: object, **_kwargs: object) -> None:
            # Hold no rows, so window featurizers come from the Arrow builder these tests watch.
            return None

        def plan(
            self,
            query_signature_ids: list[str],
            additional_cluster_seed_disallows: list[tuple[str, str]] | None = None,
            **_kwargs: object,
        ):
            query_ids = tuple(query_signature_ids)
            if captured is not None:
                captured.setdefault("planner_plans", []).append(query_ids)
                captured.setdefault("planner_plan_disallows", []).append(set(additional_cluster_seed_disallows or ()))
            plan: dict[str, Any] = {
                "query_signature_ids": query_ids,
                "query_views": ["full"] * len(query_ids),
                "query_authors": [""] * len(query_ids),
                "row_count": 0,
                "pair_count": 0,
                "row_query_signature_indices": np.asarray([], dtype=np.uint32),
                "row_component_keys": [],
                "retrieval_scores": np.asarray([], dtype=np.float32),
                "retrieval_ranks": np.asarray([], dtype=np.uint16),
                "pair_row_indices": np.asarray([], dtype=np.uint32),
                "pair_signature_ids": list(query_ids),
                "left_pair_signature_indices": np.asarray([], dtype=np.uint32),
                "right_pair_signature_indices": np.asarray([], dtype=np.uint32),
                "component_members": {},
            }
            for raw_key, _signal_key, dtype in RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS:
                plan[raw_key] = np.asarray([], dtype=dtype)
            return plan

    class FakeRustModule:
        RawBlockQueryCandidatePlanner = FakePlanner

    monkeypatch.setattr(production_module.feature_port, "_require_rust_runtime", lambda: FakeRustModule())
    monkeypatch.setattr(
        production_module.feature_port,
        "build_rust_featurizer_from_arrow_dataset",
        lambda *_args, **_kwargs: object(),
    )


def _mock_promoted_limits(
    *,
    query_batch_size: int = 1,
    predicted_peak_delta_bytes: int = 2_000,
    predicted_peak_rss_bytes: int = 3_000,
) -> model_module.memory_budget.PromotedPhaseALimits:
    return model_module.memory_budget.PromotedPhaseALimits(
        query_batch_size=int(query_batch_size),
        predicted_peak_delta_bytes=int(predicted_peak_delta_bytes),
        predicted_peak_rss_bytes=int(predicted_peak_rss_bytes),
    )
