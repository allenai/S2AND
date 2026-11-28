"""Capture the final raw Arrow linker decision for each query in a production replay."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal
from unittest.mock import patch
from urllib.parse import quote

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from s2and.incremental_linking import production, runtime
from s2and.incremental_linking.gate_buckets import QueryView
from s2and.incremental_linking.retrieval import RawArrowPlanBundle
from s2and.incremental_linking.seed_assignment import SeedLinkAssignment

Exclusion = Literal["forced_link", "all_vetoed", "no_candidate", "name_conflict", "in_pass_rejection"]

_SCHEMA = pa.schema(
    [
        ("block", pa.string()),
        ("query_signature_id", pa.string()),
        ("query_view", pa.string()),
        ("eps", pa.float64()),
        ("pass_index", pa.int64()),
        ("final_action", pa.string()),
        ("candidate_component", pa.string()),
        ("chosen_probability", pa.float64()),
        ("candidate_signature_ids", pa.list_(pa.string())),
        ("runner_up_component", pa.string()),
        ("runner_up_score", pa.float64()),
        ("exclusion_reason", pa.string()),
        ("target_component", pa.string()),
        ("target_signature_ids", pa.list_(pa.string())),
        ("target_seed_signature_ids", pa.list_(pa.string())),
    ]
)


def targets_match(left: Collection[str] | None, right: Collection[str] | None) -> bool:
    """Return whether two linked targets share a decision-time seed signature."""
    return left is not None and right is not None and not set(left).isdisjoint(right)


@dataclass(frozen=True)
class _Choice:
    component: str | None = None
    probability: float | None = None
    runner_up_component: str | None = None
    runner_up_score: float | None = None
    exclusion: Exclusion | None = None


def _choice(decision: runtime.LinkOrAbstainDecision, allowed_rows: np.ndarray, options: Mapping[str, Any]) -> _Choice:
    if not len(allowed_rows):
        return _Choice(exclusion="no_candidate")
    vetoes = options["constraint_vetoes"]
    eligible = allowed_rows if vetoes is None else allowed_rows[~vetoes[allowed_rows]]
    if not decision.forced and not len(eligible):
        return _Choice(exclusion="all_vetoed")

    # Compact decisions hide the winning row on abstention.
    best = decision.row_index
    if best is None:
        best = (options["subset_gate"] or options["fast_path"])[0]
    probabilities = options["probabilities"]
    keys = options["component_keys"]
    component = keys[best]
    runner = max((row for row in eligible if keys[row] != component), key=probabilities.__getitem__, default=None)
    conflicts = options["name_conflict_rows"]
    exclusion: Exclusion | None = None
    if decision.forced:
        exclusion = "forced_link"
    elif conflicts is not None and conflicts[best]:
        # A hard name conflict also excludes a candidate whose gate abstained.
        exclusion = "name_conflict"
    return _Choice(
        str(component),
        float(probabilities[best]),
        None if runner is None else str(keys[runner]),
        None if runner is None else float(probabilities[runner]),
        exclusion,
    )


@dataclass(frozen=True)
class _Component:
    key: str
    members: tuple[str, ...]
    seed_members: tuple[str, ...]


@dataclass(frozen=True)
class _Decision:
    view: QueryView
    pass_index: int
    choice: _Choice
    candidate: _Component | None
    target: _Component | None


class DecisionCapture:
    """Observe one sequential production replay and keep its last decision per signature."""

    def __init__(self) -> None:
        self._pending: dict[str, _Decision] = {}
        self._final: dict[str, _Decision] = {}
        self._rejected: set[str] = set()
        self._attached: set[str] = set()
        self._seeds: Mapping[str, str | int] | None = None
        self._components: dict[str, _Component] = {}
        self._pass_index = 0

    @contextmanager
    def patched(self) -> Iterator[None]:
        """Observe scoring and seed assignment in a sequential, single-process replay."""
        original_decide = runtime._decide_query_rows
        original_compact = runtime._predict_incremental_link_or_abstain_compact
        original_score = runtime._predict_incremental_link_or_abstain_from_preplanned_raw_arrow
        original_reject = production.reject_conflicting_new_links
        original_apply = production.apply_seed_links

        def score(*args: Any, **kwargs: Any) -> runtime.LinkOrAbstainProductionResult:
            # Featurizer indices may repeat in the next window. Resolve them here.
            choices: dict[int, _Choice] = {}

            def decide(rows: np.ndarray, **options: Any) -> runtime.LinkOrAbstainDecision:
                decision = original_decide(rows, **options)
                choices[decision.query_signature_index] = _choice(decision, rows, options)
                return decision

            def compact(*args: Any, **kwargs: Any) -> runtime.LinkOrAbstainCompactResult:
                with patch.object(runtime, "_decide_query_rows", decide):
                    return original_compact(*args, **kwargs)

            with patch.object(runtime, "_predict_incremental_link_or_abstain_compact", compact):
                result = original_score(*args, **kwargs)
            self._record_batch(
                result,
                choices,
                kwargs["raw_plan_bundle"],
                kwargs["rust_featurizer"],
                kwargs["partial_supervision_seed_signature_to_component"],
            )
            return result

        def reject(linked: Mapping[str, int | str], **kwargs: Any) -> tuple[dict[str, int | str], list[str]]:
            kept, rejected = original_reject(linked, **kwargs)
            self._rejected.update(rejected)
            return kept, rejected

        def apply(**kwargs: Any) -> SeedLinkAssignment:
            result = original_apply(**kwargs)
            self._finish_pass(kwargs["linked_signature_to_cluster"], result)
            return result

        with (
            patch.object(runtime, "_predict_incremental_link_or_abstain_from_preplanned_raw_arrow", score),
            patch.object(production, "reject_conflicting_new_links", reject),
            patch.object(production, "apply_seed_links", apply),
        ):
            yield
        if self._pending:
            raise ValueError("Replay ended before assigning the captured linker decisions")

    def _record_batch(
        self,
        result: runtime.LinkOrAbstainProductionResult,
        choices: dict[int, _Choice],
        plan: RawArrowPlanBundle,
        featurizer: Any,
        seeds: Mapping[str, str | int],
    ) -> None:
        if seeds is not self._seeds:
            self._seeds = seeds
            grouped: dict[str, list[str]] = defaultdict(list)
            for sid, component in seeds.items():
                grouped[str(component)].append(sid)
            self._components = {}
            for key, members in grouped.items():
                ordered = tuple(sorted(members))
                roots = tuple(sid for sid in ordered if sid not in self._attached)
                self._components[key] = _Component(key, ordered, roots)

        order = plan.signature_order.signature_ids
        ids = (
            order if getattr(featurizer, "has_signature_order", lambda _: False)(order) else featurizer.signature_ids()
        )
        views = dict(zip(plan.query_signature_ids, plan.query_views, strict=True))
        linked = result.linked_signature_clusters
        for decision in result.compact_result.decisions:
            index = decision.query_signature_index
            sid = str(ids[index])
            view = views.pop(sid)
            choice = choices.pop(index, _Choice(exclusion="no_candidate"))
            candidate = None if choice.component is None else self._components[choice.component]
            target = self._components[str(linked[sid])] if sid in linked else None
            self._pending[sid] = _Decision(view, self._pass_index, choice, candidate, target)
        if choices or views:
            raise ValueError("Raw Arrow scoring did not capture one decision per query in the plan")

    def _finish_pass(self, linked: Mapping[str, int | str], result: SeedLinkAssignment) -> None:
        rejected_by_seed_check = set(result.rejected_signature_ids)
        for sid, decision in self._pending.items():
            if sid in self._rejected:
                decision = replace(
                    decision, target=None, choice=replace(decision.choice, exclusion="in_pass_rejection")
                )
            elif sid in rejected_by_seed_check:
                decision = replace(decision, target=None, choice=replace(decision.choice, exclusion="name_conflict"))
            elif decision.target is not None:
                if str(linked.get(sid)) != decision.target.key:
                    raise ValueError(f"Seed assignment changed the scored target for signature {sid!r}")
                self._attached.add(sid)
            self._final[sid] = decision
        self._pending.clear()
        self._rejected.clear()
        self._seeds = None
        self._components.clear()
        self._pass_index += 1

    def write(self, output_dir: Path, *, blocks: Mapping[str, Sequence[str]], eps: float) -> dict[str, Path]:
        """Publish one Parquet table per block, including empty tables for bulk-only blocks."""
        expected = {sid for ids in blocks.values() for sid in ids}
        if len(expected) != sum(map(len, blocks.values())):
            raise ValueError("Replay blocks must contain distinct signature IDs")
        if extra := self._final.keys() - expected:
            raise ValueError(f"Captured signatures outside the replay request: {sorted(extra)[:10]}")
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {block: output_dir / f"block-{quote(block, safe='')}.parquet" for block in blocks}
        for block, path in paths.items():
            rows = []
            for sid in sorted(blocks[block]):
                if (decision := self._final.get(sid)) is None:
                    continue
                choice, candidate, target = decision.choice, decision.candidate, decision.target
                rows.append(
                    {
                        "block": block,
                        "query_signature_id": sid,
                        "query_view": decision.view,
                        "eps": eps,
                        "pass_index": decision.pass_index,
                        "final_action": "abstain" if target is None else "link",
                        "candidate_component": choice.component,
                        "chosen_probability": choice.probability,
                        "candidate_signature_ids": None if candidate is None else candidate.members,
                        "runner_up_component": choice.runner_up_component,
                        "runner_up_score": choice.runner_up_score,
                        "exclusion_reason": choice.exclusion,
                        "target_component": None if target is None else target.key,
                        "target_signature_ids": None if target is None else target.members,
                        "target_seed_signature_ids": None if target is None else target.seed_members,
                    }
                )
            with TemporaryDirectory(dir=output_dir) as staging:
                table_path = Path(staging) / path.name
                pq.write_table(pa.Table.from_pylist(rows, schema=_SCHEMA), table_path, compression="zstd")
                os.link(table_path, path)
        return paths
