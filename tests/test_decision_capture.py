from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from s2and.incremental_linking import production, runtime
from s2and.incremental_linking.gate_buckets import QueryView
from s2and.incremental_linking.seed_assignment import LinkPriority
from scripts.judging import judged_bank
from scripts.judging.decision_capture import DecisionCapture, targets_match
from tests.test_incremental_linking_runtime import (
    StaticArtifact,
    _empty_feature_matrix,
    _promoted_gate_config,
    _row_only_candidate_batch,
)
from tests.test_review_packets import frozen_snapshot as frozen_snapshot

pytestmark = pytest.mark.tooling


def _sid(number: int) -> str:
    return f"visibleq:s{number}"


def test_capture_final_decisions_and_build_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_snapshot: Path
) -> None:
    """Replay two attachment passes with window-local indices and fixed scores."""
    initial_seeds = {_sid(2): "a", _sid(3): "b", _sid(4): "c"}
    windows = {
        1: (("a", 0.96), ("a", 0.91), ("b", 0.80), ("c", 0.99)),
        5: (("b", 0.10), ("c", 0.99)),
        6: (("c", 0.99),),
        7: (),
        8: (("a", 0.95),),
        9: (("a", 0.93),),
        10: (("c", 0.40),),
        11: (("a", 0.90),),
        12: (("residual", 0.94),),
        13: (("c", 0.95),),
    }
    vetoes = {1: [0, 0, 0, 1], 6: [1]}
    links: dict[str, str] = {}
    priorities: dict[str, LinkPriority] = {}

    def score_fixture(*_args: object, raw_plan_bundle: SimpleNamespace, **_kwargs: object) -> SimpleNamespace:
        (query,) = raw_plan_bundle.query_signature_ids
        number = int(query.removeprefix("visibleq:s"))
        rows = windows[number]
        if rows:
            batch = _row_only_candidate_batch(
                row_query_signature_indices=np.zeros(len(rows), dtype=np.uint32),
                row_component_keys=tuple(key for key, _ in rows),
            )
            artifact = StaticArtifact(np.array([score for _, score in rows]), _promoted_gate_config(0.5))
            signals = {
                "first_name_bucket": np.array(["multi_letter_first"] * len(rows)),
                "constraint_disallow_count": np.array(vetoes.get(number, [0] * len(rows))),
                "constraint_disallow_fraction": np.array(vetoes.get(number, [0] * len(rows))),
                "constraint_pair_count": np.ones(len(rows)),
                "constraint_require_count": np.array([1, 0] if number == 5 else [0] * len(rows)),
                "constraint_name_check_disallow_count": np.array([number == 8] * len(rows)),
            }
            compact = runtime._predict_incremental_link_or_abstain_compact(
                artifact, _empty_feature_matrix(batch), row_signals=signals
            )
            (decision,) = compact.decisions
        else:
            (decision,) = runtime._no_candidate_abstain_decisions([0])
        linked = {}
        if decision.component_key is not None:
            linked[query] = links[query] = decision.component_key
            priorities[query] = LinkPriority(decision.forced, decision.score or 0.0)
        return SimpleNamespace(compact_result=SimpleNamespace(decisions=[decision]), linked_signature_clusters=linked)

    def score_window(number: int, seeds: Mapping[str, str], view: QueryView = "initial_only") -> None:
        # Each successive window has a different signature at local index zero.
        ids = (_sid(number),)
        runtime._predict_incremental_link_or_abstain_from_preplanned_raw_arrow(
            raw_plan_bundle=SimpleNamespace(
                signature_order=SimpleNamespace(signature_ids=ids), query_signature_ids=ids, query_views=(view,)
            ),
            rust_featurizer=SimpleNamespace(has_signature_order=lambda order: order == ids),
            partial_supervision_seed_signature_to_component=seeds,
        )

    def assign(seeds: Mapping[str, str], queries: list[int], *, restore: bool = False) -> None:
        split_members = {key: [sid for sid in seeds if seeds[sid] == key] for key in seeds.values()}
        recluster_map = {"c": "restored"} if restore else {}
        kept, _ = production.reject_conflicting_new_links(
            links,
            priorities=priorities,
            recluster_map=recluster_map,
            conflict=lambda left, right: {left, right} == {_sid(1), _sid(9)},
        )
        production.apply_seed_links(
            linked_signature_to_cluster=kept,
            unassigned_signature_ids=[_sid(number) for number in queries],
            recluster_map=recluster_map,
            cluster_seeds_require_inverse={recluster_map.get(key, key): ids for key, ids in split_members.items()},
            split_cluster_seeds_require_inverse=split_members if restore else None,
            prevent_new_incompatibilities=restore,
            first_names={_sid(4): "aaron", _sid(13): "bob"},
            name_tuples=frozenset(),
        )
        links.clear()
        priorities.clear()

    monkeypatch.setattr(runtime, "_predict_incremental_link_or_abstain_from_preplanned_raw_arrow", score_fixture)
    capture = DecisionCapture()
    with capture.patched():
        first = [1, 5, 6, 7, 8, 9, 10, 13]
        for number in first:
            score_window(number, initial_seeds, "full" if number == 10 else "initial_only")
        windows[10] = (("b", 0.30),)
        score_window(10, initial_seeds, "full")
        assign(initial_seeds, first, restore=True)
        grown = {**initial_seeds, _sid(1): "a", _sid(5): "b", _sid(6): "residual"}
        score_window(11, grown)
        score_window(12, grown)
        assign(grown, [11, 12])

    assert runtime._predict_incremental_link_or_abstain_from_preplanned_raw_arrow is score_fixture
    capture_dir = tmp_path / "captures"
    paths = capture.write(capture_dir, blocks={"a smith": [_sid(n) for n in range(1, 14)]}, eps=0.38)
    rows = pq.read_table(paths["a smith"]).to_pylist()
    by_id = {row["query_signature_id"]: row for row in rows}
    assert len(rows) == len(by_id) == 10
    assert set(by_id) == {_sid(n) for n in [*first, 11, 12]}
    first_link = by_id[_sid(1)]
    assert first_link["final_action"] == "link"
    assert first_link["candidate_component"] == "a"
    assert first_link["chosen_probability"] == 0.96
    assert (first_link["runner_up_component"], first_link["runner_up_score"]) == ("b", 0.80)
    forced = by_id[_sid(5)]
    assert (forced["final_action"], forced["exclusion_reason"], forced["target_component"]) == (
        "link",
        "forced_link",
        "b",
    )
    assert forced["chosen_probability"] == 0.10
    assert (forced["runner_up_component"], forced["runner_up_score"]) == ("c", 0.99)
    for number, reason in [
        (6, "all_vetoed"),
        (7, "no_candidate"),
        (8, "name_conflict"),
        (9, "in_pass_rejection"),
        (13, "name_conflict"),
    ]:
        row = by_id[_sid(number)]
        assert (row["final_action"], row["exclusion_reason"], row["target_signature_ids"]) == ("abstain", reason, None)
    assert by_id[_sid(10)]["query_view"] == "full"
    assert by_id[_sid(10)]["final_action"] == "abstain"
    assert by_id[_sid(10)]["exclusion_reason"] is None
    later_link = by_id[_sid(11)]
    assert later_link["pass_index"] == 1
    assert later_link["target_signature_ids"] == [_sid(1), _sid(2)]
    assert later_link["target_seed_signature_ids"] == [_sid(2)]
    assert by_id[_sid(12)]["target_seed_signature_ids"] == [_sid(6)]
    assert targets_match(first_link["target_seed_signature_ids"], later_link["target_seed_signature_ids"])

    strata = tmp_path / "strata.json"
    strata.write_text(
        json.dumps(
            {"package": "capture-fixture", "seed": 42, "quotas": {"full": 1, "initial_only": 3}, "controls": {}}
        ),
        encoding="utf-8",
    )
    key = tmp_path / "private/key.json"
    judged_bank.build_package(
        captured=capture_dir,
        strata=strata,
        bank=tmp_path / "bank",
        output_dir=tmp_path / "blind",
        key_path=key,
        snapshot=frozen_snapshot,
    )
    items = json.loads(key.read_text(encoding="utf-8"))["items"].values()
    assert {item["query_signature_id"] for item in items} == {_sid(n) for n in (1, 10, 11, 12)}
    abstention = next(item for item in items if item["query_signature_id"] == _sid(10))
    assert abstention["target_signature_ids"] == [_sid(3)]
    assert abstention["score"] == 0.30


def test_targets_match_on_seed_members() -> None:
    assert targets_match(("seed-a", "seed-b"), ("seed-b", "seed-c"))
    assert not targets_match(("seed-a",), ("seed-b",))
    assert not targets_match(("seed-a",), None)
    assert not targets_match((), ())
