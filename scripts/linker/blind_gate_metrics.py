"""Score query outcomes on fully judged blind gate windows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from s2and._sha256 import sha256_file
from s2and.incremental_linking.artifact import INCREMENTAL_LINKER_FEATURE_CONTRACT_VERSION
from s2and.incremental_linking_training.gate import (
    _FALSE_ABSTAIN_REPORT_WEIGHTS,
    _apply_promoted_logistic_gate_to_candidate_rows,
    _summarize_predictions,
    report_logistic_gate_error_weights,
)
from s2and.incremental_linking_training.ranker import (
    _classic_feature_matrix,
    _score_query_choices,
)
from s2and.production_model import load_production_model
from s2and.production_training_contract import load_packaged_artifact_authority
from scripts.common.arrow_roots import read_json_object
from scripts.common.block_roles import GATE_BINDINGS_FIELD, GATE_CANDIDATE_ORCIDS, BlockRoles, load_block_roles
from scripts.common.linker_seeds import GATE_SEED_DIGESTS, members_by_component, read_frozen_seeds
from scripts.linker.train_linker_and_finalize import (
    DEFAULT_TOTAL_RAM_BYTES,
    _materialize_frozen_linker_tasks,
    _positive_int,
)


def with_query_gold_scored(choices: pd.DataFrame, candidate_rows: pd.DataFrame) -> pd.DataFrame:
    """Carry window-level eligibility alongside the chosen candidate's judgment."""
    eligibility = candidate_rows.drop_duplicates("query_group_id").set_index("query_group_id")["query_gold_scored"]
    return choices.assign(query_gold_scored=choices["query_case_id"].map(eligibility))


def summarize_blind_gate_predictions(predictions: pd.DataFrame) -> dict:
    """Exclude incomplete windows from query metrics."""
    resolved = predictions.loc[predictions["query_gold_scored"]]
    return {**_summarize_predictions(resolved), "n_unscored_queries": len(predictions) - len(resolved)}


def _apply_blind_gate_labels(
    rows: pd.DataFrame, report: Mapping[str, Any], roles: BlockRoles, blocks: Mapping[str, Any]
) -> pd.DataFrame:
    """Attach agreed blind outcomes to the matching frozen production window."""
    if not rows["label"].isna().all() or rows["query_gold_scored"].any() or rows["candidate_gold_scored"].any():
        raise ValueError("Frozen production requests must have no pre-existing labels")
    vetoes = rows.get("candidate_has_vetoed_identity")
    if vetoes is None or not pd.api.types.is_bool_dtype(vetoes) or vetoes.isna().any():
        raise ValueError("Frozen gate tasks must record candidate identity vetoes. Rebuild the tasks.")
    keys = ["dataset", "query_signature_id", "candidate_component_key"]
    judged = pd.DataFrame.from_records(report.get("gate_thresh_labels", []), columns=[*keys, "verdict"])
    if judged.empty or judged.duplicated(keys).any() or not judged["verdict"].isin({"same", "different"}).all():
        raise ValueError("Expected unique, agreed gate threshold verdicts")
    indexed = pd.MultiIndex.from_frame(rows[keys])
    decisions = judged.set_index(keys)["verdict"]
    if not indexed.is_unique or len(decisions.index.difference(indexed)):
        raise ValueError("Gate threshold judgments contain candidates outside the frozen windows")
    candidates = report.get(GATE_CANDIDATE_ORCIDS)
    if (
        not isinstance(candidates, dict)
        or set(judged["dataset"]) - candidates.keys()
        or report.get(GATE_BINDINGS_FIELD) != roles.gate_judgment_bindings(candidates)
    ):
        raise ValueError("Gate threshold judgments block provenance does not match the identity ledger")
    seed_digests = report.get(GATE_SEED_DIGESTS)
    if (
        not isinstance(seed_digests, dict)
        or candidates.keys() != seed_digests.keys()
        or any(block not in blocks or digest != blocks[block]["seeds_sha256"] for block, digest in seed_digests.items())
    ):
        raise ValueError("Gate threshold judgments do not match the frozen seed assignments")
    for block, block_rows in rows.groupby("dataset", sort=False):
        members = members_by_component(read_frozen_seeds(blocks[str(block)]))
        if missing := set(block_rows["candidate_component_key"]) - members.keys():
            raise ValueError(f"Gate candidates have no frozen seed members in {block!r}: {sorted(missing)}")
    labeled = rows.copy()
    verdicts = decisions.reindex(indexed).reset_index(drop=True)
    known = verdicts.notna() & ~vetoes.reset_index(drop=True)
    labeled["label"] = (verdicts.eq("same") & known).astype("int8").to_numpy()
    labeled["candidate_gold_scored"] = known.to_numpy()
    grouped = labeled.groupby("query_group_id", sort=False)
    labeled["query_gold_scored"] = grouped["candidate_gold_scored"].transform("all")
    labeled["gold_safe_target"] = grouped["label"].transform("any")
    return labeled


def main(argv: Sequence[str] | None = None) -> None:
    """Score a fitted bundle against bound blind judgments for frozen windows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--linker-task-manifest", type=Path, required=True)
    parser.add_argument("--block-role-assignments", type=Path, required=True)
    parser.add_argument("--gate-thresh-judgments", type=Path, required=True)
    parser.add_argument("--name-counts-index-root", type=Path, required=True)
    parser.add_argument("--n-jobs", type=_positive_int, required=True)
    parser.add_argument("--total-ram-bytes", type=_positive_int, default=DEFAULT_TOTAL_RAM_BYTES)
    args = parser.parse_args(argv)

    authority = load_packaged_artifact_authority(name_counts_index_root=args.name_counts_index_root)
    model = load_production_model(args.model_path, n_jobs=args.n_jobs, expected_artifact_hashes=authority.hashes)
    artifact = model.incremental_linker_artifact
    if artifact is None or artifact.feature_contract_version != INCREMENTAL_LINKER_FEATURE_CONTRACT_VERSION:
        raise ValueError("Expected a fitted linker bundle using the current feature contract")
    roles = load_block_roles(args.block_role_assignments)
    manifest = read_json_object(args.linker_task_manifest)
    if roles.evidence_sha256 is None or manifest.get("gate_label_source") != "blind_judging":
        raise ValueError("Expected ledger-backed gate_thresh windows and blind judgments")
    table = manifest["tables"].get("gate_thresh")
    if table is None or artifact.retrieval_top_k != manifest["retrieval_top_k"]:
        raise ValueError("Fitted linker and frozen gate_thresh windows need matching retrieval limits")

    tasks, _ = _materialize_frozen_linker_tasks(
        args.linker_task_manifest,
        roles,
        target_features=artifact.feature_columns,
        name_tuples=authority.name_tuples.pairs,
        clusterer=model,
        n_jobs=args.n_jobs,
        total_ram_bytes=args.total_ram_bytes,
        max_query_groups=None,
        role="gate_thresh",
    )
    rows = _apply_blind_gate_labels(
        tasks["gate_thresh"], read_json_object(args.gate_thresh_judgments), roles, table["blocks"]
    )
    matrix = _classic_feature_matrix(rows, artifact.feature_columns).to_numpy(dtype=np.float32)
    probabilities = artifact.predict_probabilities(np.ascontiguousarray(matrix), num_threads=args.n_jobs)
    choices = _score_query_choices(rows, probabilities, query_id_column="query_group_id", include_margin=True)
    choices = with_query_gold_scored(choices, rows)
    gate = artifact.gate_model
    predictions = _apply_promoted_logistic_gate_to_candidate_rows(rows, choices, probabilities, gate)
    requested = sum(block["counts"]["requested_queries"] for block in table["blocks"].values())
    judged = int(choices["chosen_candidate_gold_scored"].sum())
    gate_config = read_json_object(artifact.artifact_dir / "metadata.json")["gate_config"]
    result = {
        "model_manifest_sha256": sha256_file(args.model_path / "manifest.json"),
        "task_manifest_sha256": sha256_file(args.linker_task_manifest),
        "judgments_sha256": sha256_file(args.gate_thresh_judgments),
        "judged_choice_coverage": {
            "requested_queries": requested,
            "queries_with_choices": len(choices),
            "judged_choices": judged,
            "share_of_choices_judged": judged / len(choices),
            "share_of_requests_with_judged_choice": judged / requested,
        },
        "resolved_window_metrics": summarize_blind_gate_predictions(predictions),
        "false_abstain_weight_report": report_logistic_gate_error_weights(
            gate_config, rows, choices, probabilities, false_abstain_weights=_FALSE_ABSTAIN_REPORT_WEIGHTS
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
