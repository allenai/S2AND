# Joint Safe-Link Handoff

The single active official bundle is:

- [data/joint_safe_link_official_stack_20260428p](data/joint_safe_link_official_stack_20260428p)

The stable code surface is:

- [scripts/joint_safe_link_official_stack.py](scripts/joint_safe_link_official_stack.py)
- [scripts/joint_safe_link_dataset_contract.py](scripts/joint_safe_link_dataset_contract.py)
- [scripts/compile_joint_safe_link_dataset_contract.py](scripts/compile_joint_safe_link_dataset_contract.py)
- [scripts/rebuild_joint_safe_link_official_stack.py](scripts/rebuild_joint_safe_link_official_stack.py)
- [scripts/run_joint_safe_link_official_classic.py](scripts/run_joint_safe_link_official_classic.py)
- [scripts/sync_joint_safe_link_official_bundle_metadata.py](scripts/sync_joint_safe_link_official_bundle_metadata.py)
- [scripts/validate_joint_safe_link_official_stack.py](scripts/validate_joint_safe_link_official_stack.py)

The old scratch promotion, split-surgery, and reconciliation scripts are historical only. The latest code archive is:

- [scratch/archived_joint_safe_link_cleanup_20260425](scratch/archived_joint_safe_link_cleanup_20260425)

## Canonical Commands

Rebuild the active bundle in place:

```powershell
uv run python scripts\rebuild_joint_safe_link_official_stack.py
```

Replay train/calibration/test from the frozen bundle:

```powershell
uv run python scripts\run_joint_safe_link_official_classic.py
```

Validate bundle structure, paths, feature coverage, gate-artifact consistency, and no-self-containing-candidate invariant:

```powershell
uv run python scripts\validate_joint_safe_link_official_stack.py
```

Run the shared regression tests:

```powershell
uv run pytest -q tests\test_joint_safe_link_official_stack.py
```

## What `20260428p` Is

`20260428p` is the active official classic stack after the full-surface rebuild, the reviewed data-cleanup integrations for:

- `a_silva`
- `j_smith`
- `a_khan`
- `s_gupta`
- `s2and`
- `hwang_clean`
- `s_park`
- `s_lee`

and the focused `2026-04-21` `a_silva` reopen eval corrections.

For `s2and`, all active calibration/test query groups have been manually packet-reviewed. The relabel keeps
manual multiple-safe-positive labels, and the current no-self-candidate rebuild leaves 1,408 active eval query
groups with 195 positive rows.
For `hwang_clean`, reviewed corrections are now applied at candidate level after self-containing candidate removal:
only surviving reviewed positive candidates count as positives, `should_abstain` corrections force no-positive labels,
and the cleaned query target is regenerated from candidate row labels.
An additional S2AND-source training holdout has been manually reviewed: 16 all-negative training query groups were
removed from training, and 15 possible groups were added to the promoted stratified calibration/test split.
The unused original S2AND construction pool was also revisited for calibration signal: 43 non-overlapping manually
reviewed no-safe-link groups were attempted as `s2and_extra_no_positive`; after removing self-containing candidate
rows, only 1 no-positive query group remains usable.
The active classic model uses the `best_by_test_minus_new15` feature set with the random_06 LightGBM
hyperparameter family at 1,500 trees.

It keeps classic evaluation frozen at `w5` and `w25`, and `bundle.json` remains the single source of truth for bundle metadata, model spec, and frozen metrics.
The first centralized dataset contract now lives under `dataset_contract/`: it records the canonical candidate
filter policy and a normalized custom-label ledger, with a comparison report showing 0 fatal mismatches against
the current active slices it covers.
The rebuild, validator, and row materialization now call shared dataset-contract filter helpers for the top-25
retrieval cap, self-containing candidate removal, and hard-disallow component removal. Hard-disallow still runs
after pairwise scoring because it depends on scored constraint labels.

## Frozen Bundle Shape

The bundle root contains:

- `bundle.json`
- `README.md`
- `PROVENANCE.md`
- `training/classic_train_union21_plus_s_lee_raw_rows.csv.gz`
- `calibration/classic_gate_possible_manual_w5_rows.csv.gz`
- `calibration/classic_gate_possible_manual_w5_base_groups.csv`
- `calibration/stratified_eval_test_split/combined_query_split_assignments.csv`
- `calibration/stratified_eval_test_split/stratum_balance.csv`
- `calibration/stratified_eval_test_split/summary.json`
- `calibration/stratified_eval_test_split/report.md`
- `calibration/total_error_4score_2margin_gate/selected_gate.json`
- `calibration/total_error_4score_2margin_gate/gate_candidate_metrics.csv`
- `calibration/total_error_4score_2margin_gate/summary.json`
- `calibration/total_error_4score_2margin_gate/report.md`
- `calibration/best_by_check_minus_new26_feature_selection/new_feature_ablation.csv`
- `calibration/best_by_check_minus_new26_feature_selection/summary.json`
- `calibration/best_by_check_minus_new26_feature_selection/trajectory.csv`
- `dataset_contract/filter_policy.json`
- `dataset_contract/custom_label_ledger.csv`
- `dataset_contract/custom_label_ledger_summary.json`
- `dataset_contract/custom_label_ledger_comparison.json`
- `dataset_contract/custom_label_ledger_report.md`
- `test/classic_gate_internal_eval_base_groups.csv`
- `test/s2and_eval_rows.csv.gz`
- `test/hwang_eval_rows.csv.gz`
- `test/hwang_cleaned_eval_overrides.csv`
- `test/hwang_candidate_level_label_overrides.csv`
- `test/hwang_candidate_level_label_overrides_summary.json`
- `test/s_park_eval_rows.csv.gz`
- `test/s_lee_eval_rows.csv.gz`
- `test/j_smith_eval_rows.csv.gz`
- `test/a_khan_eval_rows.csv.gz`
- `test/a_silva_eval_rows.csv.gz`
- `test/s_gupta_eval_rows.csv.gz`
- `test/training_s2and_source_reviewed_eval_rows.csv.gz`
- `test/s2and_extra_no_positive_eval_rows.csv.gz`
- `training/singleton_near_distance_repair_manifest.csv`
- `training/singleton_near_distance_quarantined_query_groups.txt`

## Current Counts

Persisted row counts in [data/joint_safe_link_official_stack_20260428p/bundle.json](data/joint_safe_link_official_stack_20260428p/bundle.json):

- training: `413,866` rows, `81,692` queries, `25,519` positive rows
- calibration source: `135,130` rows, `8,132` queries, `3,270` positive rows
- calibration split: `4,201` calibration groups, `4,204` internal eval groups
- promoted stratified split: `18,949` query groups; `4,740` calibration_fit, `4,736` calibration_check, `9,473` held-out test
- S2AND eval: `1,408` queries, `9,943` rows, `195` positive rows
- extra S2AND no-positive calibration source: `1` query, `1` row, `0` positive rows
- H-Wang eval: `5,078` queries, `117,417` rows, `2,728` positive rows
- H-Wang clean overrides: `5,078` queries, `1,786` positive overrides
- H-Wang candidate-level relabel manifest: `5,078` queries, `401` reviewed positive corrections, `129` surviving reviewed positives
- S-Park eval: `3,411` queries, `57,954` rows, `1,224` positive rows
- S-Lee eval: `4,032` queries, `71,441` rows, `1,081` positive rows
- J-Smith eval: `70` queries, `980` rows, `14` positive rows
- A-Khan eval: `149` queries, `2,048` rows, `22` positive rows
- A-Silva eval: `642` queries, `4,467` rows, `305` positive rows
- S-Gupta eval: `139` queries, `1,956` rows, `18` positive rows
- reviewed S2AND-source training holdout: `15` queries, `50` rows, `6` positive rows
- dataset contract custom/manual-label ledger: `29,155` rows across H-Wang, S2AND full relabel, new-block calibration/eval labels, S-Lee/S-Park eval labels, S2AND-source holdout, extra S2AND no-positive, and singleton training repair; comparison fatal mismatches `0`

## Frozen Reference Metrics

The exact frozen metrics live in [data/joint_safe_link_official_stack_20260428p/bundle.json](data/joint_safe_link_official_stack_20260428p/bundle.json).

| Model | S2AND w5 | S2AND w25 | H-Wang clean w5 | H-Wang clean w25 | S-Park w5 | S-Park w25 | S-Lee w5 | S-Lee w25 | J-Smith w5 | J-Smith w25 | A-Khan w5 | A-Khan w25 | A-Silva w5 | A-Silva w25 | S-Gupta w5 | S-Gupta w25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `classic` | `0.5810` | `0.5756` | `0.9111` | `0.9882` | `0.9994` | `0.9995` | `0.9916` | `0.9926` | `0.9545` | `0.9643` | `1.0000` | `1.0000` | `0.9789` | `0.9766` | `1.0000` | `1.0000` |

Reviewed S2AND-source training holdout:

- `w5 balanced_accuracy = 0.5000`
- `w25 balanced_accuracy = 0.5000`

Extra S2AND no-positive calibration source:

- `w5 balanced_accuracy = 0.5000`
- `w25 balanced_accuracy = 0.5000`

Promoted stratified held-out test:

- `accuracy = 0.9870`
- `balanced_accuracy = 0.9806`
- `error_rate = 0.0130`
- `errors = 123 / 9473`

Frozen abstain-rule thresholds:

- `multi_candidate|multi_letter_first score_threshold = 0.8255340123176578`
- `multi_candidate|single_letter_first score_threshold = 0.02939787022769439`
- `single_candidate|multi_letter_first score_threshold = 0.08104255050420771`
- `single_candidate|single_letter_first score_threshold = 0.009219343611039243`
- `multi_candidate|multi_letter_first margin_threshold = 0.998066394450143`
- `multi_candidate|single_letter_first margin_threshold = 0.4112226212979295`

## Verification Artifacts

The current replay outputs are:

- [scratch/joint_safe_link_official_classic_20260428p/summary.json](scratch/joint_safe_link_official_classic_20260428p/summary.json)
- [scratch/joint_safe_link_official_classic_20260428p/verification.json](scratch/joint_safe_link_official_classic_20260428p/verification.json)
- [data/joint_safe_link_official_stack_20260428p/calibration/stratified_eval_test_split/report.md](data/joint_safe_link_official_stack_20260428p/calibration/stratified_eval_test_split/report.md)
- [data/joint_safe_link_official_stack_20260428p/calibration/total_error_4score_2margin_gate/report.md](data/joint_safe_link_official_stack_20260428p/calibration/total_error_4score_2margin_gate/report.md)
- [scratch/classic_bundle_error_diff_20260421_n_vs_p/summary.json](scratch/classic_bundle_error_diff_20260421_n_vs_p/summary.json)

## Cleanup Boundary

If we want to delete old datasets and scripts safely, the target surface to keep is:

- [data/joint_safe_link_official_stack_20260428p](data/joint_safe_link_official_stack_20260428p)
- [scripts/joint_safe_link_official_stack.py](scripts/joint_safe_link_official_stack.py)
- [scripts/joint_safe_link_dataset_contract.py](scripts/joint_safe_link_dataset_contract.py)
- [scripts/compile_joint_safe_link_dataset_contract.py](scripts/compile_joint_safe_link_dataset_contract.py)
- [scripts/rebuild_joint_safe_link_official_stack.py](scripts/rebuild_joint_safe_link_official_stack.py)
- [scripts/run_joint_safe_link_official_classic.py](scripts/run_joint_safe_link_official_classic.py)
- [scripts/sync_joint_safe_link_official_bundle_metadata.py](scripts/sync_joint_safe_link_official_bundle_metadata.py)
- [scripts/validate_joint_safe_link_official_stack.py](scripts/validate_joint_safe_link_official_stack.py)
- [tests/test_joint_safe_link_official_stack.py](tests/test_joint_safe_link_official_stack.py)

Everything else in the older `20260428*` chain is now historical lineage rather than active runtime surface.
Legacy helper scripts/tests removed from the active surface in this cleanup are preserved under
[scratch/archived_joint_safe_link_cleanup_20260425](scratch/archived_joint_safe_link_cleanup_20260425).
