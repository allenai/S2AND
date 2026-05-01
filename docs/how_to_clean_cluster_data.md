# How To Clean Cluster Data

This document records the exact process used to clean `a_silva` and rerun the official classic bundle so the same workflow can be repeated on other slices.

The `a_silva` cleanup had two surfaces:

- held-out public eval: [data/joint_safe_link_official_stack_20260428p/test/a_silva_eval_rows.csv.gz](data/joint_safe_link_official_stack_20260428p/test/a_silva_eval_rows.csv.gz)
- reviewed gate source: [data/joint_safe_link_official_stack_20260428p/calibration/classic_gate_possible_manual_w5_rows.csv.gz](data/joint_safe_link_official_stack_20260428p/calibration/classic_gate_possible_manual_w5_rows.csv.gz)

The current active bundle carrying this workflow forward is [data/joint_safe_link_official_stack_20260428p](data/joint_safe_link_official_stack_20260428p).

## What We Actually Did

For each model error, we reviewed:

- the raw query-signature metadata
- the top-ranked predicted cluster
- the correct cluster if a distinct positive cluster existed
- the top model candidates and raw ranked rows

Then we assigned one of four decisions:

1. `this label is correct, definetly`
2. `i am pretty sure this label is wrong and we don't need a web search`
3. `i am pretty sure this label is wrong and we needed a web search to confirm`
4. `I am just NOT sure here and i don't expect any lightgbm model to ever get this right, regardless of the feature space`

For the cleanup bundle we used this policy:

- decision `1`: keep the query as-is
- decisions `2`, `3`: apply a manual correction to the reviewed query group
- decision `4`: drop the query or base group from the cleaned bundle

For `a_silva`, the correction mapping was:

- `negative_false_positive_link` + decision `2/3` -> `top1_should_link`
- `positive_wrong_link` + decision `2/3` -> `top1_should_link`
- `positive_false_abstain_top1_wrong` + decision `2/3` -> `top1_should_link`
- `positive_false_abstain_top1_correct` + decision `2/3` -> `should_abstain`

## Current Script Surface

These helpers came out of the `a_silva` pass:

- reusable review merge: [scratch/summarize_label_reconsideration_reviews_20260420.py](scratch/summarize_label_reconsideration_reviews_20260420.py)
- reusable review-to-manual-schema conversion: [scratch/convert_label_reconsideration_to_manual_schema_20260420.py](scratch/convert_label_reconsideration_to_manual_schema_20260420.py)
- bundle diffing after rerun: [scratch/compare_classic_bundle_eval_errors_20260420.py](scratch/compare_classic_bundle_eval_errors_20260420.py)

These are `a_silva`-specific and should be copied or generalized for another slice:

- held-out packet builder: [scratch/build_a_silva_label_reconsideration_packets_20260420.py](scratch/build_a_silva_label_reconsideration_packets_20260420.py)
- gate-source packet builder: [scratch/build_a_silva_calibration_label_reconsideration_packets_20260420.py](scratch/build_a_silva_calibration_label_reconsideration_packets_20260420.py)
- original drop-only patcher: [scratch/apply_a_silva_review_drops_to_bundle_20260420.py](scratch/apply_a_silva_review_drops_to_bundle_20260420.py)
- inline correction repair for `20260428n`: [scratch/repair_bundle_n_apply_manual_corrections_20260420.py](scratch/repair_bundle_n_apply_manual_corrections_20260420.py)
- bundle rerun wrapper: [scratch/run_joint_safe_link_official_classic_bundle_20260420.py](scratch/run_joint_safe_link_official_classic_bundle_20260420.py)

## Workflow

### 1. Pick the target slice and the cleanup scope

The cleanup scope can be either:

- public held-out eval only
- public held-out eval plus reviewed gate source

For the new-block datasets (`j_smith`, `a_khan`, `a_silva`, `s_gupta`), there are usually two surfaces:

- held-out public eval in `test/<dataset>_eval_rows.csv.gz`
- reviewed gate source rows inside `calibration/classic_gate_possible_manual_w5_rows.csv.gz`

For the legacy public slices (`s2and`, `hwang_clean`, `s_park`, `s_lee`), this workflow usually only applies to the public eval slice because, in the current official stack, the reviewed gate source is only populated with the four reviewed new-block datasets (`j_smith`, `a_khan`, `a_silva`, `s_gupta`). The legacy slices still affect gate behavior indirectly through the shared thresholds, but they do not currently have their own reviewed rows inside [data/joint_safe_link_official_stack_20260428p/calibration/classic_gate_possible_manual_w5_rows.csv.gz](data/joint_safe_link_official_stack_20260428p/calibration/classic_gate_possible_manual_w5_rows.csv.gz). If we later build a reviewed gate-source slice for one of those legacy datasets, the same cleanup process can be extended to that new surface.

For `a_silva`, we chose the broader second option and reviewed the full gate source.

### 2. Build review packets for the target slice

For `a_silva` held-out eval, we built packets with:

```powershell
uv run python scratch\build_a_silva_label_reconsideration_packets_20260420.py
```

For `a_silva` gate source, we built packets with:

```powershell
uv run python scratch\build_a_silva_calibration_label_reconsideration_packets_20260420.py
```

For another slice, copy one of those builders and change at least:

- `BUNDLE_ROOT`
- `OUTPUT_ROOT`
- raw metadata paths such as `SIGNATURES_PATH` and `PAPERS_PATH`
- the predicted cluster source such as `PREDICTED_CLUSTERS_PATH`
- the dataset filter and target eval path

Each packet should contain:

- query raw metadata
- top predicted cluster
- gold cluster when a distinct positive cluster exists
- top 5 model candidates
- raw ranked rows for that query in the current window

### 3. Do the manual review

We split packets across six reviewers and asked them to review in batches of ten.

Each reviewer wrote a TSV with these columns:

- `error_case_id`
- `query_case_id`
- `window`
- `decision_code`
- `decision_text`
- `label_should_change`
- `needs_web_search`
- `confidence`
- `notes`
- `source_urls`

The review rule was:

- use web search only for decision `3`
- otherwise decide from the packet alone
- keep notes short and concrete

### 3a. How To Use Subagents Properly

For the `a_silva` pass, the manual review was distributed across six subagents. The important part was not just "use subagents," but "use them with disjoint ownership and a merge check at the end."

The subagent rules were:

- each subagent owned exactly one review TSV
- each subagent read only its own assignment markdown and case CSV
- each subagent reviewed packets in batches of ten
- each subagent appended exactly one TSV row per assigned packet
- no subagent was allowed to edit another review TSV or rewrite packet files
- no subagent used web search unless it truly needed decision `3`

The prompt shape that worked was:

- tell the subagent which TSV it owns
- tell it to read the matching assignment file
- give the exact column order it must write
- restate the four decision options verbatim
- say that it is not alone in the codebase and must only edit its assigned TSV

After all subagents finished, the merge-and-verify step was mandatory:

- run the summarizer
- inspect `missing_reviews.csv`
- if anything is missing, check for malformed keys such as a wrong `query_case_id` or `window`

This mattered in practice: during the `a_silva` calibration review, one subagent wrote a bad `query_case_id`, the summarizer surfaced it as a missing review, and the bad row had to be corrected before promotion.

### 4. Merge the review TSVs and verify coverage

Merge the TSVs back onto the packet index:

```powershell
uv run python scratch\summarize_label_reconsideration_reviews_20260420.py --output-root scratch\a_silva_label_reconsideration_20260420
uv run python scratch\summarize_label_reconsideration_reviews_20260420.py --output-root scratch\a_silva_calibration_label_reconsideration_20260420
```

Check these outputs:

- `review_summary.json`
- `all_reviews_merged.csv`
- `missing_reviews.csv`

`missing_reviews.csv` must be empty before you promote anything.

If `missing_reviews.csv` is not empty but all review TSVs look full, inspect for key mismatches such as:

- wrong `query_case_id`
- wrong `window`
- malformed TSV column order

This happened once during the `a_silva` pass and was fixed by correcting the bad TSV row.

### 5. Convert the merged reviews into a query-level review manifest

For held-out eval:

```powershell
uv run python scratch\convert_label_reconsideration_to_manual_schema_20260420.py --output-root scratch\a_silva_label_reconsideration_20260420 --dataset a_silva
```

For the broad gate-source cleanup:

```powershell
uv run python scratch\convert_label_reconsideration_to_manual_schema_20260420.py --output-root scratch\a_silva_calibration_label_reconsideration_20260420 --dataset a_silva
```

For the calibration-only variant:

```powershell
uv run python scratch\convert_label_reconsideration_to_manual_schema_20260420.py --output-root scratch\a_silva_calibration_label_reconsideration_20260420 --dataset a_silva --restrict-base-groups-csv data\joint_safe_link_official_stack_20260428p\calibration\classic_gate_possible_manual_w5_base_groups.csv
```

The conversion step writes:

- `manual_schema/batch_01/manual_assessments.tsv`
- `manual_schema/drop_query_ids.txt`
- `manual_schema/drop_base_group_ids.txt`
- `manual_schema/summary.json`

Important query-level collapse rule for mixed window decisions:

- if any reviewed window lands on decision `2` or `3`, keep the query as `possible` and apply that correction
- otherwise, if any reviewed window lands on decision `1`, keep the query as `possible` with `correction_type = none`
- only mark the query `impossible` when every reviewed window lands on decision `4`

This matters for legacy public slices like `s_lee`, where `w5` and `w25` can disagree. A prior bad `s_lee` pass incorrectly dropped mixed `1,4` and `2,4` queries at conversion time. Do not treat "any `4`" as a drop rule.

For the broad `a_silva` gate-source pass the result was:

- `497` reviewed gate-source queries
- `495` kept with either `none`, `top1_should_link`, or `should_abstain`
- `2` dropped as truly ambiguous

### 5a. Parallel Boundary and Ownership

Steps `1` through `5` are slice-local and safe to run in parallel across different datasets because they only read the source bundle and write per-slice outputs under `scratch/<slice>...`.

Steps `6` through `8` are **not** parallel-safe. They mutate the shared candidate bundle [data/joint_safe_link_official_stack_20260428p](data/joint_safe_link_official_stack_20260428p) and overwrite shared verification outputs.

If multiple cleaners are working at once:

- each cleaner may build packets, review, summarize, and convert only for their own slice
- each cleaner owns only their own `scratch/<slice>...` review roots
- a single integrator must own steps `6` through `8`
- the integrator must apply only one slice worth of bundle changes at a time

Do not run steps `6` through `8` concurrently against:

- [data/joint_safe_link_official_stack_20260428p](data/joint_safe_link_official_stack_20260428p)
- [scratch/joint_safe_link_official_classic_20260428p](scratch/joint_safe_link_official_classic_20260428p)
- [scratch/classic_bundle_error_diff_20260420_l_vs_n](scratch/classic_bundle_error_diff_20260420_l_vs_n)

Definition of done:

- a reviewer is done when `missing_reviews.csv` is empty and `manual_schema/` has been written for that slice
- a checklist item is done only after the integrator has patched the active bundle, rerun replay and diff, and refreshed `bundle.json` counts plus frozen `expected_metrics.classic` from the new replay summary

### 6. Patch the bundle

For `a_silva` we ended up rewriting the active bundle in place so that:

- held-out eval reviews applied manual corrections for decision `2/3`
- gate-source reviews applied manual corrections for decision `2/3`
- only decision `4` rows were dropped
- when the same exact `query_case_id` appeared in both held-out eval review and gate-source review, the held-out review took precedence

The command used for that inline repair was:

```powershell
uv run python scratch\repair_bundle_n_apply_manual_corrections_20260420.py
```

For another slice, copy or generalize that repair script. It is still `a_silva`-specific.

After patching, verify:

- the public eval row file lost the expected number of queries
- the gate source lost the expected number of queries
- the calibration and internal gate split lists no longer reference dropped base groups
- `bundle.json` counts were updated
- no other cleaner is simultaneously mutating the active bundle

### 7. Rerun train, calibration, and eval

Run the classic replay on the new bundle:

```powershell
uv run python scratch\run_joint_safe_link_official_classic_bundle_20260420.py --bundle-root data\joint_safe_link_official_stack_20260428p --output-dir scratch\joint_safe_link_official_classic_20260428p
```

This produces:

- `summary.json`
- `verification.json`

After the replay, refresh `bundle.json` frozen metrics from the new `summary.json` before marking the slice done:

```powershell
uv run python scripts\sync_joint_safe_link_official_bundle_metadata.py --bundle-root data\joint_safe_link_official_stack_20260428p --summary-json scratch\joint_safe_link_official_classic_20260428p\summary.json
```

- update `expected_metrics.classic`
- update the frozen `score_threshold`
- update the frozen `margin_threshold`

If this step is skipped, later replays of the active bundle will compare against stale metrics and may use stale thresholds as the tie-break reference during gate fitting.

For `a_silva`, the big metric jump after the earlier inline active-bundle repair came from relabeling many reviewed query groups plus dropping a small ambiguous remainder. Once category `2` stopped being treated as a drop, the gate thresholds returned to the original pre-repair values.

### 8. Diff the new bundle against the old bundle

After the rerun, compare query-level errors between the old and new bundles:

```powershell
uv run python scratch\compare_classic_bundle_eval_errors_20260420.py --old-bundle <previous_active_bundle_root> --new-bundle data\joint_safe_link_official_stack_20260428p --output-root scratch\classic_bundle_error_diff_previous_vs_20260428p
```

Inspect:

- `summary.csv`
- per-slice `*_query_diff.csv`

This is how we verified that:

- most public-slice movement outside the cleaned dataset came from gate-threshold flips, not model-score changes
- after the inline active-bundle repair, category `2` corrections no longer forced a threshold shift away from the pre-repair gate

## Practical Notes

- Use `uv` for every Python command.
- Keep the review packets in `scratch/` and do not promote them into stable runtime code unless the pattern becomes reusable.
- The public eval cleanup and the gate-source cleanup should be tracked separately even for the same dataset.
- If you clean the full reviewed gate source, say that explicitly in the rerun report so nobody mistakes the internal gate eval for an untouched holdout.
- When comparing bundles, look at query-level diffs, not only balanced accuracy. A threshold move can fix many positives while introducing a small number of new abstains or false positive links elsewhere.
- Treat [data/joint_safe_link_official_stack_20260428p](data/joint_safe_link_official_stack_20260428p) as the current active official bundle. If a future candidate bundle is prepared, repeat the promotion step before changing runtime defaults.
