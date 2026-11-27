# Giant-block EPS review

Use this procedure before final release evaluation.
Through 1.4, benchmark validation supplied the initial EPS. The release owner reviewed the giant-block curves to choose the final EPS.
For 1.5, follow the [pinned EPS review](v15_training_design.md#3-review-production-eps). Use the curves below as diagnostics.

## Inputs

Use one root that combines the `inference` and `dataset` bundles, and the complete candidate model.
The 1.3 snapshot includes seven giant datasets. They are `a_khan`, `a_silva`, `h_wang`, `j_smith`, `s_gupta`, `s_lee`, and `s_park`.
Their Arrow data, profile memberships, dev labels, and initial-query source map already belong to the `dataset` bundle.
Use `--replay-bundle incremental` to select that catalog from the combined public root.
Obtain the model and index from the `inference` bundle.
No second giant-data copy is required.

## Close-pair bonus

A candidate model can carry a close-pair bonus, selected with EPS in benchmark calibration.
The sweep builds each cached tree with the candidate's bonus, so the curves review EPS for that linkage.
In a sweep review, `apply_reviewed_eps.py` keeps the bonus and applies the reviewed EPS.
To change the bonus itself, rerun the sweep on a candidate that carries the new value.
Without snapshot curves, `--close-pair BONUS DISTANCE` sets a reviewed bonus, and the review note cites the evidence.
`--release-version` names a model that differs from its source only in these linkage settings.

With the bonus, a score that weights precision above recall has no peak on giant blocks: recall falls slowly as EPS falls, so the score keeps rising.
The precision-weighted review score, (2 × precision + recall) / 3, therefore cannot select EPS for a bundle that uses the bonus.
Use the model plan's `precision_f1_tolerance` instead, in benchmark calibration and in this review.

Pass the plan's tolerance to `apply_reviewed_eps.py` as `--precision-f1-tolerance`.
The command then rejects an EPS whose pooled giant-block F1 is more than the tolerance below the best swept F1.
Benchmark calibration picks the most precise EPS within the tolerance on validation; this check keeps that choice within the same tolerance on giant blocks.
If it rejects the EPS, review the next higher value on the grid.
Sweep enough EPS values around the candidate for the best swept F1 to be meaningful.

## Freeze populations

Treat EPS reference labels as development data.
Freeze the giant dataset list, EPS grid, subblock size, input snapshot, and model before the sweep.
Keep final test labels out of EPS selection.
The B-cubed (B3) sweep reads the snapshot's frozen final-test assignments.
It resolves each initial-only query copy to its source record through the snapshot map.
A final-test record excludes its entire ORCID group from both reference views.
Excluded records remain available as inference context.
`plan.json` reports the excluded groups, their signature count, and their fraction of ORCID references.
The sweep stops if a final-test record remains in a reference.

The identity view uses repeated ORCID references after normalization.
It excludes the 19 reviewed mixed-identity groups and synthetic initial-query copies.
It also excludes each record in `orcid_error_query_signatures.json`; the rest of that ORCID group remains.
A second B3 curve, `eps_b3_with_singletons.tsv`, also keeps one-record ORCID groups.
Those records cannot lose recall, but a wrong merge lowers their precision.
The sweep review uses the repeated view and treats the singleton-inclusive curve as a diagnostic.
ORCID IDs remain hidden from native features, constraints, and subblocking repair.
Unknown identities remain in inference context; B3 scores only known references.

The default `--route production` scores what `predict_from_arrow` returns at each EPS.
It clusters the full-name subblocks from cached distances with the production block-clustering code, including the block corrections.
The ORCID-B3 sweep hides ORCIDs from those corrections and the final subblock join.
`plan.json` records this as `clustering_source`; the release review rejects a sweep without it.
Production code then reunites no-middle records that the size cap split, and attaches initial-only groups through the linker.
Those steps run on `orcid_free_arrow/` in the output directory, a dataset copy without ORCIDs.
`eps_b3_bulk_only.tsv` keeps the curve before the linker steps as a diagnostic.
`--route bulk` scores only bulk clustering of the target subblocks and keeps cross-subblock split errors.
Use it for `--max-subblocks` smoke runs; it does not select EPS.
ORCID references do not establish complete or error-free identity gold.

See [evaluation traps](negative_results.md#evaluation-traps) for pitfalls with proxy labels.

## Rehearse and measure the workload

Set explicit paths outside the checkout:

```powershell
$Snapshot = '<combined-inference-and-dataset-root>'
$Candidate = '<complete-model-directory>'
$EpsRun = '<fresh-EPS-run-directory>'
$BatchingThreshold = 15000
$Giants = @('a_khan', 'h_wang', 'j_smith', 's_gupta', 's_lee', 's_park')
$Grid = @('0.40', '0.45', '0.50', '0.55', '0.60', '0.62', '0.64', '0.65', '0.66', '0.68', '0.70', '0.75', '0.80')

uv run --no-sync python scripts/eps/sweep_eps_on_linking_gold.py `
  --dataset j_smith --arrow-root $Snapshot --replay-bundle incremental --model-path $Candidate `
  --score-mode orcid-b3 --output-dir "$EpsRun/smoke/j_smith" --route bulk `
  --batching-threshold $BatchingThreshold `
  --max-subblocks 1 --subblock-selection smallest --compute-missing-dists `
  --eps-values $Grid --n-jobs 12 --total-ram-bytes 137438953472
```

The examples use a production subblock limit of 15,000, 12 threads, and 128 GiB of RAM.
Set the limit to the one used in production and the threads and RAM to your allocation.
Set `$Giants` to the datasets with linker calibration assignments outside `s2and_eval` in the snapshot's split file.
The example lists the 1.4 development giants.
Every sweep requires the limit and records it in `plan.json`.
Inspect the smoke log and `plan.json` before a larger run.
Run `--dry-run` for each full dataset to obtain subblock, signature, and pair counts.
Use fresh output paths for plans and measured runs.
Review the runtime estimate and approved resource limits before the full run.
Keep stdout, stderr, and a process ID for long jobs.
Distance caches permit recovery after an interrupted run.
Reuse requires matching package, distance policy, and pairwise bundle feature contract. Use `--overwrite-dists --compute-missing-dists` to rebuild incompatible caches.
The production route caches every subblock of each target block and runs the linker steps once per EPS value.
Time one EPS value on the largest giant before choosing the grid.

## Run the B3 sweep

Run each giant sequentially within the approved resource budget:

```powershell
foreach ($Dataset in $Giants) {
  uv run --no-sync python scripts/eps/sweep_eps_on_linking_gold.py `
    --dataset $Dataset --arrow-root $Snapshot --replay-bundle incremental --model-path $Candidate `
    --score-mode orcid-b3 --output-dir "$EpsRun/b3/$Dataset" `
    --distance-cache-dir "$EpsRun/b3-cache/$Dataset" `
    --batching-threshold $BatchingThreshold `
    --eps-values $Grid --n-jobs 12 --total-ram-bytes 137438953472 `
    --compute-missing-dists --allow-full-run
  if ($LASTEXITCODE -ne 0) { throw "B3 sweep fails for $Dataset" }
}

uv run --no-sync python scripts/eps/plot_b3_eps_sweeps.py `
  --root "$EpsRun/b3" --datasets $Giants --output "$EpsRun/b3.png" --reference-eps 0.57
```

Set `--reference-eps` to the actual initial EPS for another release.
Keep completed output directories immutable.
Use a new output directory and the same verified distance-cache directory for a refined grid.
Each cache binds the model, Arrow generation, signature order, and ORCID policy.
Each completed curve binds its plan and score bytes.
Plots reject missing datasets, incomplete grids, and mixed models.

## Review and apply EPS

Review B3 precision, recall, and F1 for every giant.
Compare the repeated curves (solid) with the singleton-inclusive curves (dashed).
Check full reference coverage and the matched benchmark validation curve.
Verify booster hashes before reuse of a benchmark validation report.
Set `$ReviewedEps` to the value you select.
Record the reason for that choice.

```powershell
$PrecisionF1Tolerance = (Get-Content -Raw "$Snapshot/pairwise/model_plan.json" | ConvertFrom-Json).eps.precision_f1_tolerance
if ($null -eq $PrecisionF1Tolerance) { $PrecisionF1Tolerance = 0 }
uv run --no-sync python scripts/eps/apply_reviewed_eps.py `
  --source-model $Candidate --output-model "$EpsRun/reviewed/production_model_v1.3.1" `
  --sweep-root "$EpsRun/b3" --datasets $Giants --eps $ReviewedEps `
  --expected-batching-threshold $BatchingThreshold --precision-f1-tolerance $PrecisionF1Tolerance `
  --review-note '<human decision and tradeoff>' --output-report "$EpsRun/eps_review.json"
```

The command rejects stale models, partial curves, hidden-label violations, and bounded smoke results.
It requires each plan's subblock limit to match `--expected-batching-threshold`.
The command resolves the snapshot's linker calibration assignments outside `s2and_eval`
against their declared sources. It rejects an assignment whose dataset differs from its source rows.
The `--datasets` list must match the resulting datasets. The command reports and ignores other sweep folders.
The plans must all point to the same snapshot. Apply and the plotters check each plan's recorded
dataset name. Apply requires both completed curves for every requested dataset.
It reads the final-test assignments again and rejects a reference that contains a final-test record.
It requires identity B3 curves.
It writes a fresh complete model and preserves every trained artifact byte.

When internal evidence supplies the review, omit `--sweep-root` and the dataset and threshold options.
Cite the evidence in the review note.
It still checks the EPS range, the fresh output paths, the model load, and the unchanged artifacts.
Release `1.3` used this form:

```powershell
uv run --no-sync python scripts/eps/apply_reviewed_eps.py `
  --source-model $Candidate --output-model "$EpsRun/reviewed/production_model_v1.3" `
  --eps 0.53 --review-note '<decision, tradeoff, and internal evidence location>' `
  --output-report "$EpsRun/eps_review.json"
```
An EPS-only review updates `clusterer.json` and the manifest. To stamp reviewed linker cutoffs in the
same step, add `--gate-cutoffs T_FULL T_INIT`. Both reviews preserve the fitted boosters.
Repeat affected cluster evaluation and real-model inference checks with the reviewed package before acceptance.

Keep sweep outputs and distance caches local.
Keep their input roles and hashes distinct from untouched final tests.
See [the release process](release.md) for acceptance and publication.

## Recorded release decisions

| Release | Linkage | Reviewed EPS | Key metrics | Review population and caveats |
| --- | --- | --- | --- | --- |
| `1.3` (2026-10-19) | Plain average linkage | `0.53` | Signature-weighted (2 × B3 precision + B3 recall) / 3 peaked at 90.09%, versus 88.99% at `0.57`. Benchmark outer-test B3 F1 rose from 0.963514 to 0.964372. | Nine internal exports outside the public snapshot. New `a_khan`, `a_silva`, and `j_smith` had ORCID references. Claimed blocks were `f_matsen`, `s_feldman`, `s_gupta`, `s_park`, `s_lee`, and `h_wang`. The same references selected and assessed EPS. |
| `1.3.1` | Same boosters and linker as `1.3`. Close-pair bonus `8` below `0.05`. | `0.40` | Pooled ORCID-B3 F1 peaked at 0.9317 over ten EPS values from `0.30` to `0.53`. Signature-weighted benchmark outer-test precision rose from 0.973500 to 0.978019, recall from 0.956478 to 0.959062, and F1 from 0.964702 to 0.968204. | Validation selected EPS with tolerance `0.001`, and the seven giants passed the same tolerance check. Validation F1 varied by less than 0.001 from `0.40` to `0.53`. Outer-test curves were inspected during development. |
| `1.4` (2026-11-23) | Retrained boosters (fixed-parameter refit) and linker. Close-pair bonus `8` below `0.05`. | `0.38` | Production route at the 15,000 limit on six giants: pooled ORCID-B3 F1 0.920700 at `0.38`, 0.000454 below the best (`0.40`). On all 152,819 gold-ORCID signatures of those giants, `0.38` was best (0.932566). Benchmark overlap F1 0.967360. | Rule: the most precise EPS within `0.001` of the best pooled giant F1. The development reference dropped every author with a final-test linker query, which removed 552 of 582 authors with ≥50 records; the all-author check confirmed `0.38`. On 27 fresh held-out production blocks (64,770 reference signatures), `1.4` at `0.38` scored 0.94790 vs `1.3.2` at `0.40` 0.94874 (difference −0.084 pts, 95% CI [−0.413, +0.243]); `0.38` led on Batches 1–3, while the cumulative 27-block curve peaked at `0.46` (0.94909, 0.00119 above `0.38`, exceeding the 0.001 tolerance used on the selection set). |
