# Giant-block EPS review

Use this procedure before final release evaluation.
Benchmark validation supplies the initial EPS.
Giant-block curves supply the final operating-point review.
The release owner chooses EPS after review of both views and benchmark validation.
The scripts do not choose EPS automatically.

## Inputs and historical procedure

Use one root that combines the `inference` and `dataset` bundles, and the complete candidate model.
The seven giant datasets are `a_khan`, `a_silva`, `h_wang`, `j_smith`, `s_gupta`, `s_lee`, and `s_park`.
Their Arrow data, profile memberships, dev labels, and initial-query source map already belong to the `dataset` bundle.
Use `--replay-bundle incremental` to select that catalog from the combined public root.
Obtain the model and index from the `inference` bundle or `artifacts/common/v1.3`.
No second giant-data copy is required.

The final `1.3` review uses fresh internal Semantic Scholar exports instead of the snapshot giants.
The NEW `a_khan`, `a_silva`, and `j_smith` blocks carry ORCID references.
The claimed blocks are `f_matsen`, `s_feldman`, `s_gupta`, `s_park`, `s_lee`, and `h_wang`.
These exports are not part of the released data, so public users cannot repeat the final review.
Future releases can also use internal exports for this step.

The original plot script is `scratch/plot_q_class_eps_sweeps.py`.
It plots positive recall against negative separation from expanded incremental candidate labels.
Its old reference EPS is `0.6064583976`; a retained topic-split command uses `0.65`.
No retained manual decision record establishes that `0.65` is the final choice.
The maintained producer and plot now reside in `scripts/eps_sweep`.

A separate July 18 study selects `0.60` with an ORCID-masked B3 objective.
Its report remains in `scratch/big7_1250_b3_operating_point_v1_20260718/report.md` where available.
Its source is recoverable from commit `a7e2db9` as `scripts/evaluate_big7_1250_b3_operating_point.py`.
That study caps each domain at 250,000 pairs and each sampled component at 256 members.
The current procedure uses the current model and production graph subblocks of the giant Arrow datasets.
Historical thresholds do not replace current release review.

## Freeze populations

Treat EPS reference labels as development data.
Freeze the giant dataset list, EPS grid, subblock size, input snapshot, and model before the sweep.
Keep final test labels out of EPS selection.
The B3 sweep reads the snapshot's frozen final-test assignments.
It resolves each initial-only query copy to its source record through the snapshot map.
A final-test record excludes its entire ORCID group from both reference views.
Excluded records remain available as inference context.
`plan.json` reports the excluded groups, their signature count, and their fraction of ORCID references.
The sweep stops if a final-test record remains in a reference.

The identity view uses repeated ORCID references after normalization.
It excludes the 14 reviewed mixed-identity groups and synthetic initial-query copies.
It also excludes each record in `orcid_error_query_signatures.json`; the rest of that ORCID group remains.
A second B3 curve, `eps_b3_with_singletons.tsv`, also keeps one-record ORCID groups.
Those records cannot lose recall, but a wrong merge lowers their precision.
That curve is diagnostic; the repeated view selects EPS.
ORCID IDs remain hidden from native features, constraints, and subblocking repair.
Unknown identities remain in inference context; B3 scores only known references.
Full curves retain cross-subblock split errors.
ORCID references do not establish complete or error-free identity gold.

The diagnostic view uses active `calibration_fit` and `calibration_check` task assignments.
It excludes test assignments, unknown singleton supervision, self-pairs, and each initial query's source record.
Conflicting pair labels remain in a separate output for review.
An abstention can mean insufficient evidence rather than different authors.
Read the negative-separation curve as a safe-action diagnostic, not universal identity precision.

## Rehearse and measure the workload

Set explicit paths outside the checkout:

```powershell
$Snapshot = '<corrected-research-snapshot>'
$Candidate = '<complete-model-directory>'
$EpsRun = '<fresh-EPS-run-directory>'
$Giants = @('a_khan', 'a_silva', 'h_wang', 'j_smith', 's_gupta', 's_lee', 's_park')
$Grid = @('0.40', '0.45', '0.50', '0.55', '0.60', '0.62', '0.64', '0.65', '0.66', '0.68', '0.70', '0.75', '0.80')

uv run --no-sync python scripts/eps_sweep/sweep_eps_on_linking_gold.py `
  --dataset j_smith --arrow-root $Snapshot --replay-bundle incremental --model-path $Candidate `
  --score-mode orcid-b3 --output-dir "$EpsRun/smoke/j_smith" `
  --max-subblocks 1 --subblock-selection smallest --compute-missing-dists `
  --eps-values $Grid --n-jobs 20 --total-ram-bytes 137438953472
```

Inspect the smoke log and `plan.json` before a larger run.
Run `--dry-run` for each full dataset to obtain subblock, signature, and pair counts.
Use fresh output paths for plans and measured runs.
Review the runtime estimate and approved resource limits before the full run.
Keep stdout, stderr, and a process ID for long jobs.
Distance caches permit recovery after an interrupted run.

## Produce both views

Run each giant sequentially within the approved resource budget:

```powershell
foreach ($Dataset in $Giants) {
  uv run --no-sync python scripts/eps_sweep/sweep_eps_on_linking_gold.py `
    --dataset $Dataset --arrow-root $Snapshot --replay-bundle incremental --model-path $Candidate `
    --score-mode orcid-b3 --output-dir "$EpsRun/b3/$Dataset" `
    --distance-cache-dir "$EpsRun/b3-cache/$Dataset" `
    --eps-values $Grid --n-jobs 20 --total-ram-bytes 137438953472 `
    --compute-missing-dists --allow-full-run
  if ($LASTEXITCODE -ne 0) { throw "B3 sweep fails for $Dataset" }
}

uv run --no-sync python scripts/eps_sweep/build_linking_eps_gold.py `
  --bundle-root $Snapshot --datasets $Giants --output-dir "$EpsRun/proxy-gold"

foreach ($Dataset in $Giants) {
  uv run --no-sync python scripts/eps_sweep/sweep_eps_on_linking_gold.py `
    --dataset $Dataset --arrow-root $Snapshot --replay-bundle incremental --model-path $Candidate `
    --gold-path "$EpsRun/proxy-gold/${Dataset}_pair_constraints.parquet" `
    --output-dir "$EpsRun/proxy/$Dataset" `
    --eps-values $Grid --n-jobs 20 --total-ram-bytes 137438953472 `
    --compute-missing-dists --allow-full-run
  if ($LASTEXITCODE -ne 0) { throw "Diagnostic sweep fails for $Dataset" }
}

uv run --no-sync python scripts/eps_sweep/plot_b3_eps_sweeps.py `
  --root "$EpsRun/b3" --datasets $Giants --output "$EpsRun/b3.png" --reference-eps 0.57
uv run --no-sync python scripts/eps_sweep/plot_q_class_eps_sweeps.py `
  --root "$EpsRun/proxy" --datasets $Giants --output "$EpsRun/proxy.png" `
  --weight weight_query_label_balanced --reference-eps 0.57
```

Set `--reference-eps` to the actual initial EPS for another release.
The diagnostic plot uses query/label weights to match the original recall/separation plot.
Each star marks the highest tested diagnostic accuracy for its source.
Stars do not select release EPS or maximize B3.
The diagnostic plot excludes pairs from separate subblocks.
Inspect their counts in `eps_boundary_pairs.tsv`.
Use `--weight weight_query_class_balanced` for the previous equal-query view.
Keep completed output directories immutable.
Use a new output directory and the same verified distance-cache directory for a refined grid.
Each cache binds the model, Arrow generation, signature order, and ORCID policy.
Each completed curve binds its plan and score bytes.
Plots reject missing datasets, incomplete grids, and mixed models.

## Review and apply EPS

Review B3 precision, recall, and F1 for every giant.
Compare the repeated curves (solid) with the singleton-inclusive curves (dashed).
Review the diagnostic plot and its subblock-boundary counts.
Check full reference coverage and the matched benchmark validation curve.
Verify booster hashes before reuse of a benchmark validation report.
Set `$ReviewedEps` to the value you select.
Record the reason for that choice.

```powershell
uv run --no-sync python scripts/eps_sweep/apply_reviewed_eps.py `
  --source-model $Candidate --output-model "$EpsRun/reviewed/production_model_v1.3" `
  --sweep-root "$EpsRun/b3" --datasets $Giants --eps $ReviewedEps `
  --review-note '<human decision and tradeoff>' --output-report "$EpsRun/eps_review.json"
```

The command rejects stale models, partial curves, hidden-label violations, and bounded smoke results.
It requires exactly the seven frozen giants and both complete curves for each.
It reads the final-test assignments again and rejects a reference that contains a final-test record.
It requires identity B3 curves; the safe-action plot remains diagnostic evidence.
It writes a fresh complete model and preserves every trained artifact byte.
Only `clusterer.json` and `manifest.json` change.
An EPS-only change requires no pairwise or incremental fit.
Repeat affected cluster evaluation and real-model inference checks with the reviewed package before acceptance.

Keep sweep outputs and distance caches local.
Keep their input roles and hashes distinct from untouched final tests.
See [the release process](release.md) for acceptance and publication.
