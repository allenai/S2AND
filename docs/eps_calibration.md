# Giant-block EPS review

Use this procedure before final release evaluation.
Benchmark validation supplies the initial EPS.
Giant-block curves supply the final operating-point review.
The release owner chooses EPS after review of both views and benchmark validation.
The scripts do not choose EPS automatically.

## Inputs and historical procedure

Use the accepted corrected research snapshot and the complete candidate model.
The seven giant datasets are `a_khan`, `a_silva`, `h_wang`, `j_smith`, `s_gupta`, `s_lee`, and `s_park`.
Their Arrow data, profile memberships, dev labels, and initial-query source map already belong to the research snapshot.
Use `--replay-bundle incremental` to select that catalog from the combined public root.
Publish those inputs once in S3 with the snapshot.
Obtain the model and common count index from the declared Git LFS ref.
No second giant-data copy is required.

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
If final tests share giant identities, pass `--reference-exclusions` with their signature IDs, one per line.
A listed signature excludes its entire ORCID group from the reference score.
Its records remain available as inference context.

The identity view uses repeated ORCID references after normalization.
It excludes the 14 reviewed mixed-identity groups and synthetic initial-query copies.
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
  --root "$EpsRun/b3" --datasets $Giants --output "$EpsRun/b3.png" --reference-eps 0.55
uv run --no-sync python scripts/eps_sweep/plot_q_class_eps_sweeps.py `
  --root "$EpsRun/proxy" --datasets $Giants --output "$EpsRun/proxy.png" --reference-eps 0.55
```

Set `--reference-eps` to the actual initial EPS for another release.
Keep completed output directories immutable.
Use a new output directory and the same verified distance-cache directory for a refined grid.
Each cache binds the model, Arrow generation, signature order, and ORCID policy.
Each completed curve binds its plan and score bytes.
Plots reject missing datasets, incomplete grids, and mixed models.

## Review and apply EPS

Review B3 precision, recall, and F1 for every giant.
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
It requires identity B3 curves; the safe-action plot remains diagnostic evidence.
It writes a fresh complete model and preserves every trained artifact byte.
Only `clusterer.json` and `manifest.json` change.
An EPS-only change requires no pairwise or incremental fit.
Repeat affected cluster evaluation and real-model inference checks with the reviewed package before acceptance.

Retain the reference files, curves, plots, and manual decision with release evidence.
Publish optional calibration sidecars with the snapshot under its immutable S3 research prefix.
Keep distance caches local.
Keep their input roles and hashes distinct from untouched final tests.
See [the release process](release.md) for acceptance and publication.
