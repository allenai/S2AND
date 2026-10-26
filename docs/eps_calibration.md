# Giant-block EPS review

Use this procedure before final release evaluation.
Benchmark validation supplies the initial EPS.
Giant-block curves supply the final operating-point review.
The release owner chooses EPS after review of both views and benchmark validation.
The scripts do not choose EPS automatically.

## Inputs and the 1.3 decision

Use one root that combines the `inference` and `dataset` bundles, and the complete candidate model.
The seven giant datasets are `a_khan`, `a_silva`, `h_wang`, `j_smith`, `s_gupta`, `s_lee`, and `s_park`.
Their Arrow data, profile memberships, dev labels, and initial-query source map already belong to the `dataset` bundle.
Use `--replay-bundle incremental` to select that catalog from the combined public root.
Obtain the model and index from the `inference` bundle.
No second giant-data copy is required.

The final `1.3` review uses fresh internal Semantic Scholar exports instead of the snapshot giants.
The NEW `a_khan`, `a_silva`, and `j_smith` blocks carry ORCID references.
The claimed blocks are `f_matsen`, `s_feldman`, `s_gupta`, `s_park`, `s_lee`, and `h_wang`.
These exports are not part of the released data, so public users cannot repeat the final review.
Future releases can also use internal exports for this step.

The release owner chose EPS `0.53` for `1.3` on 2026-10-19.
The review scores each tested EPS on nine blocks with S = (2 × B3 precision + B3 recall) / 3.
It weights each scored signature equally.
EPS `0.53` gives the best S of the tested values: 90.09%, against 88.99% at `0.57`.
The same ORCID references select and assess EPS, so this review is not an independent test.
On the benchmark outer test, signature-weighted B3 F1 rises from 0.963514 at `0.57` to 0.964372 at `0.53`.

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

The default `--route production` scores what `predict_from_arrow` returns at each EPS.
It cuts cached distance trees for the full-name subblocks, as production's bulk step does.
Production code then reunites no-middle records that the size cap split, and attaches initial-only groups through the linker.
Those steps run on `orcid_free_arrow/` in the output directory, a dataset copy without ORCIDs.
`eps_b3_bulk_only.tsv` keeps the curve before the linker steps as a diagnostic.
`--route bulk` scores only bulk clustering of the target subblocks and keeps cross-subblock split errors.
Use it for `--max-subblocks` smoke runs; it does not select EPS.
ORCID references do not establish complete or error-free identity gold.

The diagnostic view uses active `calibration_fit` and `calibration_check` task assignments.
It excludes test assignments, unknown singleton supervision, self-pairs, and each initial query's source record.
Conflicting pair labels remain in a separate output for review.
An abstention can mean insufficient evidence rather than different authors.
Read the negative-separation curve as a safe-action diagnostic, not universal identity precision.

## Rehearse and measure the workload

Set explicit paths outside the checkout:

```powershell
$Snapshot = '<combined-inference-and-dataset-root>'
$Candidate = '<complete-model-directory>'
$EpsRun = '<fresh-EPS-run-directory>'
$Giants = @('a_khan', 'a_silva', 'h_wang', 'j_smith', 's_gupta', 's_lee', 's_park')
$Grid = @('0.40', '0.45', '0.50', '0.55', '0.60', '0.62', '0.64', '0.65', '0.66', '0.68', '0.70', '0.75', '0.80')

uv run --no-sync python scripts/eps_sweep/sweep_eps_on_linking_gold.py `
  --dataset j_smith --arrow-root $Snapshot --replay-bundle incremental --model-path $Candidate `
  --score-mode orcid-b3 --output-dir "$EpsRun/smoke/j_smith" --route bulk `
  --max-subblocks 1 --subblock-selection smallest --compute-missing-dists `
  --eps-values $Grid --n-jobs 12 --total-ram-bytes 137438953472
```

The examples use 12 threads and 128 GiB of RAM; set both to your allocation.
Inspect the smoke log and `plan.json` before a larger run.
Run `--dry-run` for each full dataset to obtain subblock, signature, and pair counts.
Use fresh output paths for plans and measured runs.
Review the runtime estimate and approved resource limits before the full run.
Keep stdout, stderr, and a process ID for long jobs.
Distance caches permit recovery after an interrupted run.
The production route caches every subblock of each target block and runs the linker steps once per EPS value.
Time one EPS value on the largest giant before choosing the grid.

## Produce both views

Run each giant sequentially within the approved resource budget:

```powershell
foreach ($Dataset in $Giants) {
  uv run --no-sync python scripts/eps_sweep/sweep_eps_on_linking_gold.py `
    --dataset $Dataset --arrow-root $Snapshot --replay-bundle incremental --model-path $Candidate `
    --score-mode orcid-b3 --output-dir "$EpsRun/b3/$Dataset" `
    --distance-cache-dir "$EpsRun/b3-cache/$Dataset" `
    --eps-values $Grid --n-jobs 12 --total-ram-bytes 137438953472 `
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
    --eps-values $Grid --n-jobs 12 --total-ram-bytes 137438953472 `
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

When internal exports supply the review, omit `--sweep-root` and `--datasets`.
The command cannot check those curves, so the review note must cite their location.
It still checks the EPS range, the fresh output paths, the model load, and the unchanged artifacts.
Release `1.3` used this form:

```powershell
uv run --no-sync python scripts/eps_sweep/apply_reviewed_eps.py `
  --source-model $Candidate --output-model "$EpsRun/reviewed/production_model_v1.3" `
  --eps 0.53 --review-note '<decision, tradeoff, and internal evidence location>' `
  --output-report "$EpsRun/eps_review.json"
```
Only `clusterer.json` and `manifest.json` change.
An EPS-only change requires no pairwise or incremental fit.
Repeat affected cluster evaluation and real-model inference checks with the reviewed package before acceptance.

Keep sweep outputs and distance caches local.
Keep their input roles and hashes distinct from untouched final tests.
See [the release process](release.md) for acceptance and publication.
