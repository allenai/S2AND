# Reproduce the 1.3.1 model from public data

Run the production training process on the published `1.3` data to approximate the released `1.3.1` model.
`1.3.1` uses the `1.3` boosters and linker with a close-pair bonus of `8` for pairs under `0.05` and a clustering distance threshold (EPS) of `0.40`.

- **Name counts.** Use the released `1.3.1` index. Rebuilding it requires internal data.
- **Final EPS review.** Internal production exports support this check, which kept the validation choice of `0.40`. See [giant-block EPS review](eps_calibration.md).
- **LightGBM fits.** Thread scheduling can change the last digits of some leaf values. Compare rebuilt models using the reference metrics.

## Set up

In a source checkout, run `git checkout <release-tag>` and set up the environment with the [install guide](install.md#repo-checkout).

> **Update at v1.3 release:** Replace `<release-tag>` with the verified release tag.

Commands use the script folders of this checkout. Release `1.4` and earlier kept scripts under
`scripts/production/`, `scripts/eps_sweep/` and `scripts/verification/`; see [moved files](../scripts/README.md#moved-files).

## Sync the 1.3.1 index and reference model

```bash
aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and/1.3.1/inference/ s2and/data/
```

`s2and/data/` now holds `production_model_v1.3.1/` and the `1.3.1` `name_counts_index/` used to fit it.
For production inference, use the current `1.3.2` inference bundle from the [data download guide](data.md#download).

## Train and evaluate: a second sync

```bash
aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and/1.3.1/dataset/ s2and/data/
ROOT=$PWD/s2and/data
RUN=$PWD/runs/1.3.1
mkdir -p "$RUN/matrix-work"
```

Run the steps in order.
Each step writes a fresh output and refuses to overwrite an existing one.
The examples use 12 threads; set `--n-jobs` and `--total-ram-bytes` to your machine.

### 1. Pairwise hyperparameter search and fit

```bash
uv run --no-sync python scripts/pairwise/train_pairwise.py \
  --model-plan "$ROOT/pairwise/model_plan.json" \
  --name-counts-index-root "$ROOT/name_counts_index" \
  --incumbent-parameters scripts/pairwise/pairwise_incumbent_h13.json \
  --output-dir "$RUN/pairwise/production_model_v1.3.1" \
  --matrix-work-dir "$RUN/matrix-work" \
  --n-jobs 12 \
  --run-full
```

The trainer selects pairs from the development blocks and computes features from the same Arrow data.
Each head runs a 50-trial tree-structured Parzen estimator (TPE) search against validation ROC AUC.
The H13 configuration competes with each search winner, as it did for `1.3`.
The output is a pairwise-only model with pending EPS.

### 2. EPS selection on validation blocks

```bash
uv run --no-sync python scripts/pairwise/release_pairwise.py calibrate-eps \
  --source-bundle "$RUN/pairwise/production_model_v1.3.1" \
  --model-plan "$ROOT/pairwise/model_plan.json" \
  --output-bundle "$RUN/calibrated/production_model_v1.3.1" \
  --output-report "$RUN/eps_calibration_report.json" \
  --name-counts-index-root "$ROOT/name_counts_index" \
  --n-jobs 12
```

Calibration clusters each source's validation blocks at every EPS in the plan's grid, once with plain
average linkage and once with the close-pair bonus.
It selects the setting with the best signature-weighted B-cubed (B3) F1.
When the plan sets `precision_f1_tolerance`, it selects the most precise setting within that tolerance of the best F1.
The `1.3.1` plan sets it to `0.001`.
If a close-pair bonus wins with a zero or omitted tolerance, calibration saves the trials beside
the report as `<output-report>.pending.json`. Rerun with the same inputs and
`--allow-zero-precision-f1-tolerance` to approve the selection and reuse those trials.
Use a fresh report path if you change the plan.
Test blocks never select EPS.

### 3. Incremental linker training and the complete model

```bash
uv run --no-sync python scripts/linker/train_linker_and_finalize.py \
  --source-bundle-root "$ROOT" \
  --target-json tests/fixtures/incremental_linker_training_target.json \
  --pairwise-model-path "$RUN/calibrated/production_model_v1.3.1" \
  --name-counts-index-root "$ROOT/name_counts_index" \
  --output-dir "$RUN/complete" \
  --n-jobs 12
```

The step computes incremental features with the calibrated pairwise model and fits the linker.
It writes the complete model, reloads it, and evaluates the declared linker test split.
Preflight resolves the frozen calibration and test assignments before featurization.
Calibration groups can use the calibration source, and each assigned evaluation table must still contribute eligible calibration rows.
Test groups must have eligible rows in their assigned evaluation tables.
The linker fit, `evaluate_classic`, and the EPS tools reject an assignment whose source key
has no declared evaluation table.
Rows marked `unlabeled_singleton_orcid` are excluded from both splits.
Preflight checks each dataset in each label table. When ten or more views are marked
`negative_singleton_orcid`, at least 10% of those views must have a positive candidate
after the linker filters the labels. Mark unreviewed views `unlabeled_singleton_orcid`
and remove split assignments for excluded queries.
Source bundle assembly and corrected dataset packaging validate labels before copying.

### 4. Outer-test evaluation

```bash
uv run --no-sync python scripts/pairwise/release_pairwise.py evaluate-heldout \
  --model-plan "$ROOT/pairwise/model_plan.json" \
  --model "$RUN/complete/production_model_v1.3.1" \
  --name-counts-index-root "$ROOT/name_counts_index" \
  --output-report "$RUN/heldout_metrics.json" \
  --n-jobs 12
```

The report contains pairwise metrics on the frozen held-out pairs and B3 on the held-out blocks.

### 5. Identity-gold linker cases

```bash
uv run --no-sync python scripts/release/verification/evaluate_incremental_gold.py evaluate \
  --snapshot "$ROOT" \
  --cases "$ROOT/evaluation_inputs/gold_identity_cases_seed13.json" \
  --model-dir "$RUN/complete/production_model_v1.3.1" \
  --output-dir "$RUN/gold" \
  --n-jobs 12 \
  --total-ram-bytes 34359738368
```

Pass `"$ROOT/production_model_v1.3.1"` as the model in steps 4 and 5 to score the released model.

## Reference results for 1.3.1

The released boosters and linker were fit before the 2026-10-22 distribution correction removed papers absent from the production Semantic Scholar Academic Graph (S2AG) and their dependent records.
The middle column reports results on the original data. The last column scores the released model on the smaller published data. Training on the published data yields counts and scores that can differ from the original fit.
The `dataset` bundle's `pairwise/population_corrections.json` pairs the original and published populations for each benchmark so step 4 can score the released model on the published held-out blocks. Step 5 scores the published identity-gold cases. See [the data guide](data.md).

The `1.3` report used an earlier scorer that rounded each dataset's score to three decimal places.
The current scorer retains full precision, prediction applies name-conflict repair, and calibration also scores the close-pair bonus.

| Measurement | Released `1.3.1` value | Released model on the published data |
| --- | --- | --- |
| Pairwise training and validation pairs | 804,716 and 60,064 | n/a |
| Main head validation ROC AUC | 0.994227 (search trial 26) | n/a |
| Nameless head validation ROC AUC | 0.974620 (H13) | n/a |
| Validation selection | EPS 0.40 with close-pair bonus 8 under 0.05: B3 precision 0.971954, recall 0.951028, F1 0.961184 on 72,649 signatures. The best F1 on the grid is 0.962106, at EPS 0.49 with the bonus | n/a |
| Outer-test signature-weighted B3 | Precision 0.978019, recall 0.959062, F1 0.968204 on 72,853 signatures | Precision 0.977285, recall 0.962248, F1 0.969488 on 68,289 signatures |
| The same test for the released `1.3` model | F1 0.964702 (0.964476 with name-conflict repair off). The `1.3` report gives 0.964372 | F1 0.967302 with the `1.3` dataset bundle |
| Outer-test pairwise AUROC | 0.989391 | 0.989664 |
| Linker test accuracy | 96.715% on 10,837 queries, from the `1.3` fit; `1.3.1` keeps that linker | n/a |
| Identity-gold cases | 134 of 152 correct, the same as `1.3` | 125 of 141 correct |

Steps 1-5 end at the released settings.
Calibration selects EPS `0.40` with the close-pair bonus, and the giant-block review kept that choice.

A reviewed EPS can differ from the calibrated one, as it did for `1.3`.
`apply_reviewed_eps.py` then writes the reviewed EPS into a fresh complete model:

```bash
uv run --no-sync python scripts/eps/apply_reviewed_eps.py \
  --source-model "$RUN/complete/production_model_v1.3.1" \
  --output-model "$RUN/released-eps/production_model_v1.3.1" \
  --eps "<reviewed-eps>" \
  --review-note "<the review and its evidence>" \
  --output-report "$RUN/released_eps.json"
```

The command changes only `clusterer.json` and the manifest; the boosters stay the same.

On 12 threads of a 12-core workstation, step 1 takes about 70 minutes and step 3 about 9 minutes.
Steps 2, 4, and 5 take about a minute each or less.
Step 1 is LightGBM's hyperparameter search; 20 threads did not finish it sooner than 12.
