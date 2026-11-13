# Reproduce the 1.3.1 model from public data

This guide runs the production training process on the published `1.3` data.
The result is approximately the released `1.3.1` model.
`1.3.1` is the `1.3` fit with new linkage settings: close-pair bonus `8` for pairs under `0.05`, and EPS `0.40`.
Its boosters and linker are the `1.3` files, so steps 1 and 3 reproduce the `1.3` fit.
The current release, `1.3.2`, is this same model bound to a rebuilt name-count index; nothing in it is refitted.
This guide uses the `1.3.1` index, the one the model was fitted with.
Three steps stay outside public reproduction:

- **Name counts.** Only AI2 staff can rebuild the name-count index from internal data. Use the released index.
- **Final EPS review.** The release owner checks the calibrated EPS against giant-block EPS curves on internal production exports. For `1.3.1` that check kept the validation choice, `0.40`; see [giant-block EPS review](eps_calibration.md).
- **LightGBM fits.** Multithreaded LightGBM sums leaf statistics in an order that depends on thread scheduling.
  A rebuilt pairwise booster can differ from the released file in the last digits of some leaf values, even
  between two runs on one machine with the same thread count. Splits, the selected hyperparameters, and every
  metric in the reference table stayed the same in five reruns at 12 and 20 threads.
  Compare a rebuilt model by those metrics, not by file hashes.
  Files written on Windows also end in CRLF line breaks, so their bytes differ from files written elsewhere.

The published data is the `1.3` data after its distribution correction of 2026-10-22.
The correction removed papers absent from production S2AG and their dependent records; see [data](data.md).
The released boosters and linker were fit before it.
A reproduction therefore trains on slightly smaller populations.
Its counts and scores are close to the reference results below, not equal to them.
The released model records its original training population.
The dataset bundle's `pairwise/population_corrections.json` pairs that population with the published one for each benchmark, so step 4 scores the released model on the published held-out blocks.

The `1.3` reports also came from an earlier scorer.
Scores are no longer rounded per dataset, prediction applies name-conflict repair, and calibration also scores the close-pair bonus.

## Set up

Check out the release tag and install the environment:

```bash
git clone https://github.com/allenai/S2AND.git
cd S2AND
git checkout <release-tag>
uv sync
```

> **Update at v1.3 release:** Replace `<release-tag>` with the verified release tag.

The training scripts live in this repository, not in the Python wheel.
The [install guide](install.md) lists platform requirements.

## Run inference: one sync

```bash
aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and/1.3.1/inference/ s2and/data/
```

`s2and/data/`, the default data folder, now holds `production_model_v1.3.1/` and `name_counts_index/`.
[Load a production model](production_inference.md#load-a-production-model) for bulk and incremental prediction.

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
uv run --no-sync python scripts/production/model/train_pairwise.py \
  --model-plan "$ROOT/pairwise/model_plan.json" \
  --name-counts-index-root "$ROOT/name_counts_index" \
  --incumbent-parameters scripts/production/model/pairwise_incumbent_h13.json \
  --output-dir "$RUN/pairwise/production_model_v1.3.1" \
  --matrix-work-dir "$RUN/matrix-work" \
  --n-jobs 12 \
  --run-full
```

The trainer selects pairs from the development blocks and computes features from the same Arrow data.
Each head runs a 50-trial TPE search against validation ROC AUC.
The H13 configuration competes with each search winner, as it did for `1.3`.
The output is a pairwise-only model with pending EPS.

### 2. EPS selection on validation B3

```bash
uv run --no-sync python scripts/production/model/release_pairwise.py calibrate-eps \
  --source-bundle "$RUN/pairwise/production_model_v1.3.1" \
  --model-plan "$ROOT/pairwise/model_plan.json" \
  --output-bundle "$RUN/calibrated/production_model_v1.3.1" \
  --output-report "$RUN/eps_calibration_report.json" \
  --name-counts-index-root "$ROOT/name_counts_index" \
  --n-jobs 12
```

Calibration clusters each source's validation blocks at every EPS in the plan's grid, once with plain
average linkage and once with the close-pair bonus.
It selects the setting with the best signature-weighted B3 F1.
When the plan sets `precision_f1_tolerance`, it selects the most precise setting within that tolerance of the best F1.
The `1.3.1` plan sets it to `0.001`.
Test blocks never select EPS.

### 3. Incremental linker training and the complete model

```bash
uv run --no-sync python scripts/production/model/train_linker_and_finalize.py \
  --source-bundle-root "$ROOT" \
  --target-json tests/fixtures/incremental_linker_training_target.json \
  --pairwise-model-path "$RUN/calibrated/production_model_v1.3.1" \
  --name-counts-index-root "$ROOT/name_counts_index" \
  --output-dir "$RUN/complete" \
  --n-jobs 12
```

The step computes incremental features with the calibrated pairwise model and fits the linker.
It writes the complete model, reloads it, and evaluates the declared linker test split.

### 4. Outer-test evaluation

```bash
uv run --no-sync python scripts/production/model/release_pairwise.py evaluate-heldout \
  --model-plan "$ROOT/pairwise/model_plan.json" \
  --model "$RUN/complete/production_model_v1.3.1" \
  --name-counts-index-root "$ROOT/name_counts_index" \
  --output-report "$RUN/heldout_metrics.json" \
  --n-jobs 12
```

The report contains pairwise metrics on the frozen held-out pairs and B3 on the held-out blocks.

### 5. Identity-gold linker cases

```bash
uv run --no-sync python scripts/verification/evaluate_incremental_gold.py evaluate \
  --snapshot "$ROOT" \
  --cases "$ROOT/evaluation_inputs/gold_identity_cases_seed13.json" \
  --model-dir "$RUN/complete/production_model_v1.3.1" \
  --output-dir "$RUN/gold" \
  --n-jobs 12 \
  --total-ram-bytes 34359738368
```

Pass `"$ROOT/production_model_v1.3.1"` as the model in steps 4 and 5 to score the released model.

## Reference results for 1.3.1

The middle column comes from the `1.3` data before its distribution correction, which AI2 holds.
The last column is the released model on the published data, where steps 4 and 5 can score it.

| Measurement | Released `1.3.1` value | Released model on the published data |
| --- | --- | --- |
| Pairwise training and validation pairs | 804,716 and 60,064 | n/a |
| Main head validation ROC AUC | 0.994227 (search trial 26) | n/a |
| Nameless head validation ROC AUC | 0.974620 (H13) | n/a |
| Validation selection | EPS 0.40 with close-pair bonus 8 under 0.05: B3 precision 0.971954, recall 0.951028, F1 0.961184 on 72,649 signatures. The best F1 on the grid is 0.962106, at EPS 0.49 with the bonus | n/a |
| Outer-test signature-weighted B3 | Precision 0.978019, recall 0.959062, F1 0.968204 on 72,853 signatures | Precision 0.977285, recall 0.962248, F1 0.969488 on 68,289 signatures |
| The same test for the released `1.3` model | F1 0.964702; 0.964476 with name-conflict repair off. The `1.3` report says 0.964372: its scorer rounded each dataset's score to three decimals | F1 0.967302, with the `1.3` dataset bundle |
| Outer-test pairwise AUROC | 0.989391 | 0.989664 |
| Linker test accuracy | 96.715% on 10,837 queries, from the `1.3` fit; `1.3.1` keeps that linker | n/a |
| Identity-gold cases | 134 of 152 correct, the same as `1.3` | 125 of 141 correct |

Steps 1-5 end at the released settings.
Calibration selects EPS `0.40` with the close-pair bonus, and the giant-block review kept that choice.

A reviewed EPS can differ from the calibrated one, as it did for `1.3`.
`apply_reviewed_eps.py` then writes the reviewed EPS into a fresh complete model:

```bash
uv run --no-sync python scripts/eps_sweep/apply_reviewed_eps.py   --source-model "$RUN/complete/production_model_v1.3.1"   --output-model "$RUN/released-eps/production_model_v1.3.1"   --eps <reviewed-eps>   --review-note "<the review and its evidence>"   --output-report "$RUN/released_eps.json"
```

The command changes only `clusterer.json` and the manifest; the boosters stay the same.

On 12 threads of a 12-core workstation, step 1 takes about 70 minutes and step 3 about 9 minutes.
Steps 2, 4, and 5 take about a minute each or less.
Step 1 is LightGBM's hyperparameter search; 20 threads did not finish it sooner than 12.
