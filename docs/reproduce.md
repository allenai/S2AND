# Reproduce the 1.3 model from public data

This guide runs the production training process on the published `1.3` data.
The result is approximately the released `1.3` model.
Three steps stay outside public reproduction:

- **Name counts.** Only AI2 staff can rebuild the name-count index from internal data. Use the released index.
- **Final EPS review.** The release owner reviews giant-block EPS curves on internal production exports. Public calibration reproduces the validation EPS choice, `0.57`. The released model uses the owner's choice, `0.53`; see [giant-block EPS review](eps_calibration.md).
- **LightGBM fits.** Multithreaded LightGBM fits can differ slightly across machines and thread counts. Use 12 threads for the closest match.

The published data reproduces the `1.3` training and validation matrices exactly.
It also reproduces the `1.3` EPS calibration report and outer-test report exactly with the released model.

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

> **Update at v1.3 release:** The S3 bundles are not published yet. Replace this note with the verified location.

```bash
aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and/1.3/inference/ s2and-1.3/
```

The folder now holds `production_model_v1.3/` and `name_counts_index/`.
[Load a production model](production_inference.md#load-a-production-model) for bulk and incremental prediction.

## Train and evaluate: a second sync

```bash
aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and/1.3/dataset/ s2and-1.3/
ROOT=$PWD/s2and-1.3
RUN=$PWD/runs/1.3
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
  --output-dir "$RUN/pairwise/production_model_v1.3" \
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
  --source-bundle "$RUN/pairwise/production_model_v1.3" \
  --model-plan "$ROOT/pairwise/model_plan.json" \
  --output-bundle "$RUN/calibrated/production_model_v1.3" \
  --output-report "$RUN/eps_calibration_report.json" \
  --name-counts-index-root "$ROOT/name_counts_index" \
  --n-jobs 12
```

Calibration clusters each source's validation blocks at every EPS in the plan's grid.
It selects the EPS with the best signature-weighted B3 F1.
Test blocks never select EPS.

### 3. Incremental linker training and the complete model

```bash
uv run --no-sync python scripts/production/model/train_linker_and_finalize.py \
  --source-bundle-root "$ROOT" \
  --target-json tests/fixtures/incremental_linker_training_target.json \
  --pairwise-model-path "$RUN/calibrated/production_model_v1.3" \
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
  --model "$RUN/complete/production_model_v1.3" \
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
  --model-dir "$RUN/complete/production_model_v1.3" \
  --output-dir "$RUN/gold" \
  --n-jobs 12 \
  --total-ram-bytes 34359738368
```

Pass `"$ROOT/production_model_v1.3"` as the model in steps 4 and 5 to score the released model.

## Reference results for 1.3

| Measurement | Released `1.3` value |
| --- | --- |
| Pairwise training and validation pairs | 804,716 and 60,064 |
| Main head validation ROC AUC | 0.994227 (search trial 26) |
| Nameless head validation ROC AUC | 0.974620 (H13) |
| Validation EPS and B3 F1 | 0.57 and 0.959781 on 72,649 signatures |
| Released EPS and its validation B3 F1 | 0.53 and 0.959287 |
| Outer-test signature-weighted B3 F1 | 0.964372 at EPS 0.53 and 0.963514 at EPS 0.57, on 72,853 signatures |
| Outer-test pairwise AUROC | 0.989391 |
| Linker test accuracy | 96.715% on 10,837 queries; EPS does not change it |
| Identity-gold cases | 134 of 152 correct at EPS 0.53 and 0.57 |

Steps 1-5 end at the validation EPS, `0.57`.
The released model uses EPS `0.53` from the final giant-block review.
To compare with the released model, apply that EPS to your complete model:

```bash
uv run --no-sync python scripts/eps_sweep/apply_reviewed_eps.py   --source-model "$RUN/complete/production_model_v1.3"   --output-model "$RUN/released-eps/production_model_v1.3"   --eps 0.53   --review-note "Released 1.3 EPS from the final giant-block review"   --output-report "$RUN/released_eps.json"
```

The command changes only `clusterer.json` and the manifest; the boosters stay the same.
Score the result with steps 4 and 5.

On 12 threads, step 1 takes about one hour and step 3 about 75 minutes.
Steps 2, 4, and 5 take a few minutes each.
