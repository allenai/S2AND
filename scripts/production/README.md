# Production commands

The release owner prepares one corrected snapshot, then writes the published release bundles from it.
Every production fit, including the owner's fit, uses the published `inference` and `dataset` bundles.
Public users therefore run the same commands as the owner.
The [release process](../../docs/release.md) separates data preparation, fit, acceptance, and publication.
The [data guide](../../docs/data.md) defines the three bundles.
The [reproduction guide](../../docs/reproduce.md) lists the public commands in order.

The [data guide](../../docs/data.md#download) lists the release `1.3` populations.
The production count authority is fresh author-table counts, A.
See the [A mainline record](../../docs/release_evidence/author_counts_mainline_20261017.md) for the rebuild evidence.
[Stage 0 of the release process](../../docs/release.md#stage-0-choose-the-input-snapshot) records the selected local snapshot.

## Corrected snapshot

Package corrected records, pairwise populations, and incremental inputs with `package_corrected_datasets.py`.
The dataset manifests store only the relative `name_counts_index` path. The model binds the index manifest hash.
See [snapshot packaging](data/README.md#package-the-current-inputs) for the command, replay options, and clustering threshold (`EPS`) defaults.

## Release bundles

Write the three published bundles from the snapshot:

```powershell
uv run --no-sync python scripts/production/data/write_release_bundles.py `
  --snapshot <combined-snapshot> `
  --output-root <new-bundle-root> `
  --model <complete-model-dir>
```

The command writes `inference`, `dataset`, and `source` under the output root.
Each bundle mirrors one S3 prefix.
`--model` is optional, because the dataset bundle exists before the final model.
Files are hardlinked on the snapshot's volume and copied across volumes.

The [data guide](../../docs/data.md#folder-layout) defines the roles in the dataset model plan, `pairwise/model_plan.json`.
Before it writes the plan, the command proves that the Arrow data rebuilds the snapshot's prepared populations.
Development equals the benchmark minus the held-out blocks and any reserved signatures.
The command also checks the clusters and the unknown-signature lists.
The pre-cut pairwise copies, correction ledgers, and audit records stay in the internal snapshot.

Sync `inference` and `dataset` into one folder to train or evaluate.
The examples below call that folder `$Root`.

## Pairwise model

Set the release root and separate output paths before the fit:

```powershell
$Root = '<synced-inference-and-dataset>'
$RunRoot = '<fresh-fit-output>'
$PairwiseModel = "$RunRoot/pairwise/production_model_v1.3.1"

uv run --no-sync python scripts/production/model/train_pairwise.py `
  --model-plan "$Root/pairwise/model_plan.json" `
  --name-counts-index-root "$Root/name_counts_index" `
  --incumbent-parameters scripts/production/model/pairwise_incumbent_h13.json `
  --output-dir $PairwiseModel `
  --matrix-work-dir "$RunRoot/matrix-work" `
  --total-ram-bytes 137438953472 `
  --n-jobs 12 `
  --run-full
```

The example uses 12 jobs and 128 GiB of RAM.
Use the approved limits for the actual run.
The output directory must be new.
The matrix workspace must be empty and local.
The trainer reads only model-development inputs; it does not require a historical baseline or an evaluation plan.

The model plan declares dataset names instead of a fixed seven-source set.
The current release plan excludes `inventors_s2and`.
Existing augmented records derive from non-patent sources and remain part of release `1.3`.

Use `--help` for pair-count, search, and seed options.
`--n-jobs` also sets `OMP_NUM_THREADS` for the run.
The plan binds each Arrow manifest by digest, and the Arrow row order fixes pair selection.
The trainer rejects invalid augmented source membership before artifact loading and feature allocation.
Unknown signatures remain in inference context but do not supply supervised labels.

Each head runs a 50-trial tree-structured Parzen estimator (TPE) search by default.
`--incumbent-parameters` adds one known LightGBM configuration per head.
The incumbent and the search candidates fit on the same training rows.
The incumbent wins ties on validation area under the receiver operating characteristic curve (ROC AUC).
`--fixed-parameters` takes the same JSON shape for both heads and fits them directly, with no search.
Use it to refit a chosen configuration after a data correction. Pass only one parameter option.
The search and both parameter options limit `num_leaves` to 256 (`MAX_FAST_PLAN_LEAVES`).
The training summary records the selected source and both validation scores.

The output is an immutable pairwise-only model with `eps_calibration: "pending"` and placeholder EPS `0.5`.
It contains both boosters and both required prediction fixtures.
The model manifest declares format and feature-contract versions and ordered main/nameless features.
Unversioned stages require an explicit rebuild or export.

## EPS calibration

Select EPS from validation data with the same model plan:

```powershell
$CalibratedModel = "$RunRoot/calibrated/production_model_v1.3.1"
uv run --no-sync python scripts/production/model/release_pairwise.py calibrate-eps `
  --source-bundle $PairwiseModel `
  --model-plan "$Root/pairwise/model_plan.json" `
  --output-bundle $CalibratedModel `
  --output-report "$RunRoot/eps_calibration_report.json" `
  --name-counts-index-root "$Root/name_counts_index" `
  --total-ram-bytes 137438953472 `
  --n-jobs 12
```

The EPS grid and minimum floors come from the model plan.
Calibration uses validation B-cubed (B3) partitions and the fixed aggregation and tie-break policy.
It requires the development blocks that the model recorded at training time.
It writes a fresh calibrated sibling and preserves the boosters.
This benchmark calibration supplies the initial EPS.
Complete [giant-block EPS review](../../docs/eps_calibration.md) after the complete candidate exists and before final release evaluation.
Apply the manual EPS choice with `scripts/eps_sweep/apply_reviewed_eps.py`.
An EPS-only change does not invalidate incremental features.
A booster or feature-contract change requires new incremental features and a fresh linker fit.

## Complete model and linker evaluation

Freeze incremental task labels, assignments, target, and evaluation policy before the fit:
Use the index path that the root's Arrow manifests resolve.
An identical index at another path does not satisfy the linker preflight check.

```powershell
uv run --no-sync python scripts/production/model/train_linker_and_finalize.py `
  --source-bundle-root $Root `
  --target-json tests/fixtures/incremental_linker_training_target.json `
  --pairwise-model-path $CalibratedModel `
  --name-counts-index-root "$Root/name_counts_index" `
  --output-dir "$RunRoot/complete" `
  --n-jobs 12 `
  --total-ram-bytes 137438953472
```

The finalizer computes features, fits once, writes the complete model, reloads it, and evaluates the declared linker test split.
The derived feature configuration aligns gate calibration with the active retrieval window.
The run report retains evaluation diagnostics and both feature summaries.
Its complete output uses the calibrated model's basename.
The complete model includes the incremental booster, gate metadata, and target JSON.
The pairwise-only input remains unchanged.
The linker binds the exact pairwise feature contract and both boosters.

The trainer validates each derived query against its source signature and paper before a model load.
Every `visibleq:` signature requires a complete source-map entry.
The feature bundle preserves the map bytes and records `initial_query_sources_sha256` in `bundle.json`.
The linker evaluation report records the same digest.
Use `--expected-initial-query-map-sha256` to require a digest from frozen run inputs.
The trainer also enforces any digest that the source bundle declares.

The EPS gold and sweep commands accept the same digest option.
Their summaries, references, and plans record the observed map digest.
EPS application rejects plans with a missing or changed map digest when a source map exists.
Create new sweep plans before EPS application if existing plans omit the map digest.

Unknown-signature sidecars require one plain basename that matches `restored_unknown_signature_ids_*.json`.
Preparation rejects an invalid basename before it creates output.
Preparation, augmentation, packaging, and model readers use the same resolver.
Legacy model inputs without a sidecar retain an empty exclusion set.

Follow [Stage 5](../../docs/release.md#stage-5-fit-evaluate-and-approve) for linker splits and evaluation.
Use the [incremental task guide](../../docs/data.md#incremental-tasks) for safe-action labels.

## Count artifacts

Reuse unchanged common assets.
Build global author-table counts with the [count index producers](counts/README.md#global-name-count-index).
Build Open Researcher and Contributor ID (ORCID) prefix counts with the [ORCID producer](counts/README.md#orcid-prefix-evidence).

The reviewed tuple source reproduces the canonical artifact byte-for-byte:

```powershell
uv run --no-sync python scripts/production/generate_canonical_name_tuples.py `
  --source s2and/data/s2and_unnormalized_filtered_name_tuples.txt `
  --output <fresh-output>/s2and_name_tuples_canonical.txt
```

Regenerate it only when the source or canonical rules change.

## Distribution and Arrow checks

Build one source distribution and one unified wheel for the local platform.
Set `$DistRoot` to a fresh external directory for an asset-only check.
Run from the repository root so Maturin reads `[tool.maturin]` in `pyproject.toml`.
Build for Python 3.11 to match the installed release smoke.

```powershell
$DistRoot = "$RunRoot\stages\local-dist"
uv run --no-project python scripts/sync_version.py --check
uv build --sdist --out-dir $DistRoot --clear
uv run --no-project --python 3.11 --with 'maturin==1.14.1' maturin build `
  --interpreter python `
  --release `
  --locked `
  --compatibility pypi `
  --out $DistRoot
uv run --no-project python `
  scripts/verification/verify_production_model_distributions.py `
  --dist-dir $DistRoot `
  --source-root .
uv run --no-project python `
  scripts/verification/verify_rust_distribution_licenses.py `
  --dist-dir $DistRoot
```

The inventory verifier checks declared runtime assets and rejects packaged model paths and default model declarations.
The license verifier checks both archives for the required notices.

For a bounded local Arrow layout check:

```powershell
uv run python scripts/verification/validate_local_arrow_release.py `
  --release-root path/to/data_root
```

Use `scripts/convert_to_arrow.py validate --dataset-dir ...` when a deep table
scan is required.

## Exact held-out metrics

Evaluate the complete model on the model plan's held-out blocks and fixed held-out pairs:

```powershell
uv run --no-sync python scripts/production/model/release_pairwise.py evaluate-heldout `
  --model-plan "$Root/pairwise/model_plan.json" `
  --model "$RunRoot/complete/production_model_v1.3.1" `
  --name-counts-index-root "$Root/name_counts_index" `
  --output-report "$RunRoot/reports/heldout_metrics.json" `
  --n-jobs 12 `
  --total-ram-bytes 137438953472
```

The command verifies plan file digests, Arrow manifests, and common artifacts.
It binds the model to the plan by the development blocks that the model recorded at training time.
That binding does not depend on the plan layout.
The published data lost records after the released models were fit, so its development blocks differ.
`pairwise/population_corrections.json`, beside the model plan, pairs each benchmark's original population fingerprint with the published one; the command then accepts the released model and prints `population_correction_applied`.
Held-out pairs must stay inside the held-out blocks.
The command rejects augmented train and validation copies of held-out signatures.
It also rejects augmented held-out copies of development signatures.
It uses saved production constraints for B3 prediction.
Unknown signatures remain in prediction context but do not supply supervised scores.
The report contains pairwise metrics, B3 metrics, counts, and input identities.
It requires no baseline and makes no comparison claim.

## Acceptance checks

Run these checks on the final complete model before publication.
Freeze their populations, thresholds, and workloads before you look at candidate scores.
Compare old and new models only on the same population and workload.
The release owner applies the thresholds to the reports and decides whether to publish.

### Full-prediction parity

The parity fixture holds raw incumbent inputs and this `meta.json`:

```json
{
  "dataset": "pubmed",
  "block": "reviewed block value",
  "paths": {
    "signatures": "signatures.json",
    "papers": "papers.json",
    "specter": "specter2.pkl"
  }
}
```

Paths may be absolute or relative to the fixture directory. `block` must occur
in `author_info.block`; the command deterministically takes up to
`--block-size` sorted signature IDs. Add
`paths.cluster_seeds_require` only when passing `--use-cluster-seeds`, or pass
`--no-specter` when deliberately omitting the embedding pickle.

```powershell
uv run python scripts/verification/compare_full_predict_arrow_parity.py `
  --fixture-dir path/to/reviewed_parity_fixture `
  --output-dir path/to/fresh/parity_work `
  --output-json "$RunRoot/reports/parity_evaluation_report.json" `
  --name-counts-index "$Root/name_counts_index" `
  --model-path "$RunRoot/complete/production_model_v1.3.1" `
  --block-size REVIEWED_BLOCK_SIZE `
  --n-jobs 12 `
  --total-ram-bytes 137438953472
```

The command compares Python and Arrow distances, features, constraints, and clusters on one block.
It fails on any mismatch and then writes no report.

### Subblocking quality

Run native subblocking on one reviewed Arrow block with the ORCID repair
disabled. Inspect subblock sizes and the share of repeated ORCID groups
preserved by the partition.

```powershell
uv run python scripts/analyze_giant_block_subblocking.py `
  --arrow-dataset "$Root/benchmarks/REVIEWED_DATASET" `
  --block "REVIEWED_BLOCK" `
  --thresholds REVIEWED_MAXIMUM_SIZE `
  --output-dir "$RunRoot/reports/subblocking" `
  --random-seed REVIEWED_SEED `
  --n-jobs 12
```

Confirm that the largest subblock fits the requested threshold and record the
preservation metrics with the release evidence.

### Promoted incremental performance

Run a bounded profile first:

```powershell
uv run --with psutil python scripts/verification/profile_promoted_incremental_arrow.py `
  --model-path "$RunRoot/complete/production_model_v1.3.1" `
  --arrow-root path/to/reviewed_arrow_collection_root `
  --dataset REVIEWED_DATASET `
  --query-limit 400 `
  --max-seed-clusters 400 `
  --n-jobs 12 `
  --require-rust-release `
  --write-json "$RunRoot/reports/performance_evaluation_report.json"
```

With the released data in `s2and/data/`, use `--arrow-root s2and/data --dataset aminer`.
The report path must be new.
Add `--full-run` after the bounded run for a workload above the built-in query or seed limits.
The report records the workload and Rust extension identity.
It records per-run time, process-tree resident set size (RSS), and summary statistics.
Keep command logs beside the report.

## Installed release smoke

Install the candidate wheel into an empty environment and run outside the checkout.
Set the path config to the downloaded public-data root before Python imports `s2and`.

The smoke command uses an 8 GiB budget.

```powershell
$DistRoot = "$RunRoot\stages\local-dist"
$PublicDataRoot = $Root
$SmokeVenv = "$RunRoot\stages\installed-smoke-venv"
$SmokePython = "$SmokeVenv\Scripts\python.exe"
$PathConfig = "$RunRoot\stages\installed-smoke-path-config.json"
$Repo = (Resolve-Path .).Path

# Write UTF-8 without a BOM; Windows PowerShell's `-Encoding utf8` adds one,
# which the S2AND config loader rejects.
$PathConfigJson = @{ main_data_dir = (Resolve-Path $PublicDataRoot).Path } | ConvertTo-Json
[IO.File]::WriteAllText($PathConfig, $PathConfigJson, (New-Object System.Text.UTF8Encoding($false)))

uv venv --python 3.11 $SmokeVenv
uv pip install --python $SmokePython `
  (Get-ChildItem "$DistRoot\*.whl").FullName

$env:S2AND_PATH_CONFIG = $PathConfig
Push-Location $RunRoot
uv run --no-project --python $SmokePython python `
  "$Repo\scripts\verification\smoke_installed_incremental_arrow.py" `
  release-candidate `
  --model-dir "$RunRoot/complete/production_model_v1.3.1" `
  --data-root $PublicDataRoot `
  --dataset REVIEWED_DATASET `
  --signature-ids REVIEWED_SEED_1 REVIEWED_SEED_2 REVIEWED_QUERY
Pop-Location
```

Success reports `bulk_pair_count: 3`, `bulk_signature_count: 3`,
`arrow_promoted_incremental: 1`, `query_view: raw_arrow`, and
`signature_count: 3`. To smoke the public packages instead of candidate
wheels, use a second empty environment with the same script and arguments:

```powershell
$PublicSmokeVenv = "$RunRoot\stages\public-smoke-venv"
$PublicSmokePython = "$PublicSmokeVenv\Scripts\python.exe"
$Version = (Get-Content -Raw (Join-Path $Repo 'VERSION')).Trim()
uv venv --python 3.11 $PublicSmokeVenv
uv pip install --python $PublicSmokePython `
  --refresh `
  --default-index https://pypi.org/simple `
  "s2and==$Version"
```

Use `$PublicSmokePython` in the `uv run --no-project --python ...` invocation
above; keep the same path config, model, data root, dataset, and signature IDs.
