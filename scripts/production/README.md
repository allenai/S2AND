# Production commands

The release owner prepares one corrected snapshot, then writes the published release bundles from it.
Every production fit, including the owner's fit, uses the published `inference` and `dataset` bundles.
Public users therefore run the same commands as the owner.
The [release process](../../docs/release.md) separates data preparation, fit, acceptance, and publication.
The [data guide](../../docs/data.md) defines the three bundles.
The [reproduction guide](../../docs/reproduce.md) lists the public commands in order.

The [data guide](../../docs/data.md#download) lists the release `1.3` populations.
The production count authority is fresh author-table counts, A.
The [mainline record](../../docs/release_evidence/author_counts_mainline_20261017.md) defines the selected policy and rebuild status.
The [earlier candidate record](../../docs/release_evidence/production_candidate_20261013.md) preserves the previous model's evidence.
[Stage 0 of the release process](../../docs/release.md#stage-0-choose-the-input-snapshot) records the selected local snapshot.

## Corrected snapshot

Create a fresh combined local root:

```powershell
uv run --no-sync python scripts/production/data/package_corrected_datasets.py `
  --corrected-source-root <final-corrected-master> `
  --pairwise-inputs-root <prepared-pairwise-package> `
  --public-data-root <prepared-public-root> `
  --linker-source-bundle <accepted-incremental-root> `
  --corrected-replay-root <current-replay-raw-root> `
  --corrected-replay-embeddings-root <current-replay-vector-root> `
  --name-counts-index <common-name-counts-index> `
  --output-root <new-combined-snapshot>
```

The output contains full corrected records, exact pairwise populations, benchmark Arrow tables, and complete incremental support.
The internal `pairwise/model_plan.json` names the JSON fit inputs and the EPS policy.
The trainer rejects it; train from the published plan in the `dataset` bundle.
The default EPS grid uses steps of `0.01` from `0.00` to `1.00`; both minimum F1 floors equal zero.
Use explicit EPS options to declare a different policy.
All operational paths remain relative to the snapshot.
One physical `name_counts_index` serves all Arrow datasets.
The dataset manifests bind the common index's exact hash.
The `inference` bundle publishes that index; the `dataset` bundle does not contain it.

A future data correction starts from `corrected_sources` in this snapshot.
Historical review archives and separate pre-correction inputs are not required.
The two replay options preserve current raw records and vectors for future updates.
Supply both options together when the snapshot includes replay sources.
See [data preparation](data/README.md) for augmentation and embedding updates.

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
$PairwiseModel = "$RunRoot/pairwise/production_model_v1.3"

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
The published `1.3` data reproduces the `1.3` training and validation matrices exactly.
The trainer rejects invalid augmented source membership before artifact loading and feature allocation.
Unknown signatures remain in inference context but do not supply supervised labels.

Each head runs the official 50-trial TPE search.
`--incumbent-parameters` adds one known LightGBM configuration per head.
The incumbent fits on the same training rows and wins ties on validation ROC AUC.
Release `1.3` used the H13 configuration in `pairwise_incumbent_h13.json`.
The `1.3` main head kept search trial 26; the nameless head kept H13.
The training summary records the selected source and both validation scores.

The output is an immutable pairwise-only model with `eps_calibration: "pending"` and placeholder EPS `0.5`.
It contains both boosters and both required prediction fixtures.
The model manifest declares format and feature-contract versions and ordered main/nameless features.
Unversioned stages require an explicit rebuild or export.

## EPS calibration

Select EPS from validation data with the same model plan:

```powershell
$CalibratedModel = "$RunRoot/calibrated/production_model_v1.3"
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
Calibration uses validation B3 partitions and the fixed aggregation and tie-break policy.
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

The fixed classic booster target uses 53 features and 675 trees.
The booster fits train rows.
Both `calibration_fit` and `calibration_check` select gate features and fit gate coefficients.
Both splits serve as dev populations.
Retain existing manual dev labels.
Exclude verified query-paper exposure before gate calibration.

The safe-action target is not universal identity gold.
An abstention can indicate ambiguity or insufficient evidence.
Recheck affected labels when query evidence or profile membership changes.
A new pairwise model requires fresh features but does not automatically require new manual judgments.
Added components can change other queries' features within the same block.

Linker reports distinguish stratified test results from full-source diagnostics.
The latter can include gate-calibration queries.
Prior development includes the inherited test population.
Its stratified test scores provide regression evidence.

Use [evaluate_incremental_gold.py](../verification/evaluate_incremental_gold.py) for frozen fresh identity cases and the complete serialized model.
Identity gold differs from manual safe-action labels.
Keep their reports separate.

## Count artifacts

Reuse unchanged common assets.
Small tuple and ORCID files remain in normal Git and the wheel.
The binary name-count index uses Git LFS and remains outside the wheel.
Release `1.3` uses `artifacts/common/v1.3/name_counts_index`.
The complete model target is `artifacts/common/v1.3/production_model_v1.3`.
This path contains the A release candidate. Final release acceptance remains pending.
See [the count producers](counts/README.md) for source columns, workload bounds, and native index construction.

The reviewed tuple source reproduces the canonical artifact byte-for-byte:

```powershell
uv run --no-sync python scripts/production/generate_canonical_name_tuples.py `
  --source s2and/data/s2and_unnormalized_filtered_name_tuples.txt `
  --output <fresh-output>/s2and_name_tuples_canonical.txt
```

Regenerate it only when the source or canonical rules change.
Generate global A counts from every row in the Athena author table.
Use the [Athena author-count command](counts/README.md#athena-author-row-export).
It preserves raw name groups before the current repository normalizer creates the count keys.

For an existing verified author-row export, use the shared Parquet producer:

```powershell
uv run --no-sync --with 'duckdb>=1.5,<2' python -m scripts.production.counts.generate_name_counts_from_parquet `
  --input-manifest <approved-name-count-export.json> `
  --expected-source-rows <raw-name-group-count> `
  --expected-source-mentions <author-table-row-count> `
  --threads 4 --memory-limit 12GB `
  --output-dir <fresh-count-output>

uv run --no-sync python -m scripts.production.counts.generate_orcid_name_prefix_counts `
  --input-csv <orcid-name-export.csv> `
  --guardrails-json <approved-orcid-bounds.json> `
  --output-dir <fresh-orcid-output>
```

The shared producer retains `mention_count` and `--expected-source-mentions` as weight field names.
For A, both values represent author-table rows.
The archived M study does not define the current production count authority.

The ORCID input contains `raw_orcid,orcid,first_name,middle,last_name` in the required source order.
The producer enforces the shared nonempty surname rule and the 14-group exclusion.
The surname comparison removes spaces after canonicalization; `Ou Yang` and `Ouyang` match.
Pairs must share their own surname key; surname agreement is not transitive.

Copy the approved ORCID JSON and tuple-dependency manifest to these package paths:

- `s2and/data/first_k_letter_counts_from_orcid.json`
- `s2and/data/first_k_letter_counts_from_orcid.manifest.json`

A tuple or ORCID asset change requires affected model exports to bind the new hashes.
Models reject a mismatch with required package assets.

## Distribution and Arrow checks

Build one local-platform candidate of each distribution and verify the Python
wheel/sdist contents. Use a fresh external `$DistRoot` for the asset
check and `$RunRoot\stages\local-dist` for the final candidate:

```powershell
$PythonDist = "$DistRoot\python"
$RustDist = "$DistRoot\rust"

uv run --no-project python scripts/sync_version.py --check
$PackageVersion = (Get-Content -Raw VERSION).Trim()
uv build --sdist --out-dir $PythonDist --clear
uv build "$PythonDist\s2and-$PackageVersion.tar.gz" --wheel --out-dir $PythonDist
uv run --no-project maturin build `
  --manifest-path s2and_rust/Cargo.toml `
  --release `
  --locked `
  --compatibility pypi `
  --out $RustDist
uv run --no-project python `
  scripts/verification/verify_production_model_distributions.py `
  --dist-dir $PythonDist `
  --source-root .
```

The verifier requires the package's tuple and ORCID runtime assets and rejects
packaged model/default paths.
The fresh source archive excludes stale files from the checkout's `build/` directory.

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
  --model "$RunRoot/complete/production_model_v1.3" `
  --name-counts-index-root "$Root/name_counts_index" `
  --output-report "$RunRoot/reports/heldout_metrics.json" `
  --n-jobs 12 `
  --total-ram-bytes 137438953472
```

The command verifies plan file digests, Arrow manifests, and common artifacts.
It binds the model to the plan by the development blocks that the model recorded at training time.
That binding does not depend on the plan layout, so the published `1.3` model passes it.
Held-out pairs must stay inside the held-out blocks.
The command rejects augmented train and validation copies of held-out signatures.
It also rejects augmented held-out copies of development signatures.
It uses saved production constraints for B3 prediction.
Unknown signatures remain in prediction context but do not supply supervised scores.
The report contains pairwise metrics, B3 metrics, counts, and input identities.
It requires no baseline and makes no comparison claim.
Keep fresh incremental identity tests and inherited safe-action reports separate.

## Acceptance checks

Run these checks on the final complete model before publication.
Freeze their populations, thresholds, and workloads before you look at candidate scores.
Compare old and new models only on the same population and workload.
No command applies thresholds; the release owner reads the reports and decides.
Keep the finalizer's linker report and the identity-gold report separate from these checks.

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
  --model-path "$RunRoot/complete/production_model_v1.3" `
  --block-size REVIEWED_BLOCK_SIZE `
  --n-jobs 12 `
  --total-ram-bytes 137438953472
```

The command compares Python and Arrow distances, features, constraints, and clusters on one block.
It fails on any mismatch and then writes no report.

### Subblocking quality

```powershell
uv run python scripts/verification/compare_graph_subblocking_arrow_quality.py `
  --arrow-root "$Root/benchmarks/REVIEWED_DATASET" `
  --component-members-parquet path/to/reviewed_components.parquet `
  --output-dir "$RunRoot/reports/subblocking" `
  --maximum-size REVIEWED_MAXIMUM_SIZE `
  --allow-full `
  --seed REVIEWED_SEED
```

The component-members file needs `signature_id` and `candidate_component_key` columns.
Rows for signatures outside the Arrow dataset are ignored.
Pass `--baseline-summary` with an earlier report to add deltas.

### Promoted incremental performance

Run a bounded profile first:

```powershell
uv run --with psutil python scripts/verification/profile_promoted_incremental_arrow.py `
  --model-path "$RunRoot/complete/production_model_v1.3" `
  --arrow-root path/to/reviewed_arrow_collection_root `
  --dataset REVIEWED_DATASET `
  --query-limit 400 `
  --max-seed-clusters 400 `
  --n-jobs 12 `
  --require-rust-release `
  --write-json "$RunRoot/reports/performance_evaluation_report.json"
```

The report path must be new.
Add `--full-run` after the bounded run for a workload above the built-in query or seed limits.
The report records the workload, the Rust extension identity, per-run time and process-tree RSS, and summary statistics.
Keep command logs beside the report.
Historical measurements remain in [the profiling directory](../../docs/rust/profiling/); they are evidence snapshots, not active commands.

## Installed release smoke

Install the exact candidate wheels into an empty environment and run from
outside the checkout. The path config must select the downloaded public-data
root before Python imports `s2and`:

The smoke command uses an 8 GiB budget. The common count index alone exceeds the previous 2 GB budget.

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
  (Get-ChildItem "$DistRoot\rust\*.whl").FullName `
  (Get-ChildItem "$DistRoot\python\*.whl").FullName

$env:S2AND_PATH_CONFIG = $PathConfig
Push-Location $RunRoot
uv run --no-project --python $SmokePython python `
  "$Repo\scripts\verification\smoke_installed_incremental_arrow.py" `
  release-candidate `
  --model-dir "$RunRoot/complete/production_model_v1.3" `
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
  "s2and==$Version" `
  "s2and-rust==$Version"
```

Use `$PublicSmokePython` in the `uv run --no-project --python ...` invocation
above; keep the same path config, model, data root, dataset, and signature IDs.
