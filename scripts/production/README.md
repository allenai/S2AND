# Production commands

Use the complete corrected snapshot for production fits.
This is the output of all accepted additions and corrections, not a reconstruction from old source archives.
The [release process](../../docs/release.md) separates data preparation, fit, acceptance, and publication.
The [data guide](../../docs/data.md) defines common Git LFS assets and the S3 research snapshot.

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
`pairwise/model_plan.json` declares the production fit datasets and EPS policy.
The default EPS grid uses steps of `0.01` from `0.00` to `1.00`; both minimum F1 floors equal zero.
Use explicit EPS options to declare a different policy.
All operational paths remain relative to the snapshot.
One physical `name_counts_index` serves all Arrow datasets.
The dataset manifests bind the common index's exact hash.
Exclude that index from the S3 payload.

A future data correction starts from `corrected_sources` in this snapshot.
Historical review archives and separate pre-correction inputs are not required.
The two replay options preserve current raw records and vectors for future updates.
Supply both options together when the snapshot includes replay sources.
See [data preparation](data/README.md) for augmentation and embedding updates.

## Pairwise model

Set the snapshot and separate output paths before the fit:

```powershell
$Snapshot = '<combined-snapshot>'
$RunRoot = '<fresh-fit-output>'
$PairwiseModel = "$RunRoot/pairwise/production_model_v1.3"

uv run --no-sync python scripts/production/model/train_pairwise.py `
  --model-plan "$Snapshot/pairwise/model_plan.json" `
  --arrow-root "$Snapshot/pairwise/arrow/release" `
  --name-counts-index-root "$Snapshot/name_counts_index" `
  --output-dir $PairwiseModel `
  --matrix-work-dir "$RunRoot/matrix-work" `
  --total-ram-bytes 137438953472 `
  --n-jobs 20 `
  --run-full
```

The example uses 20 jobs and 128 GiB of RAM.
Use the approved limits for the actual run.
The output directory must be new.
The matrix workspace must be empty and local.
The trainer reads only model-development inputs; it does not require a historical baseline or an evaluation plan.

The model plan declares dataset names instead of a fixed seven-source set.
It can include `inventors_s2and` with the benchmark datasets and augmented fixed pairs.
The current patents snapshot has 2,500 blocks, 41,468 signatures, and 41,316 papers.
Its block split can share papers; it is not whole-paper independent.
Existing augmented records remain unchanged for the next patents-inclusive fit.

Use `--help` for pair-count, search, and seed options.
`--n-jobs` also sets `OMP_NUM_THREADS` for the run.
Source order affects pair selection, so keep the exact source records and declared splits.
The trainer rejects invalid augmented source membership before feature allocation.
Unknown signatures remain in inference context but do not supply supervised labels.

The output is an immutable pairwise-only model with `eps_calibration: "pending"` and placeholder EPS `0.5`.
It contains both boosters and both required prediction fixtures.
The model manifest declares format and feature-contract versions and ordered main/nameless features.
The generating runtime is provenance, not an exact package-version lock.
Unversioned stages require an explicit rebuild or export.

## EPS calibration

Select EPS from validation data with the same model plan:

```powershell
$CalibratedModel = "$RunRoot/calibrated/production_model_v1.3"
uv run --no-sync python scripts/production/model/release_pairwise.py calibrate-eps `
  --source-bundle $PairwiseModel `
  --model-plan "$Snapshot/pairwise/model_plan.json" `
  --output-bundle $CalibratedModel `
  --output-report "$RunRoot/eps_calibration_report.json" `
  --name-counts-index-root "$Snapshot/name_counts_index" `
  --arrow-root "$Snapshot/pairwise/arrow/release" `
  --total-ram-bytes 137438953472 `
  --n-jobs 20
```

The EPS grid and minimum floors come from the model plan.
Calibration uses validation B3 partitions and the fixed aggregation and tie-break policy.
It writes a fresh calibrated sibling and preserves the boosters.
An EPS-only change does not invalidate incremental features.
A booster or feature-contract change requires new incremental features and a fresh linker fit.

## Complete model and linker evaluation

Freeze incremental task labels, assignments, target, and evaluation policy before the fit:

```powershell
uv run --no-sync python scripts/production/model/train_linker_and_finalize.py `
  --source-bundle-root $Snapshot `
  --target-json tests/fixtures/incremental_linker_training_target.json `
  --pairwise-model-path $CalibratedModel `
  --name-counts-index-root "$Snapshot/name_counts_index" `
  --output-dir "$RunRoot/complete" `
  --n-jobs 20 `
  --total-ram-bytes 137438953472
```

The finalizer computes features, fits once, writes the complete model, reloads it, and evaluates the declared linker test split.
The derived feature configuration aligns gate calibration with the active retrieval window.
The run report retains evaluation diagnostics and both feature summaries.
Its complete output uses the calibrated model's basename.
The complete model includes the incremental booster, gate metadata, and target JSON.
The pairwise-only input remains unchanged.
The linker binds the exact pairwise feature contract and both boosters.

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
The accepted complete model uses `artifacts/common/v1.3/production_model_v1.3`.
See [the count producers](counts/README.md) for source columns, workload bounds, and native index construction.

The reviewed tuple source reproduces the canonical artifact byte-for-byte:

```powershell
uv run --no-sync python scripts/production/generate_canonical_name_tuples.py `
  --source s2and/data/s2and_unnormalized_filtered_name_tuples.txt `
  --output <fresh-output>/s2and_name_tuples_canonical.txt
```

Regenerate it only when the source or canonical rules change.
Generate global counts from the full source corpus rather than a research query subset:

```powershell
uv run --no-sync --with 'duckdb>=1.5,<2' python -m scripts.production.counts.generate_name_counts_from_parquet `
  --input-manifest <approved-name-count-export.json> `
  --expected-source-rows <approved-row-count> `
  --expected-source-mentions <approved-mention-count> `
  --threads 4 --memory-limit 12GB `
  --output-dir <fresh-count-output>

uv run --no-sync python -m scripts.production.counts.generate_orcid_name_prefix_counts `
  --input-csv <orcid-name-export.csv> `
  --guardrails-json <approved-orcid-bounds.json> `
  --output-dir <fresh-orcid-output>
```

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

Evaluate the complete model on the snapshot's declared outer holdouts and fixed pairs:

```powershell
uv run --no-sync python scripts/production/model/release_pairwise.py evaluate-heldout `
  --inputs-json "$Snapshot/pairwise/inputs.json" `
  --model "$RunRoot/complete/production_model_v1.3" `
  --arrow-root "$Snapshot/pairwise/arrow/evaluation" `
  --name-counts-index-root "$Snapshot/name_counts_index" `
  --output-report "$RunRoot/reports/heldout_metrics.json" `
  --n-jobs 20 `
  --total-ram-bytes 137438953472
```

The command verifies source hashes, Arrow provenance, common artifacts, and the model's fit-plan identity.
It uses saved production constraints for B3 prediction.
Unknown signatures remain in prediction context but do not supply supervised scores.
The report contains pairwise metrics, B3 metrics, counts, and input identities.
It requires no baseline and makes no comparison claim.
Keep fresh incremental identity tests and inherited safe-action reports separate.

## Promotion policy

The commands below apply release acceptance gates.
They are not prerequisites for snapshot preparation, pairwise fit, or EPS calibration.
Freeze their populations, metric contracts, thresholds, and workloads before candidate scores are available.
A matched baseline uses those same inputs and workloads.
Historical rates from different data or splits are not a matched comparison.
The current acceptance CLI requires a baseline record; this requirement belongs to promotion.

## Run preparation for promotion

Prepare this separate acceptance plan after the fit inputs are fixed.
Preparation checks train/validation/held-out identity overlap and writes the
two plans used by later stages:

```powershell
uv run python scripts/production/model/release_pairwise.py `
  prepare-run `
  --release path/to/run/release.json
```

The run then contains:

```text
release.json
model_plan.json
evaluation_plan.json
run_binding.json  # added only after bind-candidate
stages/
reports/
final/            # created by prepare-run; no current command writes to it
```

This is the smallest random-split `release.json` shape accepted by
`prepare-run`; it demonstrates the schema and path binding only. Paths are
resolved relative to the run directory. The numbers are examples, not reviewed
release values:

```json
{
  "release_version": "1.3",
  "model": {
    "datasets": {
      "random": {
        "signatures": "../inputs/train/random_signatures.json",
        "papers": "../inputs/train/random_papers.json",
        "specter_embeddings": "../inputs/train/random_specter2.pkl",
        "clusters": "../inputs/train/random_clusters.json"
      }
    },
    "eps": {
      "grid": [0.3, 0.6],
      "minimum_dataset_f1": 0.0,
      "minimum_signature_weighted_f1": 0.0
    }
  },
  "evaluation": {
    "pairwise": {
      "random": {
        "signatures": "../inputs/test/random_signatures.json",
        "papers": "../inputs/test/random_papers.json",
        "specter_embeddings": "../inputs/test/random_specter2.pkl",
        "pairs": "../inputs/test/random_pairs.json"
      }
    },
    "cluster": {
      "random": {
        "signatures": "../inputs/test/random_signatures.json",
        "papers": "../inputs/test/random_papers.json",
        "specter_embeddings": "../inputs/test/random_specter2.pkl",
        "clusters": "../inputs/test/random_clusters.json",
        "blocks": "../inputs/test/random_blocks.json"
      }
    },
    "performance": {
      "arrow_root": "../inputs/performance_arrow",
      "workload": {
        "dataset": "random",
        "target_block": "",
        "query_limit": 400,
        "max_seed_clusters": 400,
        "seed_source": "clusters",
        "runs": 5,
        "n_jobs": 4,
        "batching_threshold": null,
        "total_ram_bytes": null,
        "synthetic_seeds_when_clusters_missing": false
      }
    },
    "parity": {
      "fixture_dir": "../inputs/parity_fixture",
      "workload": {
        "block_size": 100,
        "compare_features": true,
        "include_specter": true,
        "n_jobs": 4,
        "total_ram_bytes": 1000000000,
        "use_cluster_seeds": false
      }
    },
    "subblocking": {
      "component_members_parquet": "../inputs/candidate_components.parquet",
      "dataset": "pubmed",
      "workload": {
        "allow_full": true,
        "comparison_mode": "python-vs-rust",
        "graph_config": {
          "neighbor_mode": "projection",
          "neighbors": 16,
          "min_edge_score": 0.3,
          "specter_weight": 1.0,
          "coauthor_weight": 0.35,
          "affiliation_weight": 0.2,
          "max_exact_knn_group_size": 25000,
          "projection_count": 12,
          "projection_window": 12,
          "max_candidate_edges": 5000000,
          "pack_components": true,
          "component_pack_strategy": "edge-greedy",
          "sparse_evidence_edges": true,
          "sparse_evidence_max_posting_size": 8,
          "sparse_evidence_neighbors": 1,
          "sparse_evidence_min_weight": 0.4,
          "sparse_evidence_include_coauthors": true,
          "sparse_evidence_include_affiliations": false,
          "component_pack_top_k": 8,
          "local_move_passes": 0,
          "adaptive_projection": false,
          "adaptive_projection_max_group_size": 5000,
          "adaptive_projection_count": 24,
          "adaptive_projection_window": 24
        },
        "limit": null,
        "maximum_size": 100,
        "orcid_subblocking": true,
        "python_source": "arrow",
        "sample_mode": "random",
        "seed": 42,
        "top_diff_subblocks": 30
      }
    },
    "baseline": {
      "path": "../inputs/matched_baseline.json",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    },
    "gates": {
      "cluster_signature_weighted_b3_f1_max_drop": 0.005,
      "pairwise_aggregate_auroc_max_drop": 0.001,
      "pairwise_aggregate_macro_f1_max_drop": 0.005,
      "pairwise_dataset_auroc_max_drop": 0.001,
      "pairwise_dataset_macro_f1_max_drop": 0.005,
      "peak_rss_absolute_max_gb": 4.0,
      "runtime_max_ratio": 1.1,
      "subblocking_maximum_size": 100
    }
  }
}
```

The referenced baseline record also has an exact shape. It contains no
authoritative paths: population and performance identities are content hashes,
and its workload must equal the `release.json` workload:

```json
{
  "metric_contract": {
    "cluster_b3_aggregation": "signature_count_weighted",
    "pairwise_macro_f1_threshold": {"operator": ">", "value": 0.5},
    "pairwise_probability_combination": "arithmetic_mean_main_nameless",
    "performance_statistic": "predict_seconds_p50"
  },
  "metrics": {
    "cluster_signature_weighted_b3_f1": 0.804,
    "pairwise_aggregate": {"auroc": 0.9005, "macro_f1": 0.804},
    "pairwise_datasets": {
      "random": {"auroc": 0.9005, "macro_f1": 0.804}
    },
    "predict_seconds_p50": 10.0
  },
  "performance": {
    "arrow_root_manifest_sha256": "<lowercase SHA-256>",
    "workload": {
      "dataset": "random",
      "target_block": "",
      "query_limit": 400,
      "max_seed_clusters": 400,
      "seed_source": "clusters",
      "runs": 5,
      "n_jobs": 4,
      "batching_threshold": null,
      "total_ram_bytes": null,
      "synthetic_seeds_when_clusters_missing": false
    }
  },
  "populations": {
    "pairwise": {
      "random": {
        "signatures": "<lowercase SHA-256>",
        "papers": "<lowercase SHA-256>",
        "specter_embeddings": "<lowercase SHA-256>",
        "pairs": "<lowercase SHA-256>"
      }
    },
    "cluster": {
      "random": {
        "signatures": "<lowercase SHA-256>",
        "papers": "<lowercase SHA-256>",
        "specter_embeddings": "<lowercase SHA-256>",
        "clusters": "<lowercase SHA-256>",
        "blocks": "<lowercase SHA-256>"
      }
    }
  },
  "provenance": {
    "source_commit": "e54c6ba9c0e3ca4c2b5a40dcaa9a55c2c771d87d",
    "model": {"version": "1.21", "eps": 0.65},
    "data": {"identity": "<reviewed data identity>"},
    "environment": {"identity": "<reviewed environment identity>"},
    "commands": ["<exact historical baseline command>"]
  }
}
```

Every object has an exact key set and extra keys fail, except
`evaluation.performance.workload`: `prepare-run` only requires a nonempty
mapping equal to the baseline record's workload, and the performance profiler
enforces its exact keys later. A random-split model
dataset uses `signatures`, `papers`, `specter_embeddings`, and `clusters`, and
has same-named pairwise and cluster evaluation datasets. A fixed-pair model
dataset replaces `clusters` with `train_pairs` and `val_pairs` and has a
same-named pairwise evaluation dataset. Model and evaluation identities must
be disjoint.

Fixed-pair training and validation CSVs use `signature_id_1`, `signature_id_2`,
and `label` columns. Columns are selected by name, so their order does not
matter. Labels may be `YES`/`NO` or `1`/`0`.

Before candidate reports, bind the complete model and combined snapshot to the prepared acceptance plans:

```powershell
uv run --no-sync python scripts/production/model/release_pairwise.py bind-candidate `
  --run-root <acceptance-run-root> `
  --candidate-model "$RunRoot/complete/production_model_v1.3" `
  --public-data-root $Snapshot
```

The acceptance model plan must identify the same fit inputs, EPS policy, and model-plan content as the candidate's training record.
Check this binding before report generation.
Do not substitute a full benchmark projection for the declared fit or evaluation population.

## Evaluation

The component evaluators write numeric pairwise, cluster, subblocking, parity,
and performance reports. Pairwise, cluster, parity, and performance depend on
the reloaded complete model. Subblocking depends only on the frozen Arrow and
candidate-component inputs. The finalizer's linker evaluation report covers a
different source-bundle split and is retained separately; it is not one of
these five aggregate-gate inputs.

Put the five reports in the run report directory:

```text
pairwise_evaluation_report.json
cluster_evaluation_report.json
performance_evaluation_report.json
subblocking_evaluation_report.json
parity_evaluation_report.json
```

The parity fixture contains raw incumbent inputs and this `meta.json`:

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

For subblocking, pass exactly the file frozen as
`evaluation.subblocking.component_members_parquet` in `release.json`. The
evaluator compares both its resolved path and SHA-256 with the evaluation plan,
so a copy (for example, the one inside the assembled linker-source bundle) is
rejected. The file must contain `signature_id` and `candidate_component_key`
columns; rows for signatures outside the selected Arrow root are ignored.

Produce pairwise, cluster, subblocking, and parity reports:

```powershell
uv run python scripts/production/model/release_pairwise.py evaluate-pairs `
  --model path/to/production_model_vX.Y `
  --evaluation-plan path/to/run/evaluation_plan.json `
  --run-binding path/to/run/run_binding.json `
  --name-counts-index-root path/to/name_counts_index `
  --arrow-root "$Snapshot/pairwise/arrow/evaluation" `
  --output-report path/to/run/reports/pairwise_evaluation_report.json `
  --n-jobs REVIEWED_N_JOBS `
  --total-ram-bytes REVIEWED_TOTAL_RAM_BYTES

uv run python scripts/production/model/release_pairwise.py evaluate-clusters `
  --model path/to/production_model_vX.Y `
  --evaluation-plan path/to/run/evaluation_plan.json `
  --run-binding path/to/run/run_binding.json `
  --name-counts-index-root path/to/name_counts_index `
  --arrow-root "$Snapshot/pairwise/arrow/evaluation" `
  --output-report path/to/run/reports/cluster_evaluation_report.json `
  --n-jobs REVIEWED_N_JOBS

uv run python scripts/verification/compare_graph_subblocking_arrow_quality.py `
  --public-data-root $Snapshot `
  --arrow-root "$Snapshot/benchmarks/REVIEWED_DATASET" `
  --evaluation-plan path/to/run/evaluation_plan.json `
  --run-binding path/to/run/run_binding.json `
  --output-dir path/to/run/reports `
  --python-source arrow `
  --comparison-mode python-vs-rust `
  --component-members-parquet path/to/reviewed_components.parquet `
  --maximum-size REVIEWED_MAXIMUM_SIZE `
  --allow-full `
  --seed REVIEWED_SEED

uv run python scripts/verification/compare_full_predict_arrow_parity.py `
  --fixture-dir path/to/reviewed_parity_fixture `
  --evaluation-plan path/to/run/evaluation_plan.json `
  --output-dir path/to/fresh/parity_work `
  --output-json path/to/run/reports/parity_evaluation_report.json `
  --name-counts-index path/to/name_counts_index `
  --model-path path/to/production_model_vX.Y `
  --run-binding path/to/run/run_binding.json `
  --block-size REVIEWED_BLOCK_SIZE `
  --n-jobs REVIEWED_N_JOBS `
  --total-ram-bytes REVIEWED_TOTAL_RAM_BYTES
```

The parity command verifies every fixture file and all behavior-affecting
options against the frozen evaluation plan. Subblocking verifies the bound
public-data root, resolves the frozen dataset name through that root's
manifest, and verifies the component-members file and full
workload/configuration. No component report can stamp the run binding on an
alternate population or workload.

### Promoted incremental performance report

The release performance workload is promoted incremental linking against the
reviewed Arrow replay root and complete production model. Run a bounded
profile first:

```powershell
uv run --with psutil python scripts/verification/profile_promoted_incremental_arrow.py `
  --evaluation-plan path/to/run/evaluation_plan.json `
  --model-path path/to/production_model_vX.Y `
  --run-binding path/to/run/run_binding.json `
  --require-rust-release `
  --write-json path/to/run/reports/performance_evaluation_report.json
```

The report path must be fresh. The frozen evaluation plan supplies the Arrow
root and exact workload. For an approved release workload above the built-in
query or seed limits, add `--full-run` after the bounded rehearsal; the flag
does not change the plan's workload or authorize a larger run.

The command records the exact workload, Rust extension identity, per-run
timing and process-tree RSS, and summary statistics. Release evaluation
consumes:

- the prepared run's `run_binding_sha256`;
- the exact reviewed `workload` and `arrow_root`;
- `summary.predict_seconds.p50`; and
- `summary.peak_rss_gb.max`.

Keep command logs beside the report. Historical measurements remain in
[the profiling directory](../../docs/rust/profiling/); they are evidence
snapshots, not active commands.

### Apply release gates

Apply the gates directly:

```powershell
uv run python scripts/production/model/release_pairwise.py `
  evaluate-release `
  --evaluation-plan path/to/run/evaluation_plan.json `
  --run-binding path/to/run/run_binding.json `
  --report-dir path/to/run/reports `
  --output-report path/to/run/reports/evaluation_report.json
```

Every component report must carry the same `run_binding_sha256`. The command
rejects a missing or mixed binding, validates required numeric fields, writes
the sorted checks and top-level decision with that binding, and exits nonzero
on a failed gate.

## Installed release smoke

Install the exact candidate wheels into an empty environment and run from
outside the checkout. The path config must select the downloaded public-data
root before Python imports `s2and`:

The smoke command uses an 8 GiB budget. The common count index alone exceeds the previous 2 GB budget.

```powershell
$DistRoot = "$RunRoot\stages\local-dist"
$PublicDataRoot = $Snapshot
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
