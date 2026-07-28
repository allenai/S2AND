# Production commands

Status date: 2026-07-26

This is a command reference. The dependency-ordered v1.3 sequence is
[../../docs/release.md](../../docs/release.md).

## Pairwise model

Start a fresh run directory containing only the reviewed `release.json`.
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
stages/
reports/
final/
```

This is the smallest random-split `release.json` shape. Paths are resolved
relative to the run directory. The numbers are examples, not reviewed release
values:

```json
{
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
    "baselines": {
      "cluster_signature_weighted_b3_f1": 0.804,
      "pairwise_aggregate": {"auroc": 0.9005, "macro_f1": 0.804},
      "pairwise_datasets": {
        "random": {"auroc": 0.9005, "macro_f1": 0.804}
      },
      "predict_seconds_p50": 10.0
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

Every object has an exact key set; extra keys fail. A random-split model
dataset uses `signatures`, `papers`, `specter_embeddings`, and `clusters`, and
has same-named pairwise and cluster evaluation datasets. A fixed-pair model
dataset replaces `clusters` with `train_pairs` and `val_pairs` and has a
same-named pairwise evaluation dataset. Model and evaluation identities must
be disjoint.

`model_plan.json` has training and validation inputs plus EPS policy, but no
held-out path. `evaluation_plan.json` has held-out inputs, gates, baselines,
and the reviewed performance Arrow root and workload. Both contain resolved
input identities; changing an input starts a new run.

Then run the release trainer:

```powershell
uv run python scripts/production/model/train_pairwise.py `
  --production-version X.Y `
  --model-plan path/to/run/model_plan.json `
  --name-counts-index-root path/to/name_counts_index `
  --matrix-work-dir D:/local-unsynced/s2and-matrix-work `
  --output-dir path/to/run/stages/pairwise/production_model_vX.Y `
  --validation-pairs-size REVIEWED_VALIDATION_PAIRS_SIZE `
  --run-full
```

The output is a pairwise-only v5 bundle. It is loadable by training and
finalization code but is not a complete runtime production model.

The trainer has one full-release mode. It requires an external name-count
index, an empty local matrix workspace, finite validation metrics, and explicit
acknowledgement of the expensive run. It cannot resolve held-out test paths.

## EPS calibration

Run `release_pairwise.py calibrate-eps` against validation data using only the
frozen model plan:

```powershell
uv run python scripts/production/model/release_pairwise.py `
  calibrate-eps `
  --source-bundle path/to/run/stages/pairwise/production_model_vX.Y `
  --model-plan path/to/run/model_plan.json `
  --output-bundle path/to/run/stages/calibrated/production_model_vX.Y `
  --output-report path/to/run/reports/eps_calibration_report.json `
  --name-counts-index-root path/to/name_counts_index
```

The EPS grid and floors come from the plan; objective, aggregation, and
smallest-EPS tie-break are fixed in code. EPS calibration does not fit the
release linker. An EPS-only change does not invalidate linker features or
require a new linker fit. A pairwise-booster or feature-contract change does.

## Linker source assembly

`linker_source_bundle.py` copies reviewed inputs into a fresh linker source
bundle and public data root:

```powershell
uv run python scripts/production/model/linker_source_bundle.py `
  --source-root path/to/reviewed_linker_source `
  --benchmark-arrow-root path/to/benchmark_arrow `
  --replay-arrow-root path/to/replay_arrow `
  --output-source-bundle path/to/linker_source_bundle `
  --output-data-root path/to/public_data_root
```

Assembly refuses existing output directories, loads the assembled bundle, and
preflights the selected rows. There is no separate source-manifest validator or
digest inventory.

## Complete model and linker evaluation

After held-out populations and gates are frozen, train the linker exactly once
against the final calibrated pairwise bundle:

```powershell
uv run python scripts/production/model/train_linker_and_finalize.py `
  --source-bundle-root path/to/run/stages/linker_source_bundle `
  --target-json s2and/data/production_model_v1.21/reproducibility/incremental_linker_training_target.json `
  --pairwise-model-path path/to/run/stages/calibrated/production_model_vX.Y `
  --name-counts-index-root path/to/name_counts_index `
  --output-dir path/to/run/stages/linker_release `
  --n-jobs REVIEWED_N_JOBS `
  --total-ram-bytes REVIEWED_LINKER_RAM_BYTES
```

The retained 53-feature target has SHA-256
`6b2c47963b10f2187478483f406021df4b8d3e58eded4e83b3c205b54c9f78f3`.
Changing it is a new target-selection decision and requires a new fit.

The command materializes features, fits once, writes the complete v5 bundle,
reloads it, and then evaluates held-out linker data through the reloaded model.
For version `X.Y`, the complete model is
`stages/linker_release/production_model_vX.Y` and the retained report is
`stages/linker_release/linker_evaluation_report.json`. The linker retains its
runtime binding to the pairwise feature contract and both boosters; that check
prevents a silent wrong-answer model mismatch.

Runtime callers load only a complete bundle and always supply its path:

```python
from s2and.production_model import load_production_model

clusterer = load_production_model("/path/to/production_model_vX.Y")
```

There is no packaged or implicit default production model.

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

For subblocking, use the candidate-member parquet selected for the same dataset
by `assets.candidate_members.datasets` in the frozen linker-source
`bundle.json`. It must contain `signature_id` and
`candidate_component_key` columns and match the selected Arrow root.

Produce all five reports:

```powershell
uv run python scripts/production/model/release_pairwise.py evaluate-pairs `
  --model path/to/production_model_vX.Y `
  --evaluation-plan path/to/run/evaluation_plan.json `
  --name-counts-index-root path/to/name_counts_index `
  --output-report path/to/run/reports/pairwise_evaluation_report.json `
  --n-jobs REVIEWED_N_JOBS `
  --total-ram-bytes REVIEWED_TOTAL_RAM_BYTES

uv run python scripts/production/model/release_pairwise.py evaluate-clusters `
  --model path/to/production_model_vX.Y `
  --evaluation-plan path/to/run/evaluation_plan.json `
  --name-counts-index-root path/to/name_counts_index `
  --output-report path/to/run/reports/cluster_evaluation_report.json `
  --n-jobs REVIEWED_N_JOBS

uv run python scripts/verification/compare_graph_subblocking_arrow_quality.py `
  --arrow-root path/to/reviewed_arrow_root `
  --output-dir path/to/run/reports `
  --python-source arrow `
  --comparison-mode python-vs-rust `
  --component-members-parquet path/to/reviewed_components.parquet `
  --maximum-size REVIEWED_MAXIMUM_SIZE `
  --allow-full `
  --seed REVIEWED_SEED

uv run python scripts/verification/compare_full_predict_arrow_parity.py `
  --fixture-dir path/to/reviewed_parity_fixture `
  --output-dir path/to/fresh/parity_work `
  --output-json path/to/run/reports/parity_evaluation_report.json `
  --name-counts-index path/to/name_counts_index `
  --model-path path/to/production_model_vX.Y `
  --block-size REVIEWED_BLOCK_SIZE `
  --n-jobs REVIEWED_N_JOBS `
  --total-ram-bytes REVIEWED_TOTAL_RAM_BYTES

uv run --with psutil python scripts/verification/profile_promoted_incremental_arrow.py `
  --evaluation-plan path/to/run/evaluation_plan.json `
  --model-path path/to/production_model_vX.Y `
  --require-rust-release `
  --write-json path/to/run/reports/performance_evaluation_report.json `
  --full-run
```

The performance Arrow root and complete workload come from the frozen
evaluation plan. The other explicit paths and parameters are frozen before
held-out prediction.

Apply the gates directly:

```powershell
uv run python scripts/production/model/release_pairwise.py `
  evaluate-release `
  --evaluation-plan path/to/run/evaluation_plan.json `
  --report-dir path/to/run/reports `
  --output-report path/to/run/reports/evaluation_report.json
```

The command validates required numeric fields, writes the sorted checks and
top-level decision, and exits nonzero on a failed gate.

## Canonical benchmark names

Join the reviewed upstream names by signature ID before Arrow conversion:

```powershell
uv run python scripts/convert_to_arrow.py join-canonical-names `
  --signatures path/to/dataset_signatures.json `
  --canonical-names path/to/canonical_names.json `
  --output path/to/canonical_signatures.json
```

The canonical input is a JSON list with `signature_id`, `first`, `middle`, and
`last`. The command refuses an existing output, duplicate IDs, and any
missing/extra ID; it reports changed signatures and per-field differences.
One canonical JSON must contain exactly one dataset's signature IDs; a global
all-dataset file is rejected as having extras.

First stage the canonical benchmark source outside the run directory. Replace
the `REVIEWED_*` placeholders with the reviewed paths and resource value:

```powershell
$RunRoot = 'D:\s2and-v1.3-YYYYMMDD-attempt-N'
$ReviewedInputWork = 'D:\s2and-v1.3-reviewed-inputs'
$CanonicalNamesRoot = 'REVIEWED_CANONICAL_NAMES_ROOT'
$ReviewedBenchmarkRoot = 'REVIEWED_BENCHMARK_ROOT'
$ReviewedReplayRawRoot = 'REVIEWED_REPLAY_RAW_ROOT'
$ReviewedReplayEmbeddingsRoot = 'REVIEWED_REPLAY_EMBEDDINGS_ROOT'
$NameCountsIndex = 'REVIEWED_NAME_COUNTS_INDEX'
$NJobs = REVIEWED_N_JOBS
$BenchmarkDatasets = @(
  'aminer','arnetminer','inspire','kisti','medline','pubmed','qian','zbmath'
)
$CanonicalBenchmarkRoot = "$ReviewedInputWork\benchmark_source_canonical"

New-Item -ItemType Directory -Path $CanonicalBenchmarkRoot | Out-Null
foreach ($Dataset in $BenchmarkDatasets) {
  $RawDataset = Join-Path $ReviewedBenchmarkRoot $Dataset
  $StagedDataset = Join-Path $CanonicalBenchmarkRoot $Dataset
  New-Item -ItemType Directory -Path $StagedDataset | Out-Null
  Get-ChildItem -LiteralPath $RawDataset -File |
    Where-Object { $_.Name -notin @("${Dataset}_signatures.json", 'signatures.json') } |
    Copy-Item -Destination $StagedDataset
  $RawSignatures = Join-Path $RawDataset "${Dataset}_signatures.json"
  if (-not (Test-Path -LiteralPath $RawSignatures)) {
    $RawSignatures = Join-Path $RawDataset 'signatures.json'
  }
  uv run python scripts/convert_to_arrow.py join-canonical-names `
    --signatures $RawSignatures `
    --canonical-names (Join-Path $CanonicalNamesRoot "$Dataset.json") `
    --output (Join-Path $StagedDataset "${Dataset}_signatures.json")
}
```

These conversion commands expect the reviewed inputs and the `stages/`
directory established by
[Stage 2 of the release runbook](../../docs/release.md#stage-2-build-training-and-evaluation-data):

```powershell
$ReplayDatasets = @(
  'a_khan','a_silva','arnetminer','h_wang','inspire','j_smith','kisti',
  'pubmed','qian','s_gupta','s_lee','s_park','zbmath'
)
$BenchmarkArrowRoot = "$RunRoot\stages\benchmark_arrow"
$ReplayArrowRoot = "$RunRoot\stages\replay_arrow"
New-Item -ItemType Directory -Path $BenchmarkArrowRoot | Out-Null
Copy-Item -LiteralPath $NameCountsIndex `
  -Destination "$BenchmarkArrowRoot\name_counts_index" -Recurse
uv run python scripts/convert_to_arrow.py benchmark `
  --source-root $CanonicalBenchmarkRoot `
  --output-root $BenchmarkArrowRoot `
  --datasets $BenchmarkDatasets `
  --name-counts-index-root $BenchmarkArrowRoot `
  --n-jobs $NJobs

New-Item -ItemType Directory -Path $ReplayArrowRoot | Out-Null
Copy-Item -LiteralPath $NameCountsIndex `
  -Destination "$ReplayArrowRoot\name_counts_index" -Recurse
uv run python scripts/convert_to_arrow.py linker-replay `
  --raw-root $ReviewedReplayRawRoot `
  --embeddings-root $ReviewedReplayEmbeddingsRoot `
  --output-root $ReplayArrowRoot `
  --datasets $ReplayDatasets `
  --name-counts-index-root $ReplayArrowRoot `
  --n-jobs $NJobs
```

The commands require fresh output paths for the reviewed full set. The
benchmark source is
`<root>/<dataset>/<dataset>_signatures.json` plus its other benchmark files.
Replay source is `<raw>/<dataset>/{signatures.json,papers.json}` plus
`<embeddings>/<dataset>/specter2.pkl`. Copy the final index into each Arrow root
first: these converter commands take the parent containing
`name_counts_index`, while model commands take the index directory itself.

Run the bounded `ANDData -> Arrow -> Rust` feature-parity fixture before this
full export.

## Linker source and data roots

Stage only the reviewed support files—`bundle.json`, `components/`, `labels/`,
and `splits/`—in a fresh support root. Do not include a legacy `datasets/`
directory because it would overwrite the fresh replay Arrow data. Install the
reviewed assignments and summary at the paths declared by both split
references in `bundle.json`:

```text
splits/combined_query_split_assignments_base_group_seed13.csv
splits/summary_base_group_seed13.json
```

Their reviewed SHA-256 values are respectively:

```text
b67b7ba7a5258b99d71f624ae12a2b5c6f938ba207215d1b7ebf63b791eadc64
51cc222218f1ac3906bfcae3d8c1e0f070dcc6c0eb815cd5aaea6908d9b40106
```

Verify those hashes, then assemble and preflight once:

```powershell
uv run python scripts/production/model/linker_source_bundle.py `
  --source-root "$RunRoot\stages\linker_support" `
  --benchmark-arrow-root "$RunRoot\stages\benchmark_arrow" `
  --replay-arrow-root "$RunRoot\stages\replay_arrow" `
  --output-source-bundle "$RunRoot\stages\linker_source_bundle" `
  --output-data-root "$RunRoot\stages\public_data_root"
```

The command refuses existing outputs, loads all referenced data, checks
base-group split disjointness, and reports selected rows. A failure starts a
fresh attempt; there is no repair or compatibility mode.

## Count artifacts

Regenerate the canonical tuple text directly from its reviewed source:

```powershell
uv run python scripts/production/generate_canonical_name_tuples.py `
  --source s2and/data/s2and_unnormalized_filtered_name_tuples.txt `
  --output path/to/fresh/s2and_name_tuples_canonical.txt
```

The count producers are module entry points:

```powershell
uv run python -m scripts.production.counts.generate_name_counts --help
uv run python -m scripts.production.counts.generate_orcid_name_prefix_counts --help
```

Full runs consume reviewed CSV exports and follow the
[release operating rules](../../docs/release.md#operating-rules). The
repository has no warehouse client or credentials.

Export name counts with this reviewed query, setting the limit to
`max_source_rows + 1`:

```sql
select coalesce(first_name, '') as first_name,
       coalesce(last_name, '') as last_name,
       count(*) as count
from content.authors
group by coalesce(first_name, ''), coalesce(last_name, '')
order by first_name, last_name
limit <MAX_SOURCE_ROWS_PLUS_ONE>;
```

Save exactly `first_name,last_name,count`, then run:

```powershell
uv run python -m scripts.production.counts.generate_name_counts `
  --input-csv path/to/reviewed_name_counts.csv `
  --guardrails-json path/to/name_count_guardrails.json `
  --output-dir path/to/fresh/count_publication
```

The native index is written at
`path/to/fresh/count_publication/name_counts_index`.
The guardrail file has exactly four positive integer fields. This is a shape
example, not a reviewed full-run bound:

```json
{
  "min_source_rows": 1,
  "max_source_rows": 100,
  "min_keys_per_mapping": 1,
  "max_keys_per_mapping": 100
}
```

Each minimum must be no greater than its matching maximum.

Export ORCID names with the following query and the same limit convention:

```sql
with source_rows as (
  select pae.corpus_paper_id, pae.orcid as raw_orcid, pae.position,
         pae.first_name, pa.middle,
         upper(regexp_replace(
           regexp_substr(
             coalesce(pae.orcid, ''),
             '(?<![0-9x])[0-9]{4}[-‐‑‒–—−﹘﹣－]?[0-9]{4}[-‐‑‒–—−﹘﹣－]?[0-9]{4}[-‐‑‒–—−﹘﹣－]?[0-9]{3}[0-9x](?![0-9x])',
             1, 1, 'ip'
           ),
           '[-‐‑‒–—−﹘﹣－]'
         )) as canonical_orcid_compact
  from content_ext.paper_authors_orcids pae
  join content_ext.papers p
    on pae.corpus_paper_id = p.corpus_paper_id
  join content_ext.paper_authors pa
    on pae.corpus_paper_id = pa.corpus_paper_id
   and pae.position = pa.position + 1
   and lower(pae.last_name) = lower(pa.last)
  join content_ext.authors au
    on pa.corpus_author_id = au.corpus_author_id
  where pae.source = 'Crossref'
    and nullif(trim(coalesce(pae.first_name, '')), '') is not null
)
select raw_orcid,
       case when canonical_orcid_compact = '' then null
            else substring(canonical_orcid_compact, 1, 4) || '-'
              || substring(canonical_orcid_compact, 5, 4) || '-'
              || substring(canonical_orcid_compact, 9, 4) || '-'
              || substring(canonical_orcid_compact, 13, 4)
       end as orcid,
       first_name, middle
from source_rows
order by orcid nulls last, first_name, middle, corpus_paper_id, position
limit <MAX_SOURCE_ROWS_PLUS_ONE>;
```

Save exactly `raw_orcid,orcid,first_name,middle`, preserving query order, then
run:

```powershell
uv run python -m scripts.production.counts.generate_orcid_name_prefix_counts `
  --input-csv path/to/reviewed_orcid_names.csv `
  --guardrails-json path/to/orcid_guardrails.json `
  --name-tuples-path path/to/fresh/s2and_name_tuples_canonical.txt `
  --output-dir path/to/fresh/orcid_counts
```

The ORCID guardrail file has exactly five positive integer fields. Again,
these values demonstrate the shape only:

```json
{
  "min_source_rows": 1,
  "max_source_rows": 100,
  "max_names_per_orcid": 100,
  "max_pair_keys": 1000000,
  "min_orcid_pair_keys": 1
}
```

It requires `min_source_rows <= max_source_rows`,
`max_names_per_orcid >= 2`, and
`min_orcid_pair_keys <= max_pair_keys`.

The large name-count index remains a manifest-backed native binary directory.
Its file layout, normalization contract, and model binding prevent using
incompatible count features at runtime. Its v3 manifest contains only the
schema and normalization versions plus the four binary file facts.

Canonical tuple generation writes one text file. ORCID generation writes one
JSON file plus a minimal name-tuple dependency manifest. Both are loaded and
semantically validated directly.

Copy the reviewed outputs to these exact package paths:

| Output | Package path |
|---|---|
| Fresh canonical tuple | `s2and/data/s2and_name_tuples_canonical.txt` |
| ORCID JSON | `s2and/data/first_k_letter_counts_from_orcid.json` |
| ORCID manifest | `s2and/data/first_k_letter_counts_from_orcid.manifest.json` |

## Distribution and Arrow checks

Build one local-platform candidate of each distribution and verify the Python
wheel/sdist contents. Use a fresh external `$DistRoot` for the Stage 1 asset
check and `$RunRoot\stages\local-dist` for the final candidate:

```powershell
$PythonDist = "$DistRoot\python"
$RustDist = "$DistRoot\rust"

uv run --no-project python scripts/sync_version.py --check
uv build --sdist --wheel --out-dir $PythonDist --clear
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

For a bounded local Arrow layout check:

```powershell
uv run python scripts/verification/validate_local_arrow_release.py `
  --release-root path/to/data_root
```

Use `scripts/convert_to_arrow.py validate --dataset-dir ...` when a deep table
scan is required.

## Installed release smoke

Install the exact candidate wheels into an empty environment and run from
outside the checkout. The path config must select the downloaded public-data
root before Python imports `s2and`:

```powershell
$DistRoot = "$RunRoot\stages\local-dist"
$PublicDataRoot = "$RunRoot\stages\public_data_root"
$SmokeVenv = "$RunRoot\stages\installed-smoke-venv"
$SmokePython = "$SmokeVenv\Scripts\python.exe"
$PathConfig = "$RunRoot\stages\installed-smoke-path-config.json"
$Repo = (Resolve-Path .).Path

@{
  main_data_dir = (Resolve-Path $PublicDataRoot).Path
  internal_data_dir = ''
} | ConvertTo-Json | Set-Content -LiteralPath $PathConfig -Encoding utf8

uv venv --python 3.11 $SmokeVenv
uv pip install --python $SmokePython `
  (Get-ChildItem "$DistRoot\rust\*.whl").FullName `
  (Get-ChildItem "$DistRoot\python\*.whl").FullName

$env:S2AND_PATH_CONFIG = $PathConfig
Push-Location $RunRoot
uv run --no-project --python $SmokePython python `
  "$Repo\scripts\verification\smoke_installed_incremental_arrow.py" `
  release-candidate `
  --model-dir "$RunRoot\stages\linker_release\production_model_v1.3" `
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
uv venv --python 3.11 $PublicSmokeVenv
uv pip install --python $PublicSmokePython `
  --refresh `
  --default-index https://pypi.org/simple `
  "s2and==$Version" `
  "s2and-rust==$Version"
```

Use `$PublicSmokePython` in the `uv run --no-project --python ...` invocation
above; keep the same path config, model, data root, dataset, and signature IDs.
