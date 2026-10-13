# Release data preparation

Use these commands for reproducible release inputs. Keep large data outside Git. Commit code, tests, and the release evidence record.

Preserve accepted raw name parts. The runtime owns name normalization. A second normalization pass can change compound first names.

Use a fresh output directory for each stage. These commands preserve source files. Check a small dataset before a full run.

## Benchmark and augmented inputs

The `benchmarks` command applies reviewed cluster corrections and preserves uncertain signature membership. The uncertainty sidecar excludes these signatures from supervised labels.

Seed 1111 reserves the outer test blocks for external evaluation. The production trainer splits the remaining blocks again. Seed 12455678 controls pair samples.

```powershell
uv run --no-sync --with ijson --with orjson python scripts/production/data/prepare_pairwise_inputs.py benchmarks `
  --source-root <accepted-benchmark-root> --fixed-pair-root s2and/data-backup `
  --correction-receipt <stage-receipt.json> --correction-root <reviewed-cluster-root> `
  --output <prepared-root> --seed 1111 --pair-seed 12455678 --n-jobs 1

uv run --no-sync --with ijson --with orjson python scripts/production/data/prepare_pairwise_inputs.py augmentation `
  --prepared-root <prepared-root> --legacy-root <historical-data-root> `
  --output <prepared-root>/augmented --seed 1111 --pair-seed 12455678 --n-jobs 1

uv run --no-sync --with ijson --with orjson python scripts/production/data/prepare_pairwise_inputs.py finalize `
  --root <prepared-root>/augmented --prepared-root <prepared-root> `
  --encode --models-root <local-pinned-model-root> --device cuda --batch-size 16 --n-jobs 1
```

The bounded augmentation defaults use 100,000 train candidates and 10,000 evaluation candidates per dataset. Six omission probabilities equal 0.5.

Reuse a cached translation only when its original source title exactly equals the current raw title. Do not request new translations.

Reuse current SPECTER2 vectors when the text stays unchanged. Encode only changed informative text and informative text without a source vector. Preserve absence for blank text.

The encoder requires local model files. Its optional dependencies load only with `--encode`. Use the model environment that contains Torch and adapters.

Set `TOKENIZERS_PARALLELISM=false` for the encoder. Set `CARGO_BUILD_JOBS=1` if that environment needs a native extension build.

## Global name counts

Use the full source corpus for the production index. A research query subset does not cover production names.

See [the count producer instructions](../counts/README.md). The Parquet producer requires DuckDB 1.5 or later and explicit workload bounds.

## Linker support and replay inputs

```powershell
uv run --no-sync python scripts/production/data/prepare_linker_support.py `
  --source-root <accepted-linker-support-root> --benchmark-labels-root <prepared-root>/benchmark_source `
  --review-handoff <review-directory> --review-units <review-units.json> `
  --unknown-filename restored_unknown_signature_ids_20261009.json `
  --output-root <prepared-root>/linker_support --report-path <separate-report.json>

uv run --no-sync python scripts/production/data/prepare_linker_replay.py `
  --raw-root <original-replay-raw-root> --embeddings-root <original-replay-vector-root> `
  --benchmark-root <prepared-root>/benchmark_source --output-root <prepared-root>/replay_source `
  --run-full --n-jobs 1
```

The review directory contains `SHA256SUMS`, `final_linker_review.jsonl`, and `orcid_error_query_signatures.json`.
The support command uses current gold labels and accepted ORCID review decisions. It excludes uncertain query views and conflicting lifecycle identities.

The replay command preserves signature identities and order. It replaces matched benchmark metadata and vectors with accepted current records. It preserves giant-block source records.

## Arrow exports and portable packages

```powershell
uv run --no-sync python scripts/convert_to_arrow.py benchmark `
  --source-root <prepared-root>/benchmark_source --output-root <prepared-root>/arrow/benchmark `
  --name-counts-index-root <prepared-root>/name_counts_global/name_counts_index --run-full --n-jobs 12

uv run --no-sync python scripts/convert_to_arrow.py linker-replay `
  --raw-root <prepared-root>/replay_source/raw --embeddings-root <prepared-root>/replay_source/embeddings `
  --output-root <prepared-root>/arrow/replay `
  --name-counts-index-root <prepared-root>/name_counts_global/name_counts_index --run-full --n-jobs 1
```

Copy each uncertainty sidecar beside its exported cluster file. Bind both files with hashes in the export evidence. The converter does not copy these masks.

Convert `populations/model_source` and `populations/evaluation_source` with the same benchmark command. Use output roots `arrow/release` and `arrow/evaluation`.

Convert augmented feature files once. Use a feature-only source directory. The converter registers the staging collection automatically.

```powershell
$TaskRoot = '<prepared-root>'
$TaskFeatures = Join-Path $TaskRoot 'augmented_feature_source/augmented'
New-Item -ItemType Directory -Path $TaskFeatures
foreach ($TaskFile in @('augmented_signatures.json', 'augmented_papers.json', 'augmented_specter2.pkl')) {
  Copy-Item -LiteralPath (Join-Path "$TaskRoot/augmented" $TaskFile) -Destination $TaskFeatures
}
uv run --no-sync python scripts/convert_to_arrow.py benchmark `
  --source-root "$TaskRoot/augmented_feature_source" --output-root "$TaskRoot/arrow/augmented_staging" `
  --datasets augmented --name-counts-index-root "$TaskRoot/name_counts_global/name_counts_index" --n-jobs 2
uv run --no-sync python -c 'import shutil, sys; from pathlib import Path; from scripts.convert_to_arrow import _upsert_root_manifest; p = Path(sys.argv[1]); roots = [p / "arrow" / mode for mode in ("release", "evaluation")]; [(shutil.copytree(p / "arrow/augmented_staging/augmented", r / "augmented"), _upsert_root_manifest(r, dataset_name="augmented", dataset_dir=r / "augmented")) for r in roots]' $TaskRoot
```

The final command copies identical feature bytes into both private collections. It also adds each augmented root-manifest entry.
All three Arrow collections share the same parent directory. Their relative count-index paths therefore remain valid.

```powershell
uv run --no-sync python scripts/production/data/package_pairwise_inputs.py `
  --prepared-root <prepared-root> --output-root <portable-pairwise-root>

uv run --no-sync python scripts/production/model/linker_source_bundle.py `
  --source-root <prepared-root>/linker_support `
  --benchmark-arrow-root <prepared-root>/arrow/benchmark --replay-arrow-root <prepared-root>/arrow/replay `
  --name-counts-index <prepared-root>/name_counts_global/name_counts_index --model-plan <versioned-model-plan.json> `
  --output-source-bundle <portable-linker-source-root> --output-data-root <portable-public-data-root>
```

The private package contains source snapshots, relative paths, hashes, both Arrow collections, and one shared count index. It verifies source identity and native loaders.

The public assembler verifies lifecycle source rows and the release root. A data-only assembly plan establishes the release version. It does not authorize production model fits.

Keep total concurrent CPU use within the approved limit. Large INSPIRE exports also require a memory limit. Retain logs, process IDs, and final output checks.

See [the October 12 release record](../../../docs/release_evidence/data_preparation_20261012.md) for this run's inputs, counts, outputs, and open items.
