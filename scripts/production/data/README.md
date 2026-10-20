# Corrected research snapshot

Use the final output of all accepted additions and corrections as the master dataset.
The next correction cycle starts from that output.
Old review archives and pre-correction records are not required inputs.
Keep full paper text, current labels, uncertainty masks, vectors, and exact identities in the master.
Keep model outputs outside the immutable dataset root.

## Package the current inputs

Reuse accepted augmentation when source identities and omission policy remain valid.
Release `1.3` excludes patent data.
Pairwise inputs contain eight sources: seven non-patent benchmarks and the existing non-patent augmented dataset.
Incremental inputs contain the 13 inherited non-patent datasets.
The selected local snapshot path is `D:/local-unsynced/s2and_ca_mainline_20261017/snapshot`.
The snapshot passes relocation checks.

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

The producer copies complete corrected sources and exact fit/evaluation inputs.
The paired replay options preserve accepted raw replay records and vectors.
They are optional, but the complete release snapshot supplies both.
It writes `pairwise/inputs.json` and an internal `pairwise/model_plan.json` with relative paths.
Publish and train from the bundles that `write_release_bundles.py` writes from this snapshot.
See [release bundles](../README.md#release-bundles).
It includes fit/evaluation Arrow projections, benchmark projections, and the complete incremental source dataset.
All projections reference one physical count index at the snapshot root.
It validates hashes, boundaries, source identities, and native loaders.
A changed package requires a new output path.

The default EPS grid runs from `0.00` to `1.00` at steps of `0.01`.
Both default minimum F1 floors equal zero.
Set `--eps-grid`, `--minimum-dataset-f1`, and `--minimum-signature-weighted-f1` for a different declared policy.
These options do not authorize a model fit.

Move a small real package to another directory before the full publication package.
Verify the model plan and native Arrow datasets from the new directory.
Verify incremental labels, membership tables, and split assignments there.
The package does not use directory junctions or external support paths.

## Prepare the next data update

Copy the previous `corrected_sources` into a fresh work directory.
Apply new record and label corrections there.
Preserve raw name parts; the runtime owns canonical normalization.
A second normalization pass can change compound first names.
Preserve record order and exact split identities unless the new release deliberately replaces them.

Prepare benchmark populations from the corrected records:

```powershell
uv run --no-sync --with ijson --with orjson python scripts/production/data/prepare_pairwise_inputs.py benchmarks `
  --source-root <corrected-source-root> `
  --fixed-pair-root <current-fixed-pair-root> `
  --output <prepared-root> `
  --datasets <declared-dataset-names> `
  --seed 1111 --pair-seed 12455678 --n-jobs 1
```

The optional correction receipt and correction root apply only when a separate new correction overlay exists.
The producer does not require `<dataset>_added_signatures.json`.
It preserves that sidecar when supplied.
Current cluster labels and uncertainty masks remain authoritative.
Unknown signatures remain available for inference but do not supply supervised labels.

Seed 1111 reserves the declared outer test blocks.
The production trainer splits the remaining development blocks again.
Seed 12455678 controls pair samples.
Preserve exact populations and pair lists because source order also affects selection.
A block split does not guarantee whole-paper independence.

Add `--model-exclusions <reserved-scope.json>` for fresh incremental dev and test populations.
The producer reserves declared blocks, authors, and query papers from model development inputs.
It preserves the exact outer test population.

## Regenerate augmentation only when required

Use the accepted augmented records unchanged when their inputs and policy remain valid.
A source correction or new augmentation policy can require new derived records and pair labels.
Generate omission variants from the current corrected records:

```powershell
uv run --no-sync --with ijson --with orjson python scripts/production/data/prepare_pairwise_inputs.py augmentation `
  --prepared-root <prepared-root> --output <prepared-root>/augmented `
  --datasets <declared-dataset-names> `
  --seed 1111 --pair-seed 12455678 --n-jobs 1
```

`--legacy-root` is optional and supplies existing translation caches.
It is not a required historical source dataset.
Reuse a translation only when its original source title equals the current source title.
Do not request new translations as part of this process.

The bounded defaults use 100,000 train candidates and 10,000 evaluation candidates per dataset.
The six omission probabilities equal `0.5`.
The source identities use `<dataset>___<signature_id>`.
Training pairs must satisfy the actual source training membership rules.
Evaluation pairs must remain outside the declared development population.

## Embedding updates

Reuse SPECTER2 vectors when their input text remains unchanged.
Encode changed informative text and informative text without a source vector.
Keep absent vectors for blank text.
Use the declared local encoder model, adapter, tokenizer, pooling, and truncation configuration.

```powershell
uv run --no-sync --with ijson --with orjson python scripts/production/data/prepare_pairwise_inputs.py finalize `
  --root <prepared-root>/augmented --prepared-root <prepared-root> `
  --datasets <declared-dataset-names> `
  --encode --models-root <local-pinned-model-root> --device cuda --batch-size 16 --n-jobs 1
```

The encoder dependencies load only with `--encode`.
Use the environment that contains Torch and the required adapters.
Set `TOKENIZERS_PARALLELISM=false` before that process starts.
Do not describe runtime Arrow tables as a replacement for the full corrected paper text.
The seven current giant replay sources contain abstract presence instead of full abstracts.
Preserve their available records and cached vectors.
A new abstract-based encoding for those giants requires another source of full abstract text.

## Incremental data updates

Initial-only rows use query and paper copies; original master and profile records retain full names.
The trainer uses origin maps to exclude original queries from candidate profiles.

Reuse the corrected incremental source tables, labels, and assignments when their task evidence remains valid.
Profile-member changes and query evidence changes require affected label checks.
A changed pairwise model requires fresh incremental feature computation.
The production finalizer performs that computation from the corrected source dataset.
A feature change alone does not automatically require new manual labels.

The dataset contains full-name and initial-only task views.
Keep their judgments scoped to the supplied evidence.
An abstention means the attachment lacks sufficient support under the safe-action target.
Do not convert every abstention into a different-author label.

Retain exposure annotations for pairwise, EPS, and paper overlap.
Keep regression-only test views distinct from active release evaluation assignments.
Adding components can alter retrieval features for other queries in the same block.
Recompute the affected block's features.

## Arrow projections and publication

Use `scripts/convert_to_arrow.py` for changed record/vector projections.
Build tables and batch indexes together.
Carry current cluster labels and uncertainty masks into the prepared projections.
`package_pairwise_inputs.py` packages the exact fit/evaluation populations before the combined snapshot step.
Use `linker_source_bundle.py` only when the accepted support or replay Arrow roots change.

The snapshot uses Arrow IPC, Parquet, JSON, and CSV according to the current contracts.
It is not an all-IPC dataset.
Precomputed features are derived caches and do not replace the corrected master.

Publish the common index and complete model through Git LFS.
Publish the `inference`, `dataset`, and `source` bundles that `write_release_bundles.py` writes.
Do not publish the snapshot itself.
Verify one local root that combines `inference` and `dataset`.

See [the production commands](../README.md) for the model fit and EPS calibration.
See [the release process](../../../docs/release.md) for acceptance and publication.
