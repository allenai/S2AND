# Data and Models

S2AND has one common artifact set and one corrected research snapshot.
The research snapshot contains the final output of all accepted additions and corrections.
Use that snapshot as the input to the next correction cycle.
Historical correction archives are not required.

## Choose the required data

| User | Required content | Distribution |
| --- | --- | --- |
| Inference | Complete production model and binary name-count index | Versioned Git LFS assets |
| Research | Common assets plus corrected records, labels, embeddings, exact splits, and incremental tasks | S3 research snapshot |
| Release owner | Research snapshot plus maintained preparation and training scripts | Same snapshot and this repository |

Small canonical name tuples and ORCID prefix counts remain in normal Git and the Python wheel.
Research data does not contain a second count index or production model.
The complete model contains both pairwise boosters, calibrated EPS, the incremental linker, and required prediction fixtures.
See [the inference contract](production_inference.md#complete-model-bundles).

## Dataset download

The current common count index is available from PR 80's `canonical-v2-migration` branch.
The complete model and research S3 release remain unpublished.
The release notes will identify the common Git ref, model path, research S3 prefix, and manifest hashes.
Do not use an older public Arrow prefix with this runtime.
Older manifests lack the required public format `1` contract.

Common repository paths for release `1.3` are:

- `artifacts/common/v1.3/name_counts_index`
- `artifacts/common/v1.3/production_model_v1.3`

The complete model path receives the accepted model after its fit and evaluation.
Fetch the common assets at the Git ref declared by the release notes:

```bash
git lfs pull --include="artifacts/common/v1.3/**"
```

Use `artifacts/common/v1.3` as the local root when the research files share that root.
The S3 download then adds research files beside the existing index without another index copy.
Exclude `name_counts_index/*` and `production_model_*/**` from the research upload.

Use one local root for the common count index and research snapshot:

```text
release-root/
  name_counts_index/       # Common Git LFS asset; one physical copy
  production_model_vX.Y/   # Common Git LFS asset
  manifest.json           # Research public-data manifest
  corrected_sources/      # Complete best-known records and labels
  corrected_replay/       # Available giant replay records and vectors
  pairwise/
    inputs.json
    model_plan.json
    sources/              # Exact fit and evaluation source populations
    arrow/                # Fit and evaluation Arrow projections
  benchmarks/             # Full benchmark Arrow projections
  replay_manifest.json
  bundle.json             # Incremental source configuration
  labels/
  components/
  splits/
  datasets/               # Incremental replay Arrow projections
```

The S3 snapshot contains the research files in this layout.
Fetch its matching common index separately into `release-root/name_counts_index`.
Inference users do not need `corrected_sources`, benchmark labels, or incremental training tasks.
Dataset manifests use portable paths to the one common index.

The release publisher must exclude common assets from the S3 research upload.
A standalone local export can contain its own index for transfer.
Such an export is not the canonical combined publication.

## Corrected source records

`corrected_sources` contains the final records after additions and corrections, not the earlier inputs or correction history.
It preserves signature identities, record order, full paper text, current labels, uncertainty masks, embeddings, and required producer sidecars.
Full abstracts permit later text corrections and SPECTER2 updates.
Runtime Arrow paper tables retain abstract presence rather than full abstract text.
They therefore do not replace the corrected source records.

Benchmark masters preserve full text where it exists.
The seven current giant replay sources supply abstract presence, not full abstract text.
The snapshot preserves their available current records and vectors.
New abstract-based vectors for those giants require another source of full abstracts.

Benchmark cluster gold remains in `clusters.json` files.
Pair-labeled datasets use explicit pair files.
Unknown signatures remain available for inference but do not supply supervised labels.
Exact pair lists and split identities preserve the experiment population; a random seed alone is insufficient.

The snapshot also contains augmented records and their explicit pair labels.
Augmented source identities link each derived record to its corrected source signature.
Maintain those identities when the source dataset changes.

## Incremental tasks

The incremental dataset includes query identities, query views, seed-profile memberships, candidate memberships, labels, and exact split assignments.
It distinguishes full-name queries from initial-only queries.
Its binary target describes a safe attachment action.
An abstention label can indicate insufficient evidence; it does not necessarily indicate different authors.

Use the declared fit, calibration, and test assignments.
Keep inherited regression tests distinct from independent release evaluation.
Exposure annotations describe known pairwise, EPS, and paper overlap.
They do not prove independence from every past model or dataset.

## File formats and compatibility

Runtime records and vectors use Arrow IPC.
Incremental support uses Parquet, JSON, and CSV where those formats form the current contract.
Cluster labels remain JSON; fixed pairs and some split assignments remain CSV.
The snapshot is not an all-IPC dataset.
Precomputed features are optional derived caches, not the corrected source of truth.

Dataset manifests require `kind: "s2and_arrow_dataset"`, `format_version: 1`, portable `paths`, and a checksum inventory.
Name-count manifests require `kind: "s2and_name_counts"` and `format_version: 1`.
Dataset and index compatibility depends on their formats and schemas, not an exact Python package version.
Models additionally bind feature meanings, ordered feature names, and exact required artifact hashes.
Python and Rust packages must still have compatible internal interfaces; the current install requires matched package versions.

Do not edit an open Arrow root.
Publish changed records at a new path.
Build Arrow tables and their batch indexes together.
See [the public format specification](rust/arrow_dataset_spec.md) for table and manifest contracts.

## Update rules

| Change | Required update |
| --- | --- |
| Correct records or labels; add or remove a dataset | Publish a new corrected snapshot and rebuild affected projections and derived inputs |
| Change paper text | Refresh affected embeddings with the declared encoder recipe |
| Change feature calculations | Recompute affected feature tables and retrain affected models |
| Change a pairwise booster | Recompute incremental features and retrain the incremental model |
| Change EPS alone | Re-evaluate affected partitions; retain unchanged incremental features |
| Change profile membership or query evidence | Recheck affected safe-action labels and recompute affected block features |
| Change common assets | Publish changed assets once and bind affected models to their hashes |

Added components can change retrieval statistics for other queries within the same block.
A feature change alone does not require manual relabels if the label evidence and task scope remain valid.
Each release identifies its exact snapshot, feature contract, model, and common assets.

## Configuring `s2and/data/path_config.json`

Classic `ANDData` defaults to `<main_data_dir>/name_counts_index`.
`s2and/data/path_config.json`, or the file named by `S2AND_PATH_CONFIG`, sets `main_data_dir`.
Arrow routes and production scripts take explicit paths.

```json
{
  "main_data_dir": "/absolute/path/to/release-root"
}
```

Set `main_data_dir` to the directory that contains the common index.
The checked-in placeholder falls back to the package data directory with a warning.
Other values must name an existing directory.

See [the production commands](../scripts/production/README.md) for snapshot preparation and model fits.
