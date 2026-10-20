# Data and Models

S2AND publishes each release as three bundles.
Each bundle is one S3 folder.
Sync the bundles that you need into one local folder; each bundle adds files beside the others.
The `source` bundle contains the final output of all accepted additions and corrections.
Use it as the input to the next correction cycle.
Historical correction archives are not required.

## Choose the required data

| Bundle | Use | Contents | Release `1.3` size |
| --- | --- | --- | --- |
| `inference/` | Run the production model | Complete production model and binary name-count index | 2.4 GB |
| `dataset/` | Train, tune, and evaluate models | Benchmark Arrow data with labels, held-out test blocks and pairs, augmented pairs, the model plan, and incremental linker tasks | 9.1 GB |
| `source/` | Correct records and prepare the next release | Editable JSON records with full paper text, SPECTER2 pickles, and giant-block replay records | 25.6 GB |

Inference needs `inference/` only.
Training and evaluation need `inference/` and `dataset/`.
Release preparation needs all three bundles.
The model and index also remain Git LFS assets under `artifacts/common/v1.3`.
Small canonical name tuples and ORCID prefix counts remain in normal Git and the Python wheel.
The complete model contains both pairwise boosters, the reviewed EPS, the incremental linker, and required prediction fixtures.
See [the inference contract](production_inference.md#complete-model-bundles).

## Download

> **Update at v1.3 release:** The S3 bundles are not published yet.
> Replace this note with the verified bucket path, the release Git ref, and the manifest hashes.

```bash
aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and/1.3/inference/ s2and-1.3/
aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and/1.3/dataset/ s2and-1.3/
# Release preparation only:
aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and/1.3/source/ s2and-1.3/
```

The [reproduction guide](reproduce.md) runs the full training process on the synced folder.
Release `1.3` excludes patent data.
Its pairwise data contains eight sources: seven non-patent benchmarks and the existing non-patent augmented dataset.
Its incremental data contains the 13 inherited non-patent datasets.
Do not use an older public Arrow prefix with this runtime.
Older manifests lack the required public format `1` contract.

The production count authority is A: one contribution per source author-table row.
The Athena export preserves all raw first/last groups, including null names.
The current repository normalizer creates the lookup keys after export.
The [count guide](../scripts/production/counts/README.md) defines the complete build command.
Only AI2 staff can build the index, because its input is an internal Semantic Scholar author-table export.
Other users use the released index and cannot regenerate it.
Each future refresh of the index is a staff step.

Git LFS supplies the same model and index:

```bash
git lfs pull --include="artifacts/common/v1.3/**"
```

To use that copy, sync `dataset/` into `artifacts/common/v1.3`.
Arrow manifests find the index at `name_counts_index/` in the same folder.

## Folder layout

```text
s2and-1.3/
  name_counts_index/          # inference: binary name-count index
  production_model_v1.3/      # inference: complete model
  manifest.json               # dataset: public-data manifest
  benchmarks/<source>/        # dataset: Arrow tables, <source>_clusters.json, unknown-signature list
  pairwise/
    model_plan.json           # dataset: training populations, held-out tests, EPS policy
    heldout/<source>/         # dataset: test_blocks.json and test_pairs.json
    augmented/                # dataset: augmented Arrow tables and fixed pairs
  replay_manifest.json        # dataset: incremental Arrow catalog
  bundle.json                 # dataset: incremental source configuration
  datasets/                   # dataset: incremental Arrow datasets
  labels/  components/  splits/   # dataset: incremental tasks
  evaluation_inputs/          # dataset: frozen identity-gold cases
  incremental_patent_tasks/   # dataset: initial-query source map
  corrected_sources/          # source: editable benchmark records
  corrected_replay/           # source: giant-block replay records and vectors
  augmented/                  # source: editable augmented records
```

The model plan names one complete benchmark Arrow dataset per source.
It also names that source's clusters, held-out blocks, frozen held-out pairs, and unknown-signature list.
The development population is the benchmark minus its held-out blocks.
The trainer selects pairs and computes features from the same Arrow data.
The published `1.3` data reproduces the `1.3` training and validation matrices exactly.
Dataset manifests use portable paths to the one common index.

The release owner writes the bundles from an internal snapshot with
[`write_release_bundles.py`](../scripts/production/README.md#release-bundles).
That snapshot also holds pre-cut pairwise copies, correction ledgers, and audit records.
They are not published.
[Stage 0 of the release process](release.md#stage-0-choose-the-input-snapshot) records the selected snapshot.
Patent records and research task packets remain outside the release inputs.

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

Benchmark cluster gold remains in `<source>_clusters.json` files.
Pair-labeled datasets use explicit pair files.
Unknown signatures remain available for inference but do not supply supervised labels.
Held-out blocks, fixed pair lists, and the recorded seed preserve the experiment population.

The bundles also contain augmented records and their explicit pair labels.
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

The `dataset` bundle supports giant-block EPS review on its seven giant Arrow datasets.
Their profile memberships, dev labels, and initial-query source map need no second S3 copy.
Calibration reference files and review outputs are derived sidecars.
Keep them distinct from untouched final test data.
See [giant-block EPS review](eps_calibration.md) for the commands and the `1.3` EPS decision.

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
