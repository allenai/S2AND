# Release Notes

## 1.0.0b14 and model 1.4 (2026-11-23)

- Model and data release `1.4` is published under `s3://ai2-s2-research-public/s2and/1.4/`
  (`inference/` and `dataset/`; no `source/` bundle). Load `production_model_v1.4` with
  `s2and` `1.0.0b14` or later. See [data](data.md).
- `1.4` retrains both pairwise heads and the incremental linker on corrected labels and
  November giant-block ORCID data, with a refreshed name-count index and ORCID counts.
  EPS is `0.38`; see [the EPS decisions](eps_calibration.md#recorded-release-decisions).
- Against `1.3.2`: benchmarks tie (overlap F1 0.9674 vs 0.9683); on six giant blocks scored on all
  gold-ORCID authors, F1 rises 0.67 points; on 27 held-out production blocks the two tie
  (pooled F1 0.9479 vs 0.9487).
- Every paper in the `1.4` data is in the 2026-11-17 S2AG snapshot. Abstract text appears only
  where sharing is permitted, and emails are hashed.

- Linker source preflight resolves calibration and test assignments with the trainer's selector
  and rejects missing eligible rows, mismatched `base_group_id` values, inconsistent eval-table
  paths, labels for missing replay datasets, and duplicate or conflicting candidates, before any copy.
  The fit, `evaluate_classic` and the EPS tools reject an assignment whose source key has no eval table.
- Linker source preflight checks each dataset within each label table: with ten or more marked
  `negative_singleton_orcid` views, at least 10% must have a positive candidate. Mark unreviewed
  views `unlabeled_singleton_orcid`.
- Giant EPS sweeps require `--batching-threshold` and record it. Applying a snapshot sweep requires
  `--expected-batching-threshold`, one snapshot, and a `--datasets` list that matches the snapshot's
  calibration assignments outside `s2and_eval`; it rejects curves recorded for another dataset.
  The EPS reference excludes the same final-test split as `evaluate_classic`.
- Corrected dataset packaging requires `--precision-f1-tolerance`. Calibration requires
  `--allow-zero-precision-f1-tolerance` when a close-pair bonus wins at zero tolerance, and saves
  the refused run's trials for the retry.
- Cap LightGBM trees at 256 leaves in pairwise search, supplied pairwise
  configurations, and classic linker training. This is the largest tree the
  native fast scorer supports.
- Refit both pairwise heads with `train_pairwise.py --fixed-parameters` and a
  JSON file containing `main` and `nameless` parameters. The trainer skips the
  search and rejects combining this option with `--incumbent-parameters`.
  The run's `--n-jobs` and the featurizer's monotone constraints take
  precedence over file values. The fit summary records the applied parameters.
- Select subblocking's Open Researcher and Contributor ID (ORCID) prefix counts
  from the feature contract hash. Prediction before export and after reload uses
  the same counts. Models without the hash use the packaged default.

## 1.0.0 (unreleased)

The package, production model, and public data use separate version numbers.
The final package release is `1.0.0`. The production data snapshot is `1.3`.

### Migrating Arrow callers from 0.51

Services that sent Arrow tables to S2AND 0.51 need these changes:

- When the final release ships, pin `s2and==1.0.0`. The `s2and` wheel includes
  the native extension. Drop any `s2and-rust` pin and the `rust` extra.
- The wheel no longer contains a model or a name-count pickle. Download the model and the
  name-count index from the `inference` S3 bundle (see [data](data.md)), and load the model with
  `load_production_model(path, n_jobs=N)`. `production_model_bundle_version` is now
  `production_model_release_version`, for example `"1.3"`.
- `predict_from_arrow_paths` and `predict_incremental_from_arrow_paths` are removed. Build batch
  indexes, open the tables with `ArrowDataset.open_trusted`, and call `predict_from_arrow` or
  `predict_incremental_from_arrow`.
- Seeds, exclusions, and altered profiles are prediction arguments, not path keys. 0.51 read the
  `cluster_seeds`, `cluster_seed_disallows`, and `altered_cluster_signatures` tables from the path
  mapping. `open_trusted` now rejects those keys and `profile_exclusions`.
- `altered_cluster_signatures=None` means that no profile is altered. 0.51 read a sidecar table
  instead.
- Incremental prediction returns a mapping; read `result["clusters"]`. `return_clusters_only` is
  removed.
- Omit `name_tuples`, or pass `None`, to use the packaged alias list.
- `FEATURE_BLOCK_SCHEMA_VERSION`, `NAME_COUNTS_INDEX_SCHEMA_VERSION`, `s2and.file_cache`, and
  `s2and.data.NAME_COUNTS_PATH` are removed. Check table schemas with
  `s2and.arrow_schema.validate_arrow_schema`.
- Papers may omit `predicted_language`, `is_reliable`, and `language_reliability`; S2AND then
  detects the language from the title. Present values must be consistent: a language for every
  reliability value, no empty language, and reliability `0.0` when `is_reliable` is false.

```python
from s2and.arrow_inputs import ArrowDataset
from s2and.incremental_linking.feature_block import (
    read_altered_cluster_signatures_arrow,
    read_cluster_seed_disallows_arrow,
    read_cluster_seeds_arrow,
    write_raw_arrow_batch_lookup_indexes,
)

# tables: {"signatures": path, "papers": path, "paper_authors": path, "specter": path}
paths, _ = write_raw_arrow_batch_lookup_indexes(tables, request_directory)
paths["name_counts_index"] = name_counts_index_directory
with ArrowDataset.open_trusted(paths) as dataset:
    result = clusterer.predict_incremental_from_arrow(
        block_signatures,
        dataset,
        cluster_seeds_require=read_cluster_seeds_arrow(seeds_path),
        cluster_seeds_disallow=read_cluster_seed_disallows_arrow(disallows_path),
        altered_cluster_signatures=read_altered_cluster_signatures_arrow(altered_path),
    )
clusters = result["clusters"]
```

Bulk prediction takes the same `cluster_seeds_require` and `cluster_seeds_disallow` arguments.

### Breaking changes

- Releases publish three Amazon Simple Storage Service (S3) bundles:
  `inference` (model and count index), `dataset` (Arrow training and evaluation
  data), and `source` (editable records).
  `scripts/production/data/write_release_bundles.py` writes them after checking
  that the Arrow data rebuilds the prepared populations. The
  [reproduction guide](reproduce.md) uses `inference` and `dataset`.
- Production pairwise training reads datasets and splits from `--model-plan`.
  A random-block dataset declares its benchmark Arrow manifest, clusters,
  held-out blocks, frozen held-out pairs, and optional unknown-signature and
  excluded-signature lists. A fixed-pair dataset declares its Arrow manifest and
  train, validation, and held-out pairs. Plans require Arrow papers and embeddings.
  The published `1.3` data reproduces the `1.3` training and validation matrices.
- `train_pairwise.py` and `train_linker_and_finalize.py` require `--n-jobs`.
- Arrow batch indexes use `S2ABI003` with a 32-byte header. Rebuild older indexes
  and regenerate their dataset manifests. Index validators and `overwrite=False`
  reuse check layout and size. `ArrowDataset.open_trusted(paths)` accepts
  caller-owned tables and indexes without a dataset manifest or checksum scan.
- Arrow dataset manifests use `kind: "s2and_arrow_dataset"`, public format `1`,
  portable `paths`, and a flat semantic-role content inventory. Open a manifest
  with `ArrowDataset.open(root)` to validate files and indexes and retain a
  reusable handle with native readers. Pass request-local seeds to prediction.
  The release validator requires the root and nested replay datasets to resolve
  `paths.name_counts_index` to the publication-root index.
- The dataset manifest records the name-count index path. The model's feature
  contract binds the index's manifest hash. New dataset manifests omit the index
  from the content inventory and the opener ignores older inventory entries
  for the index. `ArrowDataset.generation_id` depends on the opened index.
  S2AND `1.0.0b10` and earlier still require the index the dataset manifest names.
- Promoted stratified splits require `base_group_id`. Assign each base identity,
  including its masked views, to one split.
- ORCID prefix counts are packaged in two sets: the original `1.3.x` counts and
  refreshed `1.4` counts. Each set has a validated JSON file and a tuple-dependency
  manifest. Existing bundles select a set by the hash in the feature contract.
  New training contracts use the refreshed set, which excludes five additional
  reviewed mixed-identity ORCID groups. Producer provenance, schema, and metrics
  sidecars are removed.
- Name-count indexes use public format `1` and the flat
  `name_counts_index/{manifest.json,first.bin,last.bin,first_last.bin,last_first_initial.bin}`
  layout. The manifest stores the sizes and SHA-256 digests for four fixed roles.
  Rebuild older layouts and any model artifacts bound to their manifest digest.
- Pairwise training requires an external name-count index, a local matrix
  workspace, and explicit full-run acknowledgement. Linker training takes the
  final pairwise bundle, performs one fit, writes the complete bundle with the
  embedded replay target, reloads it, and evaluates it. See the
  [release runbook](release.md) for the prepared inputs and acceptance checks.
- The pairwise trainer drops `--production-version`, `--data-dir`, `--datasets`,
  `--include-augmented`, `--specter-suffix`, `--signatures-suffix`, `--chunk-size`,
  `--cluster-n-iter`, `--val-test-size`, `--use-cache`, and `--negative-one-for-nan`.
- Production pairwise preflight checks augmented and benchmark overlap in both
  directions. An augmented training member must belong to its source training
  blocks. Augmented evaluation cannot contain a benchmark development signature.
- Production conversion and evaluation use SPECTER2. Runtime conversion writes
  one embedding table under manifest key `specter`, usually `specter2.arrow`.
- Python and Rust share the `canonical_v2` name and feature behavior. Titles
  retain letters and digits, CLD2 uses plain-text mode, malformed email is
  treated as missing, and query-author features use canonical name fields.
  Incremental six-decimal values round ties to even. Discrete features match
  exactly and numeric feature parity uses a `1e-6` tolerance.
- `ANDData(preprocess=False)` leaves the per-signature
  `author_info_coauthors` and `author_info_coauthor_blocks` fields unset.
  Python featurization requires `preprocess=True`. Graph subblocking and
  evaluation can derive evidence from paper authors.
- Runtime conversion uses typed PyArrow tables instead of the intermediate
  `FeatureBlock` object graph. Producers use
  `write_raw_planner_arrow_from_anddata()` or `raw_planner_arrow_tables_from_anddata()`,
  and write table mappings with `write_raw_planner_arrow_tables()`.
- FastCluster condensed distance matrices use `float64`. Recompute any cached
  `float16` matrices passed as `val_dists_precomputed` or `dists`. Calibration
  uses the explicit validation split. Recompute matrices built for the previous
  calibration population.
- `scripts/convert_to_arrow.py --name-counts-index-root` takes the
  `name_counts_index` directory itself. The former parent directory argument
  now produces a missing-artifact error.
- `Signature.author_info_first_normalized_without_apostrophe` and
  `Signature.author_info_middle_normalized_without_apostrophe` are now
  `author_info_first_normalized` and `author_info_middle_normalized`. The
  canonical fields already delete apostrophes. `s2and.text.normalize_text`
  drops `special_case_apostrophes` and continues replacing apostrophes with
  spaces in venues, affiliations, and suffixes.
- `ANDData` groups by `author_info.block`. The `block_type` constructor
  argument, `Signature.author_info_given_block`, `get_original_blocks()`,
  and `get_s2_blocks()` are removed. Legacy
  `author_info.given_block` JSON is accepted as an ignored field. Preserve
  original benchmark partitions externally if you need them.

### Added

- Name-conflict repair merges clusters when at most two odd mentions explain all
  name conflicts, subject to name, ORCID, seed, and exclusion checks. It also
  handles joined-initials readings when either cluster spells the reading.
  Each side keeps at least five mentions and a full first name after the odd
  mentions are left out. Repair treats tied leading ORCIDs symmetrically and
  runs up to five rounds. The odd-mention pass compares the average distance,
  adjusted for any close-pair bonus, against
  `Clusterer.name_conflict_repair_max_distance` (default `0.2`). When
  the bonus carries a merge, at least 10% of pairs must be certainly close,
  the plain average must be at most `0.40`, and each side must keep at least
  10 mentions. Set the distance to `None` to disable repair. Bundle export
  requires the default. See [Name-conflict repair](production_inference.md#name-conflict-repair).
- Year-gap split cuts a cluster at an empty stretch of more than 50 years.
  It is on by default, requires a close-pair bonus, and runs independently of
  name-conflict repair. A certain close pair, a shared ORCID, or a shared
  required seed group can bridge the gap. The first word of a full first name
  can also bridge it when the original surname block has fewer than 11 mentions
  and any available last-name-plus-first-initial count is below 11. Undated
  mentions sharing a bridge move together. Years `1601` and `1800` count as
  undated and, without a bridge, preferentially follow the part spanning the
  recorded year. Set `Clusterer.year_gap_split = False` to disable it.
  Bundle export requires the default. Both `predict_eps_grid` and
  `predict_eps_grid_from_rust_featurizer` accept `small_block_signatures` so
  subblocked sweeps use the same surname-block information as production.
  See [Year-gap split](production_inference.md#year-gap-split).
- ORCID joins run only across subblocks of the same original Arrow block.
  Clusters must share a well-supported leading ORCID and agree on the dominant
  full first name without a hard conflict among supported name forms. An
  initials-only cluster can join when the ORCID covers at least 20% of its
  mentions and the other clusters agree on one full first name. The join
  respects seed and exclusion constraints and skips ORCID-suppressed requests.
  It is on by default. Set `Clusterer.orcid_join = False` to disable it.
  Bundle export requires the default. See [ORCID joins](production_inference.md#orcid-joins).
- The ORCID name waiver is on by default. It can restore the model distance
  between a mention with an ORCID and one without it. The name on the
  ORCID-bearing mention must conflict with other mentions of its ORCID. The
  other name must agree with at least two of them. The ORCID needs at least
  five full-first-name mentions, of which at most 10% can conflict with the
  other name. The Rust block-constraint methods accept an optional
  `orcid_name_waiver` argument. Set `Clusterer.orcid_name_waiver = None` to turn
  the waiver off after loading. Bundle export requires the default.
  See [ORCID name waiver](production_inference.md#orcid-name-waiver).
- The close-pair bonus lowers average-linkage height by `close_pair_bonus`
  times the share of pairs under `close_pair_distance`. `FastCluster` takes
  both parameters, and model bundles store them in `clusterer.json`.
  `calibrate-eps` selects the bonus with the clustering threshold (EPS), and
  `apply_reviewed_eps.py` accepts `--close-pair` for review without sweep curves
  and `--release-version` for the new bundle.
  See [Close-pair bonus](production_inference.md#close-pair-bonus).
- Model plans can set `eps.precision_f1_tolerance`. Calibration then selects the
  most precise setting within that tolerance of the best F1. The
  `--precision-f1-tolerance` option to `apply_reviewed_eps.py` checks the
  reviewed EPS against giant-block sweeps. It is required when reviewing
  those curves for a bundle with a close-pair bonus. Plans without a tolerance
  select the best F1.
- Profile exclusions in full-block `predict_from_arrow` and `predict` evict an
  excluded signature from its anchor's cluster. Supply the
  `profile_exclusions` argument with rows from the optional
  `profile_exclusions.arrow` table. See
  [Profile exclusions](production_inference.md#profile-exclusions).
- The pairwise `--incumbent-parameters` option lets one known configuration per
  head compete with the search winner on validation area under the receiver
  operating characteristic curve (ROC AUC). The incumbent wins ties.
  `scripts/production/model/pairwise_incumbent_h13.json` holds the `1.3`
  configuration.
- `evaluate_incremental_gold.py prepare` binds frozen cases to the benchmark
  manifest, conversion provenance, cluster copy, and unknown-signature list.
  Evaluation needs only the `dataset` bundle.

### Changed

#### Name rules

- Canonical names read two or three capitals such as `HG` as initials when
  the last name passes the case check. Mixed-case forms such as `Jh` can
  qualify when the lowercase letters contain no vowel or `y`. Given-name
  abbreviations, titles, suffixes, and real names such as `ED`, `Md`, and `Ng`
  keep their readings. Python and Rust apply the same rule, and the packaged
  aliases and counts use it.
- The hard name rule accepts joined initials when another name spells them.
  Examples include `Ka Hahn` with `Kristan Allan Hahn` and `Mal Leite` with
  `Marco Aurelio Lisboa Leite`. Incremental linking applies the same reading
  when it checks a restored profile. Canonical names, features, and count keys
  keep their original readings.
- In the middle-name field, joined initials follow the same test as short first
  names, including exceptions for suffixes, Roman numerals, and `Mc`.
  `Johanna IP` and `Johanna A IP` can match `Johanna Inge Petra` and
  `Johanna A. Inge Petra`, respectively. `Jip` also spells `Johanna IP`.
  `IP A` stays unspelled because the joined form must be the last word.
  The rule affects hard name matches only.
- When both complete given names are in the same script (Han, kana, or hangul),
  their full transliterations must agree. For example, `重喜` and `重郷` no
  longer match by prefix. The comparison combines first and middle names,
  normalizes equivalent Unicode characters, and expands `々`. Mixed scripts,
  romanized names, and incomplete transliterations use the ordinary rules.
  The same check applies to incremental links to restored profiles.
- The surname veto accepts the packaged spelling variants such as `mueller`
  and `muller`, and the Slavic endings `-sky`, `-skiy`, and `-skii`, within
  one block. Stored names, name-count keys, and features keep their spelling.
- Georgian and Armenian names use the transliteration tables in the canonical
  name rules. For example, `ბერიძე` becomes `beridze`. The rule updates
  canonical names, name-count keys, and last-name vetoes. Older indexes
  count these names under the former spellings.

See the [canonical name contract](canonical_names.md#canonical-name-contract).

#### Models and data

- Model `1.3.2` uses the `1.3.1` boosters, linker, and linkage settings with
  a name-count index rebuilt for the `1.0.0b11` Georgian and Armenian
  name rules. `scripts/production/model/rebind_name_counts_index.py` creates
  a new bundle with the updated index binding without refitting.
- Model `1.3.1` uses the `1.3` boosters and linker with EPS `0.40` and a
  close-pair bonus of `8` for pairs under `0.05`. It requires S2AND
  `1.0.0b10` or later. Its dataset bundle differs from `1.3` only in the model
  plan. See [the recorded EPS decisions](eps_calibration.md#recorded-release-decisions).
- Model `1.3` uses EPS `0.53` from the final giant-block review. Benchmark
  validation selects `0.57`. Omit `--sweep-root` and `--datasets` when using
  `scripts/eps_sweep/apply_reviewed_eps.py` to apply a release-owner choice
  without sweep curves. See [giant-block EPS review](eps_calibration.md).
- `release_pairwise.py calibrate-eps` uses validation blocks and requires the
  model's recorded population. `release_pairwise.py evaluate-heldout` scores the
  plan's outer test. A published plan can score a later model. For corrected
  benchmark populations, `pairwise/population_corrections.json` pairs the old
  and corrected fingerprints. The `1.3` and `1.3.1` dataset bundles include it.
- Calibration and held-out evaluation use unrounded per-dataset B-cubed (B3)
  scores. `s2and.metrics.b3_precision_recall_fscore` still rounds totals.
- Giant-block EPS sweeps run the production native graph planner and cluster
  cached distances through `Clusterer.predict_eps_grid_from_rust_featurizer`.
  The default `--route production` includes no-middle reunite and initial-only
  linker attachment. Use `--route bulk` for `--max-subblocks` smoke runs.
  Production runs also write `eps_b3_bulk_only.tsv`. ORCID-B3 selection keeps
  ORCID labels hidden during inference.
- When reviewing curves, `apply_reviewed_eps.py` requires `--route production`
  output with the recorded clustering source and `DISTANCE_POLICY_VERSION = 13`.
  Rerun older sweeps before using them to select EPS.

#### Prediction and evaluation

- Bulk Arrow prediction keeps zero-valued caller supervision as distance overrides
  during initial-only attachment. Standalone incremental prediction retains
  forced links. Bulk prediction ignores altered-profile markers and keeps
  effective required groups. Incremental linking still splits altered profiles
  temporarily for scoring and restores their original identity.
- Subblock keys give each compound first name one token. All-initial compounds
  such as `J.-D.` key as `j`. Other compounds join their words when the block
  supplies strong evidence for the joined form. Otherwise they key as their
  first token. See [compound first names](subblocking.md#compound-first-names).
- Subblocked Arrow prediction reunites no-middle records of a first-name dead
  end split by the size cap, between bulk prediction and initial-only attachment.
  The native subblocker returns those records by dead end as a third value.
  See [reunite-only step](subblocking.md#reunite-only-step).
- `load_production_model(path, n_jobs=N)` sets the prediction worker count at
  load time. Omit it to use the stored count.
- The [prediction tutorial](../scripts/tutorial_for_predicting_with_the_prod_model.py)
  accepts `--arrow-dataset` or `--json-dir` and can write clusters to JSON.
  It also supports a seeded incremental request through `--incremental-block`
  on Arrow. `--batching-threshold` requires the Arrow route. Use
  `release_pairwise.py evaluate-heldout` for release metrics.
- `analyze_giant_block_subblocking.py` reads `--arrow-dataset` with an optional
  `--block` and runs native graph subblocking.
- Prebuilt distance matrices passed as `dists=` skip the post-linkage
  in-block corrections. Use the EPS grid APIs with matrices from the same
  dataset or featurizer to include the corrections during calibration.
- The raw Arrow planner resolves competing query disallows deterministically:
  require-forced decisions first, then descending initial score, then signature
  ID. A conflict rebuilds the query's plan with the winning component excluded.

#### Training and publication

- Pairwise training and release evaluation compute features and distances with
  the Rust featurizer from the model plan's Arrow data. The featurizer reads
  the signatures being scored. Cluster evaluation processes one block at a time.
- `train_pairwise.py` stores feature names in the boosters.
  `PairwiseModeler.fit()` accepts optional `feature_names` for the final fit.
- Generated `within_block_random` pairs use exact seeded rank sampling. This
  preserves candidate order, selected pairs, and labels while memory scales
  with the requested sample and number of blocks. Fixed-pair inputs reject
  unordered pairs that appear in more than one split.
- Name-count indexes and model bundles validate in sibling staging directories
  and publish to a new path by rename. Linker finalization leaves its pairwise
  source intact. Release evaluation reports require finite metrics.
- Model manifests carry format and feature-contract versions, ordered feature
  names, runtime provenance, EPS lifecycle, and checksums. The feature contract
  records exact SHA-256 values for name tuples, ORCID prefix counts, and the
  name-count manifest. Bundle export and load check these bindings, and the
  linker's feature-contract digest covers all three. Models that use name counts
  also check the manifest digest before feature computation with Python, Arrow,
  or a prebuilt Rust featurizer.
- Arrow converter provenance binds signatures, papers, selected embeddings,
  and optional clusters by SHA-256. Model plans bind fixed-pair inputs
  independently. The converter verifies source content before publishing the
  root manifest and removes stale optional pair files on overwrite.
- `package_corrected_datasets.py` builds a portable corrected snapshot with
  complete incremental inputs and one common count index. Corrected source
  records preserve full paper text where available. Existing giant replay
  sources have no full abstracts.
- Release training records `featurization_backend` and, per dataset,
  `arrow_dir`, the plan's file roles, and the split mode. The training
  configuration drops `chunk_size`.
- The release loader uses the threshold in `clusterer.json`. Historical
  `1.2` and `1.21` comparisons need the corresponding runtime, which
  overrode the stored threshold to `0.65`.
- The legacy format-1 experiment fields retain historical default values for
  compatibility. They no longer affect runtime behavior.

#### Installation and release tooling

- Platform wheels for Python 3.11, 3.12, and 3.13 bundle the native extension
  as `s2and._s2and_rust`. Continuous integration tests all three Python versions,
  and the release workflow smoke-tests installed wheels.
- `uv` uses timestamp cache keys for `pyproject.toml`, Rust manifests, sources,
  and other native build inputs. If it misses a change, refresh the installed
  extension with `uv sync --extra dev --reinstall-package s2and`. See
  [build recovery](install.md#build-the-rust-extension-from-source).
- The release workflow builds the unified `s2and` wheels, checks them outside
  the source tree, and publishes through the PyPI environment from `main` only.
  It stops if the version is already on PyPI. Maturin 1.14.1 is the build floor
  and the release action version. Wheel and source distribution checks require
  the exact package inventory. The release owner also smoke-tests the real
  external model and data before publication.
- `VERSION` accepts pre-releases such as `1.0.0b1`. `sync_version.py` writes
  the corresponding Cargo version, such as `1.0.0-beta.1`. The release workflow
  can build a pre-release from any branch. See [beta pre-releases](release.md#beta-pre-releases).
- Code uses the MIT license and distributed data assets use CC BY 4.0. See
  [LICENSE](../LICENSE) and [LICENSE-DATA](../LICENSE-DATA). The distribution
  includes notices for bundled CLD2 and mimalloc.

### Fixed

- Benchmark Arrow conversion publishes validated fixed-pair labels. For
  MEDLINE, it declares `train_pairs.csv`, `test_pairs.csv`, and optional
  `val_pairs.csv` in the dataset manifest. Converters validate common count
  inputs and pair labels before preprocessing. They emit canonical pair
  columns, reject self-pairs and conflicting labels, and preserve the
  multiplicity of consistently labeled repeated pairs.
- `convert_to_arrow.py service-json`, `benchmark`, and `linker-replay` finish
  reporting successfully after writing the artifacts.
- Arrow subblocking restores seed membership with the shared cannot-link
  policy. Candidate planning and scoring use the same effective caller
  supervision. On Windows, retained Arrow readers reopen the validated handle
  to isolate Python and native cursors while retaining file identity.
- Caller supervision is undirected in prediction and distance APIs. Equal
  reverse entries are deduplicated. Conflicting directions, NaN, and infinity
  raise `ValueError` before prediction starts.
- Arrow ignores dataset disallows when
  `use_default_constraints_as_supervision=False` and keeps explicit caller
  supervision active. Incremental retrieval honors caller overrides before
  excluding candidates. Altered-profile preparation stops promoting disabled
  dataset disallows into caller supervision.
- Arrow incremental prediction rejects active explicit disallows inside an
  original seed group with `incremental_seed_disallow_conflict` before
  splitting or scoring. The check runs after applying caller overrides and the
  default-constraint switch. Follow-on attachment in subblocked bulk prediction
  preserves computed bulk groups and retains exclusions involving new
  signatures or different groups.
- Linker evaluation features are independent of target labels. Regenerate
  metrics built from label-assisted features. Input seed holdouts still
  preserve name conflicts and explicit disallows.
- JSON seed `require` pairs form connected components. Explicit `disallow`
  pairs retain precedence. Incremental altered-profile splitting honors the
  ignore-seeds flag through postprocessing. Arrow-backed facet evaluation
  computes deferred canonical names before comparing them.
- Subblocked prediction keeps original input blocks independent. Initial-only
  groups attach within their block, and an active required component exceeding
  `batching_threshold` within a block raises `ValueError`. Mixed initial and
  full-name required profiles prefer a full-name subblock. Blocks containing
  only initials use bulk prediction.
- Arrow bulk prediction normalizes integer signature IDs and supervision keys
  to strings on both subblocked and unbatched routes, including returned IDs.
  Explicit disallows and caller supervision stay active across successive
  initial-only attachment passes.
- Completion canonicalizes first names when prepared names are absent, so raw
  `"J."` is recognized as an initial. Explicitly prepared empty names stay
  empty.
- Altered-profile split components receive IDs that cannot collide with
  existing profile IDs.
- Linker source assembly rejects Arrow inputs that differ from their recorded
  manifest checksums.
- Caller distances in bulk prediction are checked against seed groups only
  within each request block. Near-hard values within `1e-6` of `0` or `10000`
  and bulk overrides below `10000` between different required seed groups
  raise an error.
- Incremental prediction keeps forced links (an exact-zero caller distance or
  a shared ORCID) to an altered profile through the final name check and
  reports `phase_b_residual_count` afterwards. Residual grouping uses the
  initial of the complete transliteration for Chinese, Japanese, and Korean
  given names.
- Fixed-pair CSVs preserve textual signature IDs such as `00`, and seed ID
  `-1` no longer matches unseeded signatures. The count rebind script rejects
  output inside its source directory. `generate_name_counts_from_parquet.py
  --resume` can adopt an index published just before a stop. An EPS retry
  reuses an ORCID-free dataset copy only for the same source.
- `apply_reviewed_eps.py` works after a plain `uv sync`. The ORCID count
  producer imports the optional `orjson` dependency only when publishing counts.

### Removed

- Use `predict_incremental_from_arrow` in place of `Clusterer.predict_incremental`
  and `predict_from_arrow` for subblocked prediction. Classic `predict`
  still clusters full blocks. The Python Arrow graph-subblocking fallback
  `make_arrow_graph_subblocking_cluster_fn` is removed. The `ANDData` graph
  subblocking reference remains.
- Prediction and bundle export raise `ValueError` if a clusterer still sets any
  of the seven name-conflict repair tuning attributes from `1.0.0b12`:
  `name_conflict_repair_uses_close_pair_bonus`,
  `name_conflict_repair_requires_full_first_names`, `name_conflict_repair_rounds`,
  `name_conflict_repair_initials_side_min_mentions`,
  `name_conflict_repair_bonus_min_close_share`,
  `name_conflict_repair_bonus_max_plain_average`, or
  `name_conflict_repair_bonus_min_side_mentions`. Delete those attributes.
- Full-file source fingerprints and the `fingerprint_source` Arrow batch-index
  writer option were removed with the switch to `S2ABI003`.
- Runtime normalization no longer includes legacy shims, Sinonym rewriting,
  fastText, or reference features. Production Rust featurization takes validated
  Arrow IPC (interprocess communication) inputs. Classic JSON/`ANDData`
  remains available for training and full-block Python prediction.
- The `S2AND_BACKEND` environment variable is removed. Direct
  `s2and.runtime.build_runtime_context(...)` calls without `backend` now
  select Python.
- `ArrowDataset.name_counts_manifest`, `ValidatedNameCountsManifest`, and
  `NameCountsBinding` are removed. Use `ArrowDataset.name_counts_index`,
  `NameCountsIndex.path`, and `NameCountsIndex.manifest_sha256`.
- `NameTupleArtifact.identity` and the native `read_name_tuple_artifact_identity`
  API are removed. The loader reads the canonical alias text directly and retains
  the aliases and `data_sha256`. Python-driven Rust flows pass the alias pairs
  explicitly.
- The Python query adapter and the native retriever's Python-object surface are
  removed. The raw Arrow planner accepts caller-declared query IDs through
  `from_query_signatures` and can derive queries with `from_auto_queries`.
- Legacy migration tools and release dispatchers are retired. Use
  [the performance profiler](../scripts/verification/profile_promoted_incremental_arrow.py),
  [the full-block parity check](../scripts/verification/compare_full_predict_arrow_parity.py),
  and [the subblocking quality check](../scripts/verification/compare_graph_subblocking_arrow_quality.py).
- `RustFeaturizer.update_cluster_seeds`, `signature_rule_metadata`, and
  `update_signature_name_counts` are removed. Apply seed overlays with
  `with_cluster_seeds`.
- The quick-start tutorial replaces `scripts/tutorial.ipynb`.

### Performance

- Bulk Arrow prediction featurizes and scores tiles in Rust, writing the
  condensed distance matrix directly. Linker candidates are scored during
  native featurization as well.
- Average linkage reuses an owned distance matrix. The memory guard accounts
  for the shared one-byte-per-pair correction data when a correction can run.
- Arrow readers decode batches in parallel within the request's thread budget.
  Supported builds use mimalloc to reduce allocation contention.
- Attachment passes for one original block share the native featurizer and
  planner. Retrieval prunes candidates using score bounds and shares per-paper
  author data across signatures.
- Native LightGBM builds exact fast scoring plans for eligible boosters:
  QuickScorer for trees with at most 32 leaves, and lockstep traversal for
  trees with up to 256 leaves. The general scorer handles other boosters.
- `NameCountsIndex.open` checks the manifest, file sizes, headers, and section
  bounds. `NameCountsIndex.open(path, verify=True)` and `verify()` also check
  full file digests and records, as required by publication tools.
- Prediction imports omit scikit-learn, LightGBM, and pandas. Native request
  state releases memory to the operating system in the background.
- Arrow-backed `ANDData` and `calibrate-eps` read correction facts directly
  from the Rust featurizer. On seven validation populations, the calibration
  grid step took 4.7 seconds instead of 27.9 seconds with identical clusters.

On Linux at 16 threads, `1.0.0b7` and `1.0.0b8` produced identical clusters
across all 13 requests. The table reports paired wall times in seconds where
both timings were recorded.

| Block | Full block b7 → b8 | Incremental b7 → b8 |
| --- | ---: | ---: |
| `s_lee` | 837 → 732 | 23.1 → 18.1 |
| `j_smith` | 136 → 130 | 6.4 → 5.3 |
| `a_khan` | 58 → 52 | 4.1 → 3.6 |
| `s_gupta` | Within 2% | 3.7 → 3.3 |
| `a_silva` | Within 2% | n/a |

### Status

- The `inference`, `dataset`, and `source` bundles
  are on S3. The current production model is `1.3.2` at
  `s3://ai2-s2-research-public/s2and/1.3.2/inference/`. Use the `1.3.1`
  dataset bundle to evaluate it. Model `1.3.1` remains available under
  `s2and/1.3.1/inference/`. See [data and models](data.md).
- PyPI has `1.0.0b14` for integration testing. Pin `s2and==1.0.0b14` for
  that beta. Model `1.3.2` requires `s2and>=1.0.0b11`. The final `1.0.0`
  release is pending. See [the release process](release.md).
- Release `1.3` uses eight non-patent pairwise sources and the 13 inherited
  non-patent incremental datasets. The corrected source snapshot supports
  the next update cycle without historical correction archives.

## 0.51.2

- Added `fingerprint_source` to the Arrow batch-index writers. Setting it to
  `False` skipped the full-file FNV hash and recorded a size-only index for
  request-time readers. Strict readers still required fingerprinted indexes.
  The returned `source_fingerprint_kind` identified size-only indexes. The
  writers rejected combining a size-only index with `overwrite=False`.
  Missing query and seed signatures still surfaced as `KeyError`.

## 0.51.1

- Made Arrow IPC the production Rust boundary, with direct bulk and incremental
  prediction over Arrow signatures, papers, authors, embeddings, batch indexes,
  and a shared name-count index. JSON and `ANDData` remained available for
  training and parity checks.
- Added Arrow runtime contracts, conversion tools, local publication validation,
  and parity and quality checks. Published the `s2and-release-arrow` data.
- Required valid Arrow artifacts and rejected unsupported alias path keys and
  prediction models that needed reference features.
- Moved production incremental retrieval and replay to Arrow and Rust. The
  canonical `s2and_and_big_blocks_linker_dataset_20260525` Arrow and labels bundle
  supplied fresh linker-training inputs.
- Replaced embedded name counts and `name_counts.arrow` with a binary
  `name_counts_index`. Promoted graph subblocking for oversized blocks.
- Kept Arrow and Rust production paths off the persistent pair-feature SQLite
  cache and invalidated native featurizer reuse after seed changes.

## Rust extension 0.50.0

- Surfaced native extension load failures as import errors.
- Used the NumPy logistic link-or-abstain gate instead of score and margin thresholds.
- Trained the final linker booster on train and weighted calibration splits and
  fit the gate on calibration. Scored the held-out test after reloading the bundle.
- Accepted joined and first-token aliases in incremental name compatibility.
- Keyed artifact cache entries by validator type.
