# Release Notes

## Unreleased runtime changes

- Bundle the native extension in the `s2and` wheel as `s2and._s2and_rust`.
- Retire `Clusterer.predict_incremental` and classic `predict(batching_threshold=...)`.
  Use the corresponding Arrow APIs. Full-block classic prediction and training remain available.
- Run EPS sweep subblocking through the production native graph planner.
  Keep ORCID labels hidden for ORCID-B3 selection. Revalidate historical giant-block conclusions before publication.
- Preserve format-1 model fields for retired experiment settings; they no longer affect runtime behavior.


## Unreleased 1.0.0 simplification

The coordinated `s2and` and `s2and-rust` package version is `1.0.0`. The
production model and public-data version is `1.3`, on a separate version axis.

### Status

- **Update at v1.3 release:** The `inference`, `dataset`, and `source` bundles are on S3 (see [data](data.md)).
  PyPI has the `1.0.0b5` beta for integration testing; the final `1.0.0` release is pending.
  uv installs the beta's `s2and-rust` dependency only when it is pinned too
  (`s2and==1.0.0b5 s2and-rust==1.0.0b5`) or with `--prerelease=allow`; pip needs neither.
  The corrected source snapshot supports the next update cycle without historical correction archives.
  Release `1.3` excludes patent data.
  Pairwise inputs contain eight sources: seven non-patent benchmarks and existing non-patent augmentation.
  Incremental inputs contain the 13 inherited non-patent datasets.
  The complete patent-free candidate passes fit, evaluation, and installed Arrow inference checks.
  Final acceptance and S3/PyPI publication remain pending.
  See [the release process](release.md).

### Migrating Arrow callers from 0.51

Services that sent Arrow tables to S2AND 0.51 need these changes:

- Pin `s2and==1.0.0`. `s2and-rust` is now a plain dependency, and the `rust` extra is gone.
- The wheel no longer contains a model or a name-count pickle. Download the model and the
  name-count index from the `inference` S3 bundle (see [data](data.md)), and load the model with
  `load_production_model(path, n_jobs=N)`. `production_model_bundle_version` is now
  `production_model_release_version`, for example `"1.3"`.
- `predict_from_arrow_paths` and `predict_incremental_from_arrow_paths` are removed. Build batch
  indexes, open the tables with `ArrowDataset.open_trusted`, and call `predict_from_arrow` or
  `predict_incremental_from_arrow`.
- Seeds, exclusions, and altered profiles are prediction arguments, not path keys. 0.51 read the
  `cluster_seeds`, `cluster_seed_disallows`, and `altered_cluster_signatures` tables from the path
  mapping; `open_trusted` now rejects those keys.
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

- Releases publish three S3 bundles: `inference` (model and count index), `dataset`
  (Arrow training and evaluation data), and `source` (editable records).
  `scripts/production/data/write_release_bundles.py` writes them from the prepared
  snapshot after it proves that the Arrow data rebuilds the prepared populations.
  [The reproduction guide](reproduce.md) runs the production training process on
  `inference` plus `dataset`.
- Production training uses an Arrow model plan. Each random-block dataset names its
  complete benchmark Arrow manifest, clusters, held-out blocks, frozen held-out pairs,
  and optional unknown-signature and excluded-signature lists. A fixed-pair dataset
  names its Arrow manifest and train, validation, and held-out pairs.
  `train_pairwise.py` reads its datasets and splits from `--model-plan`. It drops
  `--production-version`, `--data-dir`, `--datasets`, `--include-augmented`,
  `--specter-suffix`, `--signatures-suffix`, `--chunk-size`, `--cluster-n-iter`,
  `--val-test-size`, `--use-cache`, and `--negative-one-for-nan`, and it rejects plans
  that name JSON papers or SPECTER pickles. The published `1.3` data reproduces the
  `1.3` training and validation matrices exactly.
- `release_pairwise.py calibrate-eps` selects EPS on validation blocks, and
  `release_pairwise.py evaluate-heldout` scores the plan's outer test. Both bind a
  model to its data by the recorded development-block membership, so the released
  `1.3` model evaluates on the published data.
- `train_pairwise.py` and `train_linker_and_finalize.py` require `--n-jobs`. Their
  former defaults, 25 and 20, could exceed the shared CPU budget.
- `train_pairwise.py --incumbent-parameters` lets one known LightGBM configuration per
  head compete with each search winner on validation ROC AUC; the incumbent wins ties.
  `scripts/production/model/pairwise_incumbent_h13.json` holds the `1.3` configuration.
- `evaluate_incremental_gold.py prepare` binds frozen cases to the benchmark manifest,
  conversion provenance, clusters copy, and unknown-signature list, so evaluation needs
  only the `dataset` bundle.

- Removes full-file Arrow source fingerprints and the `fingerprint_source`
  writer option. Batch indexes now use `S2ABI003` with a 32-byte header; rebuild
  older indexes and regenerate their dataset manifest. Validators and
  `overwrite=False` reuse check structure and size, not source content identity.
  `ArrowDataset.open` retains manifest SHA-256 checks; `open_trusted(paths)`
  accepts caller-owned tables/indexes without a dataset manifest or checksum scan.

- Code is licensed under MIT; distributed data assets use CC-BY-4.0. See
  [LICENSE](../LICENSE) and [LICENSE-DATA](../LICENSE-DATA). Rust distributions
  include the code license and notices for the bundled CLD2 and mimalloc
  dependencies.
- Release preparation resolves augmented source identities when checking
  held-out overlap in both directions: augmented training cannot contain held-out
  benchmark signatures, and augmented evaluation cannot contain benchmark
  model-development signatures. Before any featurization, pairwise training
  rejects augmented members outside the actual source training blocks, including
  validation, test, and omitted signatures.
- Arrow bulk prediction keeps zero-valued caller supervision as distance
  overrides during initial-only attachment, rather than treating them as
  mandatory incremental links. Standalone incremental requests retain their
  strict requirements. Arrow bulk accepts integer signature IDs by normalizing
  members and supervision keys to strings consistently, including returned IDs.
- `convert_to_arrow.py service-json` now prints its summary successfully after
  writing artifacts. Converters validate shared counts and fixed-pair labels
  before preprocessing, publish canonical pair column names, and reject self-pairs
  or conflicting labels. Consistent repeated pairs retain their multiplicity.
  Overwrite removes stale optional pair files, and conversion detects source
  changes before publishing the root manifest.

- The production trainer accepts the dataset names declared by the model plan instead of a fixed seven-source set.
  The `1.3` fit excludes `inventors_s2and` and retains the accepted non-patent augmented dataset.
- `package_corrected_datasets.py` produces a portable corrected snapshot with complete incremental support and one common count index.
  The complete model and the name-count index ship only in the `inference` S3 bundle.
  The repository no longer uses Git LFS; its two LightGBM test fixtures are ordinary Git files.
  Corrected source records preserve full paper text where available.
  Existing giant replay sources do not supply full abstract text.
- Models declare format and feature-contract versions plus ordered feature names.
  The model does not record the producer package version, so the loader does not compare it.
  Installed Python and Rust packages still require matched versions.
- `ANDData(preprocess=False)` no longer builds per-signature coauthor sets:
  `author_info_coauthors` and `author_info_coauthor_blocks` stay `None`. Python
  featurization, which already requires `preprocess=True`, is the only reader
  that needs them. Graph subblocking recomputes the same coauthor blocks from
  the paper authors, and evaluation falls back to the paper author count.
  Each set holds one entry per paper author, so papers with thousands of
  authors made them quadratic.
- Release featurization: `train_pairwise.py` and `release_pairwise.py`
  `calibrate-eps` and `evaluate-heldout` compute pair features and distances
  with the Rust featurizer from the model plan's Arrow data.
  On six benchmark datasets, featurization was about 100x
  faster than the Python featurizer, with bit-identical validation distances,
  B3, and trained AUC. `convert_to_arrow.py` now writes
  `conversion_provenance.json` with SHA-256 bindings for signatures, papers,
  selected embeddings, and optional clusters. Public pair CSVs remain outside
  manifest/provenance content binding; release-plan pair inputs are hashed
  separately. Converter revision metadata is informational. The pairwise
  training config drops `chunk_size` and records `featurization_backend` plus,
  per dataset, `arrow_dir`, the plan file roles, and the split mode. Each command builds the featurizer
  only for the signatures it scores, and cluster evaluation still processes
  one block at a time.
- The `S2AND_BACKEND` environment variable is removed. Public APIs already chose
  their backend by method, so only direct callers of
  `s2and.runtime.build_runtime_context(...)` without `backend` are affected;
  they now always get Python. `scripts/bench_preprocess_phases.py` drops its
  `--backend` option.
- `scripts/convert_to_arrow.py --name-counts-index-root` takes the name-count
  index directory itself, like the model, calibration, evaluation, and linker
  commands; it previously took the directory containing `name_counts_index/`.
  Passing the parent now fails with a missing-artifact error.
- Python and Rust now ship one `canonical_v2` name/feature behavior
  under an exactly matched package runtime. Titles retain letters and digits,
  CLD2 runs in explicit plain-text mode, malformed email is missing evidence,
  query-author text uses canonical fields only, and incremental six-decimal
  values use ties-to-even rounding. Deterministic parity is pinned at `1e-6`
  (exact for discrete/count/boolean features).
- Runtime legacy-normalization shims, Sinonym rewriting, fastText,
  and reference features are removed. `s2and-rust` is a required runtime
  dependency. Production Rust featurization and ingest enter through validated
  Arrow IPC artifacts; classic JSON/`ANDData` remains a Python training,
  fixture, and reference surface for the canonical S2 partition.
- `Signature.author_info_first_normalized_without_apostrophe` and
  `Signature.author_info_middle_normalized_without_apostrophe` are renamed to
  `author_info_first_normalized` and `author_info_middle_normalized`. They hold
  the canonical name parts, which already delete apostrophes; field order is
  unchanged. `s2and.text.normalize_text` drops its `special_case_apostrophes`
  flag. It still replaces apostrophes with spaces in venues, affiliations, and
  suffixes, so features do not change.
- Script-side `ANDData` conversion now constructs typed PyArrow
  tables directly. The intermediate `FeatureBlock`, `FeatureBlockSignature`,
  `FeatureBlockPaper`, and `FeatureBlockPaperAuthor` object graph is removed.
  Producers use `write_raw_planner_arrow_from_anddata()` or
  `raw_planner_arrow_tables_from_anddata()`; table mappings are written with
  `write_raw_planner_arrow_tables()`.
- `ANDData` blocking is S2-only. `author_info.block` is the sole
  grouping authority; the `block_type` constructor argument,
  `Signature.author_info_given_block`, `get_original_blocks()`, and
  `get_s2_blocks()` are removed, as is the `block_type` argument to
  `facet_eval()`. Its block-size, homonymity, and synonymity facets now use the
  same S2 block. A legacy `author_info.given_block` JSON field is tolerated as
  an ignored extra field, not used as a fallback. Workflows that require
  original benchmark partitions must preserve those groupings externally or
  use S2AND 0.51.x and earlier.
- FastCluster condensed distance matrices, including those retained
  for EPS calibration, are `float64` in Python and Rust; Python previously
  stored two bytes per pair. Recompute any cached `float16` matrices supplied
  as `val_dists_precomputed` or prediction `dists`. Pairwise training,
  `Clusterer.fit()`, and clustering evaluation now resolve the same signature
  splits, so EPS calibration uses the explicit validation split instead of the
  former random calibration population; recompute precomputed validation
  matrices built for the old population. Default split ordering and random
  seeds are unchanged.
- Current production conversion and evaluation are SPECTER2-only.
  Runtime dataset conversion writes exactly one selected embedding table under
  canonical manifest key `specter`, defaulting to physical `specter2.arrow`,
  and fails if that source is absent. `eval_prod_models.py` no longer accepts a
  SPECTER1 production bundle or `--specter1-model-path`; SPECTER1 remains an
  explicit `--train` research comparison, while its historical production
  surface remains in S2AND v1.21 and earlier.
- Production-bundle evaluation reuses the trainer's recorded split.
  `eval_prod_models.py` evaluates exactly the test blocks recorded in the
  bundle's training summary (see the Evaluation entry below), reads
  `data_random_seed` from `reproducibility/pairwise_training_config.json`,
  fails closed for bundles without them, and rejects an explicit `--seed`
  outside `--train`.
- Promoted stratified split assignments must carry `base_group_id`,
  and no base identity may appear in more than one split. Masked views
  (`full`, `initial_only`) of one base query previously straddled calibration
  and test (394 base identities affecting 316 test queries, or 2.67% of test
  queries, in the 20260525 assignments); the classic loader now fails closed,
  and regenerated datasets must assign splits per `base_group_id` — the same
  identity notion the classic train/holdout filter already enforces.
- ORCID prefix counts use one validated JSON file plus a minimal
  tuple-dependency manifest. Producer provenance, schema, and metrics sidecars
  are removed.
- Name-count indexes use public format `1` and the flat
  `name_counts_index/{manifest.json,first.bin,last.bin,first_last.bin,last_first_initial.bin}`
  layout. The manifest contains a stable kind, format `1`, and byte count plus
  SHA-256 for four fixed binary roles; filenames are derived as `<role>.bin`.
  Older layouts are rejected, so rebuild Arrow and model artifacts bound to
  the manifest SHA-256.
- Arrow dataset manifests use `kind: "s2and_arrow_dataset"`, public
  format `1`, portable `paths`, and a flat semantic-role content inventory.
  Serialized generation IDs, file/directory kind labels, and nested generation
  objects are removed. `ArrowDataset.open()` proves artifact safety, while the
  release validator owns publication topology and requires root and nested
  replay datasets to bind the publication-root `name_counts_index`.
- `ArrowDataset.name_counts_manifest`,
  `ValidatedNameCountsManifest`, and `NameCountsBinding` are removed. Callers
  use `ArrowDataset.name_counts_index` and `NameCountsIndex.path`,
  plus `.manifest_sha256`; feature contracts bind that digest directly.
- Release orchestration has no run-preparation step. The dataset bundle's
  `pairwise/model_plan.json` carries the release version and the training,
  validation, held-out, and EPS inputs. Acceptance uses `evaluate-heldout`,
  the identity-gold cases, and parity, subblocking, and performance reports
  with direct inputs; the release owner reviews them.
  Pairwise training requires the external name-count index, local matrix
  workspace, and explicit full-run acknowledgement.
- Linker training is one direct entrypoint. It performs one fresh fit
  against the final pairwise bundle, writes a complete bundle with the
  embedded replay target, reloads those exact bytes, and evaluates them.
- `NameTupleArtifact.identity` and
  `s2and_rust.read_name_tuple_artifact_identity` are removed. The Python loader
  validates the canonical text file directly and retains only frozen alias
  pairs plus `data_sha256`; the former `.meta.json` sidecar is removed.
  Python-driven Rust flows receive the explicit pairs rather than reopening
  the artifact in Rust.
- Subblock keys give each compound first name one token. An all-initial
  compound such as `J.-D.` keys as `j` and attaches with other initials. Other
  compounds join (`huimin`) only with strong joined-form evidence in the block,
  else they key as their first token. Canonical names, features and constraints
  do not change. Subblocked outputs of blocks with compound first names change.
  See [compound first names](subblocking.md#compound-first-names).
- Subblocked prediction on both routes reattaches no-middle records of a
  first-name dead end that the size cap cut across subblocks. The step runs
  between bulk prediction and initial-only attachment and sees only clusters
  of such records of one dead end. Outputs change only for dead ends whose
  clusters occupy two or more subblocks. The Rust subblocker binding returns
  these records by dead end as a third value. See
  [reunite-only step](subblocking.md#reunite-only-step).

### Fixed

- Benchmark Arrow conversion publishes fixed-pair labels. MEDLINE has pair
  labels but no clusters, and the converter previously copied only clusters
  JSON, so the public Arrow MEDLINE dataset shipped without labels. Its
  `train_pairs.csv`, `test_pairs.csv`, and optional `val_pairs.csv` are now
  validated, copied, and declared in the dataset manifest.
- `convert_to_arrow.py benchmark` and `linker-replay` no longer exit 1 with
  `KeyError: 'dataset'` after writing every dataset.
- Arrow subblocking restores seed membership with the shared
  cannot-link policy, including conflicting requires/disallows. Candidate
  planning uses the same effective caller supervision as scoring. Windows
  retained Arrow readers reopen the validated handle to isolate Python/native
  cursors while retaining file identity, without copying data.
- Caller supervision: prediction and distance-matrix APIs now treat pairs as
  undirected. Equal reverse entries are deduplicated; conflicting distances or
  NaN/infinity raise `ValueError` before prediction work starts. Requests no
  longer choose between contradictory directions based on signature order.
- Constraint parity: Arrow now bypasses dataset disallows when
  `use_default_constraints_as_supervision=False`, matching Python. Explicit
  caller supervision remains active. Arrow incremental retrieval also honors
  caller overrides of dataset disallows before excluding candidates, fixing
  incorrect abstentions and `partial_supervision_require_outside_retrieval_window`
  errors for otherwise eligible candidates. Python altered-profile preparation
  no longer promotes disabled dataset disallows into caller supervision, which
  could change query attachments by splitting a claimed profile. Precomputed
  restrictions, direct Rust-featurizer seed overrides, and transitive constraint
  semantics are unchanged.
- Incremental seed validation: Python and Arrow now reject an active explicit
  disallow between members of the same original seed group before splitting
  or scoring, with a matching `incremental_seed_disallow_conflict` `ValueError`.
  Previously, restoring the original group could silently violate that disallow.
  Caller overrides and the default-constraint switch are resolved first;
  valid original membership and exclusions involving new papers are preserved.
  Follow-on attachment in subblocked bulk prediction preserves already computed
  bulk groups, omitting hard negatives internal to those generated groups only
  from the attachment request. Finite bulk distances can average away a
  disallow; this does not turn a request without input seeds into a seed-conflict
  error. Exclusions involving new signatures or different groups remain active.
- Evaluation: linker features no longer depend on target labels; actual input
  seed holdouts still preserve name conflicts and explicit disallows. Metrics
  produced with label-assisted features must be regenerated. Pairwise training
  now records test block IDs and an order-independent block membership digest
  in its existing training summary. Standalone bundle evaluation verifies these
  identities for both JSON and Arrow; older bundles without them require the
  official frozen release evaluator or retraining, rather than guessing from a seed.
- JSON seed `require` pairs now form connected components, so
  overlapping declarations cannot overwrite earlier must-links. Explicit
  `disallow` pairs retain their existing precedence. Classic Python incremental
  prediction now honors the ignore-seeds flag through postprocessing when
  splitting altered profiles. Arrow-backed facet evaluation computes deferred
  canonical names before comparing them, preserving homonymity and synonymity
  metrics across JSON and Arrow ingestion.
- Subblocked prediction keeps original input blocks independent.
  Initial-only groups attach only to predictions from their own block, and a
  claimed profile spanning blocks no longer merges their clusters. Both routes
  keep each active required component in one subblock; on the Python route a
  component larger than `batching_threshold` within one block now raises
  `ValueError`, as Arrow already did. Blocks containing only initials use bulk
  prediction. Explicit disallows and caller supervision stay effective across
  sequential initial-only passes. Subblocked outputs can change accordingly.
- Mixed initial/full-name required profiles prefer a full-name
  subblock on both routes, retaining its native members first. This prevents
  packing from separating a claimed full-name member from compatible neighbours
  when they fit together. Arrow bulk prediction normalizes integer signature IDs
  and supervision keys to strings on both subblocked and unbatched routes.
- Completion canonicalizes first names when prepared names are
  absent, so raw `"J."` is recognized as an initial. An explicitly prepared
  empty name remains empty. This can change subblocked and incremental outputs.
- Altered-profile split components receive IDs that cannot collide
  with existing profile IDs. Previously, splitting profile `p` reused an
  existing profile named `p_0`, merging that profile into the split.
- Evaluation and artifacts: automatic `eval_prod_models` routing can use JSON
  when the Arrow source or requested dataset is unavailable, the requested
  SPECTER generation is undeclared, or the name-count index is undeclared.
  Generation selection uses the manifest's declared filename even when its
  resolved on-disk name differs. Automatic JSON fallback reports its cause.
  Broken collection bindings, missing required base artifacts, invalid manifests
  and corrupt declared data raise; empty Arrow files identify their table and
  path. `--no-arrow` skips Arrow inspection. Linker source assembly rejects Arrow
  inputs whose content differs from their manifests instead of silently
  regenerating checksums for altered inputs.

### Changed

- The `1.3` model uses EPS `0.53` from the final giant-block review; benchmark
  validation selects `0.57`. `scripts/eps_sweep/apply_reviewed_eps.py` now applies
  such an owner choice without snapshot sweep curves: omit `--sweep-root` and
  `--datasets`. It keeps the EPS range, fresh-output, model-load, and
  unchanged-artifact checks. See [giant-block EPS review](eps_calibration.md).
- Canonical names read a raw first name of two or three capitals, such as `HG`,
  as initials when the last name is not in capitals. A first name of one capital
  and one or two lowercase letters with no vowel and no `y`, such as `Jh`, also
  reads as initials. Given-name abbreviations, titles, suffixes, and real names
  such as `ED`, `MD`, `Md`, `Ch`, `Jr`, and `Ng` stay names. Python and Rust give
  the same result. The alias artifact drops the aliases that the rule now covers,
  and the packaged ORCID prefix counts match the new alias file. See
  [canonical names](canonical_names.md).
- Release: `VERSION` accepts PEP 440 pre-releases such as `1.0.0b1`. `sync_version.py`
  writes the Cargo spelling `1.0.0-beta.1`, and the Rust version check maps it back.
  The release workflow builds pre-releases from any branch and publishes only from `main`;
  see [beta pre-releases](release.md#beta-pre-releases).
- Development: the test suite runs under pytest-xdist (`-n auto`), and
  `run_ci_locally.py --pytest-workers` selects the worker count.
- `scripts/tutorial_for_predicting_with_the_prod_model.py` is a prediction-only
  example. It clusters every block of one Arrow dataset (`--arrow-dataset`) or
  S2AND JSON directory (`--json-dir`) and can write the clusters to JSON. It no
  longer scores splits; `release_pairwise.py evaluate-heldout` computes the
  release metrics. `scripts/tutorial.ipynb` is removed.
- `load_production_model(path, n_jobs=N)` sets the prediction worker count at
  load time. Without it, a loaded bundle keeps its stored `n_jobs=1`.
- Bulk prediction: altered-profile markers no longer trigger temporary seed
  splitting on subblocked Arrow requests. Both Python and Arrow now use ordinary
  required groups for bulk prediction, with explicit disallows retaining their
  existing precedence. The public Arrow argument remains accepted but is ignored
  in bulk mode. This can restore claimed membership previously split by Arrow;
  an active required component exceeding `batching_threshold` within one block
  now raises instead of being split to fit. Incremental linking still splits
  altered profiles temporarily for scoring and restores their original identity.
- Development: the repository-root `s2and_rust/__init__.py` workspace shim is
  removed, so code run from the checkout imports the installed `s2and_rust`
  package instead of an in-tree `maturin develop` build. `uv sync` and `uv run`
  use timestamp cache keys for native build inputs, and a fresh
  `uv sync --extra dev` checkout can run the tests without `maturin develop`.
  Tests stop before collection when the importable extension's embedded source
  snapshot differs from the checkout; force a refresh with
  `uv sync --extra dev --reinstall-package s2and-rust`. The content digest
  covers native manifests, the root lockfile, build script and Rust sources,
  excluding editor files, build outputs and the unused vendored lockfile. uv's
  cache keys cover those inputs plus `s2and_rust/pyproject.toml`, its
  `README.md`, and the Python wrapper files, which the test guard does not
  check. uv can miss backdated additions, older-file deletions and edits that
  preserve timestamps; see [build recovery](install.md#build-the-rust-extension-from-source)
  if Cargo also needs its cached artifacts cleared.
- Python support: `s2and` now supports Python 3.11, 3.12, and 3.13
  (previously 3.11 only), matching the range `s2and-rust` already declared.
  `s2and` remains a single pure-Python wheel; `s2and-rust` continues to ship
  per-version platform wheels. CI runs typecheck-and-test on all three
  interpreters, and the release workflow smokes the installed artifacts and
  compiles the `s2and-rust` sdist on each of them.
- The canonical alias artifact now contains 5,163 pairs. A manual review of all
  2,266 legacy-only candidates restored 1,343 credible aliases, retained all
  3,684 existing pairs, and excluded 906 rejects plus 17 unresolved pairs.
  Later reviews add 253 aliases and remove 117. Tests keep
  the adjudication ledger in `tests/fixtures/name_tuples/` consistent with the
  artifact.
- Promoted query-disallow resolution is request-global and deterministic:
  require-forced decisions first, then descending initial score, then signature
  ID. Conflicts rebuild and score a complete single-query plan with the winning
  component excluded, preserving outcomes across input permutations and batch
  sizes. Physical batch telemetry may vary with the RAM plan.
- The native name-count index, Arrow inputs, pairwise boosters, and linker
  metadata retain the runtime checks needed to prevent incompatible feature
  computation. The historical name-count pickle is not a runtime or published
  representation. Models selecting name-count features compare the exact
  `name_counts_manifest_sha256` at Python, Arrow, and prebuilt-Rust-featurizer
  boundaries before feature work.
- Arrow production inputs require a canonical content-addressed manifest.
  `ArrowDataset.open(root)` validates immutable files and indexes once and
  retains their native readers for its lifetime; prediction receives
  request-local seeds explicitly and reuses the open handle.
- Stored language evidence is now an all-or-nothing triple with finite
  reliability in `[0,1]`; unreliable rows must carry zero. Python and Rust
  reject partial/malformed values and agree across pair order. Raw candidate
  planning likewise has only two valid constructors: declared queries and
  explicit automatic queries; the empty-sidecar/boolean-bypass state is gone.
- Name-count indexes are assembled completely in a temporary sibling and
  published with one rename into an absent `name_counts_index` target.
  Production count generators consume reviewed CSV exports, keep warehouse
  clients and credentials outside the repository, and bound local fixtures.
  ORCID counts use one direct JSON file plus one minimal
  tuple-dependency manifest, with no producer-provenance protocol, pointer
  manifest, retry loop, or legacy fallback. The large name-count index remains
  external data rather than Python package data.
- Generated `within_block_random` pair sampling now uses exact seeded rank
  sampling. It preserves the legacy candidate order, selected pairs, and labels
  while memory scales with requested samples plus blocks instead of all
  candidate pairs. Fixed-pair CSV datasets remain fully loaded during the
  release-only training run.
- Fixed train/validation/test CSV inputs now reject any unordered pair that
  appears in more than one split. Preflight performs this schema, duplication,
  and overlap validation before expensive featurization.
- Pairwise and complete-model bundles validate in sibling staging directories
  and publish with one rename into a new path; finalization never mutates the
  pairwise source. The release evaluation report rejects missing or nonfinite
  required metrics. Wheel/sdist validation rejects undeclared production assets
  and any packaged production-model/default path.
  Production training records the canonical name-tuple and ORCID prefix-count
  data SHA-256 values plus the exact name-count manifest SHA-256 in the feature
  contract; bundle export and load require exact matches, and the linker
  binding covers all three through its ordered feature-contract digest. Model
  roots use one fixed-role manifest with format and feature-contract versions, ordered features, runtime provenance, EPS lifecycle, and checksums.
  Historical
  commit `e54c6ba` documents the published v1.21
  loader's explicit clustering-threshold override to `0.65` for versions
  `1.2`/`1.21`; the stored threshold is stale. The current canonical loader has
  no legacy override.
  An optional historical model comparison requires its compatible runtime.
  New canonical models use their own `clusterer.json` value.
- The release workflow tests the exact commit,
  builds and installs Python and Rust artifacts outside the source tree, runs
  platform wheel smokes, and enforces the `main` ref in the workflow itself
  before routing publication through the PyPI environment. It publishes Rust,
  waits for that exact dependency, and then publishes Python. The Rust
  build-system floor and release action are aligned at Maturin 1.14.1. The real
  external v1.3 model/data smoke remains a pre-publication operator gate.

### Performance

- Request-time batch lookup indexes are built with NumPy instead of a Python loop
  per row: keys hash from the Arrow buffers, records sort with `lexsort`, and runs
  write as packed arrays. Index files are byte-identical. For the a_khan request
  (95,936 signatures, 4.19M author rows) the build takes 0.9 s instead of 13.3 s.
  Runs spill to disk above 16M records instead of 1M.
- The `paper_authors` Arrow reader resolves each paper once and hashes rows with
  `ahash`, instead of three SipHash lookups and a string copy per row. On the
  j_smith incremental request (110,244 signatures, Linux, 16 threads) prediction
  takes 37.9 s instead of 40–42 s, with identical clusters.
- Retrieval scoring computes a query's cosines with a component centroid and its
  exemplars in one pass. Each sum keeps its order, so scores are bit-identical.
  At 5 threads, incremental j_smith and a_khan requests use 8–10% less CPU, and
  the subblocked a_silva full-block request 12% less (731 instead of 834 CPU
  seconds), with identical clusters.
- Subblocked attachment passes of one original block share native state. One
  featurizer over the block is built on first use, and each pass views it with
  its own seeds instead of rebuilding features; a view matches a fresh build
  exactly. The incremental planner can reload new seed sidecars when they include
  every current seed, reading only the missing rows. On the subblocked j_smith
  full-block request (26 attachment passes, Linux, 16 threads) featurizer time
  across the passes drops from 173 s to 16 s, and prediction takes 775 s instead
  of 1,091 s, with identical clusters.
- Attachment passes no longer reread or rebuild the block's seed side. The first
  planner build of a block also reads the rows of its other signatures; later
  passes plan their queries and reload their seeds from those rows instead of
  rescanning the Arrow tables. A reload keeps the features of current seeds and
  moves over the summary of each component whose ordered members are unchanged.
  Plan row signals and author-list evidence are computed in parallel, and
  residual clustering inside a pass views the block featurizer instead of
  building one from Arrow. Plans and features equal fresh builds.
- When supervision fixes some rows of a bulk chunk, Rust derives the scored rows'
  pairs from their offsets instead of parsing one Python tuple per pair.
- The incremental linker finds supervised candidate pairs by index instead of
  converting every candidate pair to ids, and plan conversion uses C-level loops.
- Retrieval scores blocks of 32 queries against each component in turn, so each
  component's SPECTER vectors stream from memory once per block instead of once
  per query. Dot products sum in eight fixed lanes, with AVX2 where available
  and the same lanes otherwise, so scores do not depend on the CPU. Sums can
  differ from the previous sequential order in the last bits; on the scholar
  export requests, ranked candidates, scores and clusters were identical. On the
  s_lee incremental request retrieval takes 13 s instead of 46 s.
- The incremental linker builds subset-gate features once per scoring batch
  instead of once per query, and still predicts each query from its own row, so
  decisions are bit-identical.
- A filtered `paper_authors` read no longer hashes every row into a duplicate
  set; the positions of skipped papers are checked per paper instead.
- Request window featurizers are built from the rows the incremental planner
  already holds, instead of reading them from Arrow again.
- The incremental ORCID loader finds its rows with a Python set, because pyarrow's
  `is_in` rehashed every requested id for each batch. For the 512,686 seeds of the
  s_lee request it takes 2.5 s instead of 4.5 s.
- Together, on Linux at 16 threads with identical clusters on all 13 scholar
  export requests, `1.0.0b3` full-block prediction takes 393 s instead of `1.0.0b2`'s
  706 s on j_smith (110,244 signatures), 2,599 s instead of 3,152 s on s_lee
  (516,747 signatures), 161 s instead of 221 s on a_khan, 163 s instead of 221 s on
  s_gupta, and 91 s instead of 123 s on a_silva. Incremental prediction takes 76 s
  instead of 133 s on s_lee, 22 s instead of 33 s on j_smith, 13 s instead of 25 s on
  a_khan, and 12 s instead of 16 s on s_gupta. (The `1.0.0b3` notes compared against
  earlier `1.0.0b2` runs that a busier machine had slowed; these `1.0.0b2` times are
  reruns beside `1.0.0b1`.)
- The extension uses mimalloc as its memory allocator on x86_64 Linux (glibc),
  macOS, and Windows. Worker threads that allocate in parallel contended for glibc's
  malloc arenas, which slowed later kernels. Musl and aarch64 Linux wheels keep the
  system allocator; the aarch64 manylinux2014 cross compiler cannot build mimalloc. With
  identical clusters, the s_lee incremental request with one new signature takes
  21.6 s instead of 31.9 s, and the j_smith full block 399 s instead of 454 s.
- Arrow reads decode and convert record batches in parallel on the request's thread
  budget and release the GIL. Any error or cross-batch duplicate falls back to the
  serial reader, so results and error messages are unchanged. On top of mimalloc,
  the one-signature s_lee request takes 17.5 s instead of 20.9 s, and the a_silva
  full block 90 s instead of 101 s.
- Together, on the same 13 requests with identical clusters, `1.0.0b4` incremental
  prediction takes 52 s instead of `1.0.0b3`'s 76 s on s_lee, 15 s instead of 22 s on
  j_smith, 8.9 s instead of 12.9 s on a_khan, and 8.2 s instead of 12.0 s on s_gupta.
  Full-block prediction takes 2,142 s instead of 2,599 s on s_lee, 342 s instead of
  393 s on j_smith, 148 s instead of 161 s on a_khan, 150 s instead of 163 s on
  s_gupta, and 82 s instead of 91 s on a_silva. One new signature into s_lee's
  512,686 seeds takes 15.8 s instead of 28.4 s. Peak RSS of the s_lee full block is
  15.2 GB instead of 17.7 GB; the smallest requests use about 0.1 GB more.
- Hard constraints split each signature's names once per call and compare
  precomputed keys, instead of splitting both names again for every pair. With
  identical clusters, the a_silva full block takes 80 s instead of 87 s and the
  j_smith full block 346 s instead of 359 s.
- Request window featurizers stage signatures and papers on the request's thread
  pool. The vendored CLD2 wrapper serializes language detection behind a global
  lock, so one thread detects titles while the others stage signatures. Title,
  venue and affiliation n-grams are counted by hashing each gram's bytes, without
  a string per gram. On the s_lee incremental request the window featurizer takes
  about 9 s instead of about 15 s.
- Large arrays are no longer copied on one thread. Featurizer outputs move into
  NumPy without a copy, the scorer copies its input batch in parallel, compact
  batches reach the native scorer as views, and model columns are gathered by a
  parallel Rust function instead of `np.take`. Linker distance summaries use array
  operations instead of per-row Python loops, and summed values keep their order.
  Upper-triangle pairs are computed from their offsets in parallel. With identical
  clusters, the j_smith full block takes 279 s instead of 317 s and the a_silva
  full block 66 s instead of 72 s.
- fastcluster linkage of one block runs on a worker thread while the next block's
  distance matrix builds; two matrices are alive only while the clustering one is
  below a tenth of the RAM budget. Peak RSS of the a_silva full block is 3.5 GB
  instead of 3.1 GB.
- Arrow prediction pauses Python's cyclic garbage collector, ORCID rows load once
  per attachment block instead of once per pass, and retrieval scores up to four
  queries per SPECTER vector chunk with bit-identical sums.
- Extension maps and sets hash with aHash instead of SipHash. With identical
  clusters, the s_lee incremental request takes 43 s instead of 50 s, one new
  signature into s_lee 13.3 s instead of 15.6 s, and the a_silva full block 58 s
  instead of 61 s.
- The incremental planner returns each plan's distinct pair signatures once, with
  per-pair positions into that list, instead of two id strings per pair. This is
  the internal `RawBlockQueryCandidatePlanner.plan` payload; the j_smith full
  block is 1-2% faster.
- Together, on the same 13 requests with identical clusters, run right after a
  rerun of the published `1.0.0b4`, `1.0.0b5` incremental prediction takes 39.5 s
  instead of 51.5 s on s_lee, 10.0 s instead of 14.7 s on j_smith, 6.8 s instead of
  8.8 s on a_khan, and 6.3 s instead of 8.0 s on s_gupta. Full-block prediction takes
  1,535 s instead of 2,062 s on s_lee, 240 s instead of 335 s on j_smith, 103 s
  instead of 148 s on a_khan, 109 s instead of 147 s on s_gupta, and 57 s instead of
  82 s on a_silva. One new signature into s_lee's 512,686 seeds takes 13.3 s instead
  of 15.6 s. Peak RSS of the a_silva full block is 3.6 GB instead of 3.1 GB; other
  requests change by at most 0.1 GB.
- `NameCountsIndex.open` no longer hashes the index files or checks every record on
  each open. It checks the manifest, file sizes, file headers, and section bounds,
  so lookups stay memory-safe. For the 2.3 GB `1.3` index an open takes 0.04 s and
  35 MB of working set instead of 3.4 s and 2.46 GB. `open(path, verify=True)` and
  `verify()` run the full checks; publication tools use them. On the s_feldman
  block this removes the index read from every new prediction process.
- Bulk Arrow prediction scores up to 100,000 pairs per Rust call instead of
  10,000, and Rust returns constraint labels as one float64 array instead of
  one Python value per pair. The memory budget still lowers the chunk size when
  needed. On the 6,495-signature s_feldman block (21.1M pairs, 8 threads),
  full-block prediction takes 18.0 s instead of 27.7 s, peak working set is
  0.54 GB instead of 0.64 GB, and clusters are identical.
- Logistic gate features reduce all queries with the same candidate count as
  one NumPy block, instead of calling NumPy once per query and feature. Gate
  matrices are bit-identical. For 6,487 s_feldman queries the gate takes 0.06 s
  instead of 7.9 s, and incremental prediction takes 18.7 s instead of 36.4 s
  with identical clusters. The gate parses its feature names once and reduces
  evidence sources without missing values together. In a 200-query request it
  re-runs for 158 single queries after links; those calls take 0.20 s instead
  of 0.66 s.
- Prediction processes no longer import scikit-learn, LightGBM, or pandas.
  Importing the runtime modules takes 0.66 s instead of 2.20 s. With the gate
  change, a 200-query s_feldman incremental request takes 3.0 s instead of
  5.1 s end to end. `FastCluster` no longer inherits from scikit-learn's `BaseEstimator` and
  `TransformerMixin`. It keeps `get_params`, `set_params`, its repr, and its
  error messages, and `sklearn.base.clone` still copies it. Training code
  imports scikit-learn, LightGBM, and pandas where it uses them.
- The Rust featurizer stores each paper's coauthor names, name blocks, and
  character n-gram counts once, instead of copying them into every signature on
  the paper. Pair features subtract each signature's own author, and they are
  bit-identical to before. On INSPIRE, whose papers have up to 3,167 authors, a
  featurizer for 174,652 signatures now builds in 7 s instead of 250 s and
  frees in 11 s instead of 18 minutes. Duplicate author positions remain rejected
  by the Arrow reader and converter.
- The raw Arrow candidate planner and `raw_arrow_labeled_candidate_plan`
  normalize authors into shared per-paper tables, instead of once per signature
  and again for every component summary that holds it. Query coauthor blocks
  are read off a per-paper block table, and paper author-list evidence
  intersects interned name ids instead of string sets. Planner payloads and
  incremental-linking clusters are bit-identical to before. On the production
  `s_lee` linker block (422k signatures, papers of up to 8,099 authors), the
  labeled-row plan over its 152,900 evaluation rows takes 31 s instead of 54 s
  and peaks at 6.7 GB instead of 8.1 GB.
- Retrieval scores coauthor and affiliation overlap through an inverted index
  of the component counters, built with the retriever, and computes query
  term weights once per query instead of once per component. Scores, top-k
  order and incremental-linking clusters are bit-identical to before. On
  `s_lee` (21,669 components), a query with about 2,900 coauthor terms takes
  43 ms instead of 317 ms, an ordinary query 6 ms instead of 9 ms, and
  planning 2,000 evaluation queries takes 17.2 s instead of 24.1 s in the
  recorded implementation benchmark. The index
  holds 3.2M postings (48 MB) and builds in 0.06 s. Python-built
  `RustHybridCentroidRetriever` summaries with colliding FNV-64 keys in coauthor,
  non-mega-coauthor, or affiliation overlap counters are rejected.
- `RustLightGBMBooster` builds an exact fast scorer at load time. Models whose
  trees all have at most 32 leaves use QuickScorer; models with up to 256
  leaves per tree use lockstep traversal over integer bin keys. Both support up
  to 256 features and 255 distinct thresholds per feature, which covers
  LightGBM's default `max_bin=255`. Raw scores and probabilities are
  bit-identical to the previous traversal. That traversal still scores every
  other model, including single-tree constants/stumps, Zero-missing splits, NaN thresholds, and parser-accepted
  shared-child graphs. On the v1.21 pairwise models and 50,000 real rows,
  8-thread float32 scoring was 6.2x (main) and 3.3x (nameless) faster. A 5-block
  AMiner bulk request (8.76M pairs) ran 2.79x faster end to end (46.3 s to
  16.6 s), with identical clusters and chunk digests. The public API is
  unchanged.
- Python pair featurization runs serially on Windows. Its thread pool was
  slower than serial there, and the output is bit-identical.
- The default bulk prediction path avoids repeating caller-supervision
  validation for ordinary subblocks. Public entry points, custom helper
  overrides, and incremental attachment passes retain their validation.
- Promoted RAM limits are refreshed before planning each query window, after
  planning, and before each scoring batch. A shrunk window plans only its safe
  prefix and defers the remaining queries; a shrunk scoring batch is scored from
  slices of the window's existing plan.
  The native LightGBM path consumes contiguous float32 features and uses
  budgeted row chunks, avoiding the previous full float64 widening transient.
  Loaded incremental linker artifacts are retained on the clusterer instead of
  being reloaded and rehashed per request.
- Arrow training iterates record batches and avoids duplicate full-table
  materialization. Paper-author inputs reject duplicate positions, empty names,
  and dangling references consistently in Python and Rust. Python subblocking
  rejects duplicate IDs with explicit runtime invariants.
- The `1.0.0` runtime removes legacy pickle/count-dictionary loading and redundant
  artifact plumbing. Bounded native name-count lookups, optimized float32
  scoring, and deterministic component publication reduce resource use and
  preserve artifact identities. **Update at v1.3 release:** Release-grade v1.3
  memory and throughput gates remain pending.

### Removed and simplified

- Release policy now has one runbook, two prepared plans, one
  final run binding, and one reusable `ArrowDataset` handle. Completed Rust migration comparisons,
  stress tools, memory calibrators, duplicated artifact protocols, their
  dispatchers, and tool-only tests are retired. The bounded release
  verification commands that remain are
  `scripts/verification/profile_promoted_incremental_arrow.py` (performance
  report), `compare_full_predict_arrow_parity.py` (parity report), and
  `compare_graph_subblocking_arrow_quality.py` (subblocking report). Obsolete cache and retired-format
  tests are removed, repeated cases are curated, and the accidentally
  unbounded booster-parity matrix is deterministically budgeted without
  excluding parity from the default suite.

## 0.51.2

- Ships the package as `0.51.2` and pins optional Rust installs to `s2and-rust==0.51.2`.
- Adds `fingerprint_source` to `write_arrow_batch_lookup_index` and `write_raw_arrow_batch_lookup_indexes` (default `True`, so existing callers are unchanged). Passing `False` skips the full-file FNV hash of each Arrow table and stores `ARROW_BATCH_LOOKUP_INDEX_UNFINGERPRINTED` in the header, for callers that just wrote the files themselves and index them per request; it cannot be combined with `overwrite=False`. Request-time readers (`read_arrow_batch_lookup_index_batch_indices_for_request`, Rust `open_for_request`) already compare size only and accept such indexes. Strict validation rejects them with an explicit message, and the Rust raw planner's missing-signature diagnostic no longer strict-validates an unfingerprinted index, so a missing query or seed signature still surfaces as `KeyError`. `source_fingerprint_kind` in the returned metrics reports `size_only` for them.

## 0.51.1

- Ships the package as `0.51.1` and pins optional Rust installs to `s2and-rust==0.51.1`.
- Makes Arrow IPC the production Rust runtime boundary. Direct prediction now uses `Clusterer.predict_from_arrow_paths(...)` or Arrow-routed `Clusterer.predict(...)` over `signatures`, `papers`, `paper_authors`, selected `specter`, raw-planner batch indexes, and shared `name_counts_index`. JSON/`ANDData` remains available for compatibility, training, fixtures, and parity checks.
- Adds canonical Arrow runtime contracts and tooling: `s2and.arrow_inputs`, `s2and/arrow_schema_contract.json`, `scripts/convert_to_arrow.py`, `scripts/arrow_conversion_helpers.py`, local Arrow release validation, and bounded parity/quality verification scripts. The documented public data release is now `s2and-release-arrow`; the legacy JSON/pickle release remains for paper-era inputs.
- Tightens production validation. Missing or malformed Arrow artifacts now raise structured `MissingArrowArtifactError` failures, Rust production routes fail fast instead of silently falling back to `ANDData`, unsupported name-alias path keys are rejected, and direct Arrow prediction refuses models that require reference features.
- Reworks promoted incremental linking around Arrow/Rust. The promoted path reads base Arrow artifacts, query signatures, cluster seeds, cluster seed disallows, and altered-profile sidecars; Rust performs raw candidate planning and promoted row-signal construction; `batching_threshold` controls promoted Rust query batch size.
- Switches promoted linker replay/training to one fresh Arrow/Rust flow against the canonical `s2and_and_big_blocks_linker_dataset_20260525` Arrow+labels bundle. Cached/precomputed feature modes are no longer supported.
- Uses the manifest-backed binary `name_counts_index/` sidecar as the sole supported name-count artifact. Embedded Arrow name-count columns and the standalone `name_counts.arrow` artifact are no longer supported.
- Updates model, eval, tutorial, and profiling flows for Arrow-first operation. `eval_prod_models.py` can auto-use Arrow when complete artifacts exist, the production tutorial defaults to Arrow input, the documented model path is the native `production_model_v1.21/` bundle, and `rust_suite.py promoted-incremental-arrow-profile` replaces the legacy big-block incremental profiler.
- Adds graph subblocking as the default fallback for oversized name groups and introduces strict Arrow-native Rust graph subblocking with batch-indexed reads, expanded telemetry, ORCID repair behavior, and scoped dash-name compatibility.
- Narrows the Rust production surface to one route per job. Legacy/debug bridge APIs are removed or demoted, including raw FeatureBlock scoring, string-pair constraint matrix APIs, retriever debug APIs, aggregate-only linker helpers, and `s2and/rust_capabilities.py`.
- Clarifies cache behavior for the new runtime boundary. Direct Arrow/Rust production prediction bypasses the persistent pair-feature SQLite cache, Rust featurizer reuse is invalidated by cluster-seed changes, and artifact cache entries no longer probe raw-ETag filenames.

## Rust extension 0.50.0

- `s2and-rust>=0.50.0` is required for Rust-backed incremental linking.
- Native extension load failures now surface as import errors instead of silently falling back to Python. Missing extension modules still use the Python fallback path.
- Incremental linking uses the NumPy logistic link-or-abstain gate artifact format; legacy score/margin gate thresholds are not supported.
- Production linker finalization trains the final booster on train plus weighted
  calibration splits and fits the logistic gate on the configured calibration
  splits. The held-out test split is materialized and evaluated only after the
  complete bundle is serialized and reloaded.
- Incremental name compatibility now accepts joined and first-token aliases in addition to exact first-name tuples.
- Artifact cache entries are keyed by validator type. Raw-ETag cache filenames are no longer probed.
