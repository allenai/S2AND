# Release Notes

## Unreleased 1.0.0 simplification

The coordinated `s2and` and `s2and-rust` package version is `1.0.0`. The
production model and public-data version is `1.3`, on a separate version axis.

### Status

- **Unreleased migration state (Update at v1.3 release: replace this bullet
  with the published version matrix and artifact locations):** the
  artifact-independent `1.0.0` code and release hardening are implemented. The reviewed canonical benchmark-name
  and ORCID source exports are still needed; production artifact generation,
  the v1.3 retrain, and release quality/scale measurements follow them. The legacy
  v1.21/v1.0-v1.2 models are not packaged and are rejected by the canonical
  loader, so 1.0.0 is not yet a usable production release. See
  [release.md](release.md).

### Breaking changes

- The production pairwise training population drops the internal `orcid`
  dataset: v1.3 trains on `aminer`, `arnetminer`, `inspire`, `kisti`, `pubmed`,
  `qian`, and `zbmath` in random-block mode plus `augmented` in fixed-pair mode.
  Model plans that still list `orcid` are rejected.
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
  bundle's training summary (see the Evaluation entry above), reads
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
- Release orchestration uses one owner-authored `release.json` and
  two generated plans. `release.json.release_version` is the one human-owned
  model/data release choice; `model_plan.json` carries it with training,
  validation, and EPS inputs, and final public-data assembly reads that plan.
  `evaluation_plan.json` owns held-out inputs, gates,
  the reviewed baseline identity, performance inputs, and exact parity and
  subblocking content/workload identities. A post-finalization
  `run_binding.json` binds both path-independent plans, the complete candidate,
  and the public-data root; all five component reports and the aggregate
  decision carry that binding. Unreleased manifest/spec formats and their
  compatibility-free CLI flags are removed.
  Fixed-role plans and reports no longer carry decorative schema labels.
  Pairwise training still requires the external name-count index, local matrix
  workspace, validation-pair size, and explicit full-run acknowledgement.
- Linker training is one direct entrypoint. It performs one fresh fit
  against the final pairwise bundle, writes a complete bundle with the
  embedded replay target, reloads those exact bytes, and evaluates them.
- `NameTupleArtifact.identity` and
  `s2and_rust.read_name_tuple_artifact_identity` are removed. The Python loader
  validates the canonical text file directly and retains only frozen alias
  pairs plus `data_sha256`; the former `.meta.json` sidecar is removed.
  Python-driven Rust flows receive the explicit pairs rather than reopening
  the artifact in Rust.

### Fixed

- Benchmark Arrow conversion publishes fixed-pair labels. MEDLINE has pair
  labels but no clusters, and the converter previously copied only clusters
  JSON, so the public Arrow MEDLINE dataset shipped without labels. Its
  `train_pairs.csv`, `test_pairs.csv`, and optional `val_pairs.csv` are now
  validated, copied, and declared in the dataset manifest.
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
  when they fit together. Arrow subblocking also preserves support for integer
  signature IDs by normalizing the original-block scope.
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

- `scripts/convert_to_arrow.py join-canonical-augmented-names` gives the
  internal augmented training set its source datasets' canonical names,
  re-shortening the first names the augmentation had cut to an initial, and
  drops signatures and pairs from `--drop-sources` (v1.3: `orcid`). It writes
  release-ready inputs: training pairs with `signature_id_1`/`signature_id_2`
  columns and held-out pairs in the pairwise-evaluation JSON format.
- `scripts/tutorial_for_predicting_with_the_prod_model.py` scores the bundle's
  recorded held-out blocks with the trainer's recorded seed, applies
  `--specter-suffix` to the JSON route as well as Arrow (no silent SPECTER1
  fallback), and drops `--load-name-counts`, which no production bundle accepts.
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
- The canonical alias artifact now contains 5,027 pairs. A manual review of all
  2,266 legacy-only candidates restored 1,343 credible aliases, retained all
  3,684 existing pairs, and excluded 906 rejects plus 17 unresolved pairs. The
  complete adjudication ledger and rubric are retained under
  `docs/release_evidence/`.
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
  roots use one fixed-role manifest with release version, exact generating
  runtime, EPS lifecycle, and checksum inventory. Clusterer configuration and
  scorer fixtures have no independent schema/version counters. Historical
  commit `e54c6ba` documents the published v1.21
  loader's explicit clustering-threshold override to `0.65` for versions
  `1.2`/`1.21`; the stored threshold is stale. The current canonical loader has
  no legacy override, so a v1.21 baseline must use that compatible historical
  runtime. Newly tuned canonical bundles use their own `clusterer.json` value.
- The release workflow has no evidence inputs. It tests the exact commit,
  builds and installs Python and Rust artifacts outside the source tree, runs
  platform wheel smokes, and enforces the `main` ref in the workflow itself
  before routing publication through the PyPI environment. It publishes Rust,
  waits for that exact dependency, and then publishes Python. The Rust
  build-system floor and release action are aligned at Maturin 1.14.1. The real
  external v1.3 model/data smoke remains a pre-publication operator gate.

### Performance

- The default bulk prediction path validates caller supervision
  once per request instead of once per subblock, while preserving custom
  prediction-helper overrides and validation at public entry points.
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
  preserve artifact identities. Release-grade v1.3 memory and throughput gates
  remain pending.

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
