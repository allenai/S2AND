# Release Notes

## Unreleased 1.0.0 simplification

The coordinated `s2and` and `s2and-rust` package version is `1.0.0`. The
production model and public-data version is `1.3`, on a separate version axis.

- Correctness audit: Arrow subblocking restores seed membership with the shared
  cannot-link policy, including conflicting requires/disallows. Candidate
  planning now uses the scorer's directional supervision precedence. Windows
  retained Arrow readers reopen the validated handle to isolate Python/native
  cursors while retaining file identity, without copying data.
- Evaluation: linker features no longer depend on target labels; actual input
  seed holdouts still preserve name conflicts and explicit disallows. Metrics
  produced with label-assisted features must be regenerated. Pairwise training
  now records test block IDs and an order-independent block membership digest
  in its existing training summary. Standalone bundle evaluation verifies these
  identities for both JSON and Arrow; older bundles without them require the
  official frozen release evaluator or retraining, rather than guessing from a seed.

- Correctness: JSON seed `require` pairs now form connected components, so
  overlapping declarations cannot overwrite earlier must-links. Explicit
  `disallow` pairs retain their existing precedence. Classic Python incremental
  prediction now honors the ignore-seeds flag through postprocessing when
  splitting altered profiles. Arrow-backed facet evaluation computes deferred
  canonical names before comparing them, preserving homonymity and synonymity
  metrics across JSON and Arrow ingestion.
- **Unreleased migration state:** the artifact-independent `1.0.0` code
  and release hardening are implemented. The reviewed canonical benchmark-name
  and ORCID source exports are still needed; production artifact generation,
  the v1.3 retrain, and release quality/scale measurements follow them. The legacy
  v1.21/v1.0-v1.2 models are not packaged and are rejected by the canonical
  loader, so 1.0.0 is not yet a usable production release. See
  [release.md](release.md).
- **Python support:** `s2and` now supports Python 3.11, 3.12, and 3.13
  (previously 3.11 only), matching the range `s2and-rust` already declared.
  `s2and` remains a single pure-Python wheel; `s2and-rust` continues to ship
  per-version platform wheels. CI runs typecheck-and-test on all three
  interpreters, and the release workflow smokes the installed artifacts and
  compiles the `s2and-rust` sdist on each of them.
- Simplification: release policy now has one runbook, two prepared plans, one
  final run binding, and one reusable `ArrowDataset` handle. Completed Rust migration comparisons,
  stress tools, memory calibrators, duplicated artifact protocols, their
  dispatchers, and tool-only tests are retired. One bounded
  `scripts/verification/profile_promoted_incremental_arrow.py` command remains
  to produce the release performance report. Obsolete cache and retired-format
  tests are removed, repeated cases are curated, and the accidentally
  unbounded booster-parity matrix is deterministically budgeted without
  excluding parity from the default suite.
- Breaking: Python and Rust now ship one `canonical_v2` name/feature behavior
  under an exactly matched package runtime. Titles retain letters and digits,
  CLD2 runs in explicit plain-text mode, malformed email is missing evidence,
  query-author text uses canonical fields only, and incremental six-decimal
  values use ties-to-even rounding. Deterministic parity is pinned at `1e-6`
  (exact for discrete/count/boolean features).
- Breaking: runtime legacy-normalization shims, Sinonym rewriting, fastText,
  and reference features are removed. `s2and-rust` is a required runtime
  dependency. Production Rust featurization and ingest enter through validated
  Arrow IPC artifacts; classic JSON/`ANDData` remains a Python training,
  fixture, and reference surface for the canonical S2 partition.
- Breaking: script-side `ANDData` conversion now constructs typed PyArrow
  tables directly. The intermediate `FeatureBlock`, `FeatureBlockSignature`,
  `FeatureBlockPaper`, and `FeatureBlockPaperAuthor` object graph is removed.
  Producers use `write_raw_planner_arrow_from_anddata()` or
  `raw_planner_arrow_tables_from_anddata()`; table mappings are written with
  `write_raw_planner_arrow_tables()`.
- Breaking: `ANDData` blocking is S2-only. `author_info.block` is the sole
  grouping authority; the `block_type` constructor argument,
  `Signature.author_info_given_block`, `get_original_blocks()`, and
  `get_s2_blocks()` are removed, as is the `block_type` argument to
  `facet_eval()`. Its block-size, homonymity, and synonymity facets now use the
  same S2 block. A legacy `author_info.given_block` JSON field is tolerated as
  an ignored extra field, not used as a fallback. Workflows that require
  original benchmark partitions must preserve those groupings externally or
  use S2AND 0.51.x and earlier.
- Breaking: current production conversion and evaluation are SPECTER2-only.
  Runtime dataset conversion writes exactly one selected embedding table under
  canonical manifest key `specter`, defaulting to physical `specter2.arrow`,
  and fails if that source is absent. `eval_prod_models.py` no longer accepts a
  SPECTER1 production bundle or `--specter1-model-path`; SPECTER1 remains an
  explicit `--train` research comparison, while its historical production
  surface remains in S2AND v1.21 and earlier.
- Breaking: production-bundle evaluation reuses the trainer's recorded split
  seed. `eval_prod_models.py` reads `data_random_seed` from
  `reproducibility/pairwise_training_config.json`, fails closed for bundles
  without one, and rejects an explicit `--seed` outside `--train`. This only
  reproduces the same split when dataset bytes and ordering are identical;
  canonical release evaluation remains blocked on persisted split identities.
- Breaking: promoted stratified split assignments must carry `base_group_id`,
  and no base identity may appear in more than one split. Masked views
  (`full`, `initial_only`) of one base query previously straddled calibration
  and test (394 base identities affecting 316 test queries, or 2.67% of test
  queries, in the 20260525 assignments); the classic loader now fails closed,
  and regenerated datasets must assign splits per `base_group_id` — the same
  identity notion the classic train/holdout filter already enforces.
- Breaking: ORCID prefix counts use one validated JSON file plus a minimal
  tuple-dependency manifest. Producer provenance, schema, and metrics sidecars
  are removed.
- Breaking: name-count indexes use public format `1` and the flat
  `name_counts_index/{manifest.json,first.bin,last.bin,first_last.bin,last_first_initial.bin}`
  layout. The manifest contains a stable kind, format `1`, and byte count plus
  SHA-256 for four fixed binary roles; filenames are derived as `<role>.bin`.
  Older layouts are rejected, so rebuild Arrow and model artifacts bound to
  the manifest SHA-256.
- Breaking: Arrow dataset manifests use `kind: "s2and_arrow_dataset"`, public
  format `1`, portable `paths`, and a flat semantic-role content inventory.
  Serialized generation IDs, file/directory kind labels, and nested generation
  objects are removed. `ArrowDataset.open()` proves artifact safety, while the
  release validator owns publication topology and requires root and nested
  replay datasets to bind the publication-root `name_counts_index`.
- Breaking: `ArrowDataset.name_counts_manifest`,
  `ValidatedNameCountsManifest`, and `NameCountsBinding` are removed. Callers
  use `ArrowDataset.name_counts_index` and `NameCountsIndex.path`,
  plus `.manifest_sha256`; feature contracts bind that digest directly.
- Breaking: release orchestration uses one owner-authored `release.json` and
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
- Breaking: linker training is one direct entrypoint. It performs one fresh fit
  against the final pairwise bundle, writes a complete bundle with the
  embedded replay target, reloads those exact bytes, and evaluates them.
- Breaking: `NameTupleArtifact.identity` and
  `s2and_rust.read_name_tuple_artifact_identity` are removed. The Python loader
  validates the canonical text file directly and retains only frozen alias
  pairs plus `data_sha256`; the former `.meta.json` sidecar is removed.
  Python-driven Rust flows receive the explicit pairs rather than reopening
  the artifact in Rust.
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
- Promoted RAM limits are refreshed before each exact batch and after planner
  and featurizer allocation. When a refreshed limit shrinks the batch, the
  unscored remainder is queued and the safe prefix is replanned before scoring.
  The native LightGBM path consumes contiguous float32 features and uses
  budgeted row chunks, avoiding the previous full float64 widening transient.
  Loaded incremental linker artifacts are retained on the clusterer instead of
  being reloaded and rehashed per request.
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
- Arrow training iterates record batches and avoids duplicate full-table
  materialization. Paper-author inputs reject duplicate positions, empty names,
  and dangling references consistently in Python and Rust. Python subblocking
  rejects duplicate IDs with explicit runtime invariants.
- The `1.0.0` runtime removes legacy pickle/count-dictionary loading and redundant
  artifact plumbing. Bounded native name-count lookups, optimized float32
  scoring, and deterministic component publication reduce resource use and
  preserve artifact identities. Release-grade v1.3 memory and throughput gates
  remain pending; the retained 2026-05-28 Rust snapshot is dirty-worktree
  historical evidence, not an accepted baseline.

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
