# Release Notes

## Unreleased canonical-v2 migration

The manifests currently say `0.60.0`; the coordinated release may retain that
package version or become `1.3.0`. Model/data bundle v1.3 is a separate version
axis. This decision is release blocker B01.

- **Unreleased migration state:** the artifact-independent canonical-v2 code
  and release hardening are implemented, but canonical name counts, canonical
  ORCID prefix counts, benchmark-name re-export, the v1.3 retrain, and
  release-candidate quality/scale measurements are still pending. The legacy
  v1.21/v1.0-v1.2 models are not packaged and are rejected by the canonical
  loader, so 0.60.0 is not yet a usable production release. See
  [1_3_release_todo.md](1_3_release_todo.md).
- Breaking: Python and Rust now share one `canonical_v2` name contract and one
  versioned feature contract. `FEATURIZER_VERSION` is 10. Titles retain letters
  and digits, CLD2 runs in explicit plain-text mode, malformed email is missing
  evidence, query-author text uses canonical fields only, and incremental
  six-decimal values use ties-to-even rounding. Deterministic parity is pinned
  at `1e-6` (exact for discrete/count/boolean features).
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
- Breaking: ORCID prefix counts now use one
  `first_k_letter_counts_from_orcid.manifest.json`; the former runtime metadata
  and producer-report sidecars are removed.
- Breaking: pairwise `--datasets` runs are non-publishable smoke runs by
  definition. The redundant `--smoke-only` and guessed disk-headroom flags are
  removed.
- Breaking: linker training now has explicit `preflight`, `materialize`,
  `candidate`, and `publish` commands. Candidate runs retain the exact
  evaluated model and measured report beside `candidate_target.json`; they
  cannot approve a release. They also persist deterministic query-level
  decisions and their digest inventory. The existing `publish` command is not a
  v1.3 release authority: B20 still requires a thin no-training v5 assembly
  wrapper, and the aggregate quality report alone decides release eligibility.
- Breaking: `NameTupleArtifact.identity` and
  `s2and_rust.read_name_tuple_artifact_identity` are removed. The Python loader
  validates each artifact once and retains only frozen alias pairs plus
  `data_sha256`; Python-driven Rust flows receive those explicit pairs rather
  than reopening the artifact in Rust.
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
  metadata carry and verify normalization, generation, size, SHA-256, and
  feature-contract provenance. The historical name-count pickle is not a
  runtime or published representation. Manifest-relative paths cannot escape
  their authority or depend on process CWD. Equal-size/equal-mtime mutation of
  altered-profile/disallow inputs is detected before altered-presplit reuse.
  Models selecting name-count features compare the exact
  `name_counts_manifest_sha256` at Python, Arrow, and prebuilt-Rust-featurizer
  boundaries before feature work.
- Arrow production inputs require a canonical content-addressed generation
  manifest. Immutable dataset files and indexes are inventoried centrally;
  request-local seed/query sidecars are kept outside that identity and parsed
  once per request. Hot featurizer-cache checks bind exact immutable paths and
  generation identity without rehashing files on a hit.
- Stored language evidence is now an all-or-nothing triple with finite
  reliability in `[0,1]`; unreliable rows must carry zero. Python and Rust
  reject partial/malformed values and agree across pair order. Raw candidate
  planning likewise has only two valid constructors: declared queries and
  explicit automatic queries; the empty-sidecar/boolean-bypass state is gone.
- Name-count indexes are assembled completely in a temporary sibling and
  published with one rename into an absent `name_counts_index` target. Existing
  targets are immutable; regeneration uses a new output directory. Warehouse
  access requires an explicit full-run flag and local fixtures are bounded.
  ORCID counts now use one direct JSON file plus one provenance manifest, with
  no pointer manifest, retry loop, or legacy fallback. The large name-count
  index belongs in the immutable external data release, not Python package
  data. The code-only checkout declares neither ORCID file. The approved
  canonical JSON and manifest are added with both package-data declarations in
  one Stage 1 promotion commit.
- Generated `within_block_random` pair sampling now uses exact seeded rank
  sampling. It preserves the legacy candidate order, selected pairs, and labels
  while memory scales with requested samples plus blocks instead of all
  candidate pairs. Fixed-pair CSV datasets remain fully loaded and therefore
  still require the bounded pre-sampled smoke root in B22.
- Fixed train/validation/test CSV inputs now reject any unordered pair that
  appears in more than one split. B11 still requires this schema, duplication,
  and overlap validation to run during preflight before expensive
  featurization.
- Pairwise bundles and linker artifacts validate in sibling staging
  directories and publish with one rename into a new path; finalization never
  mutates the pairwise source in place. Metric promotion rejects
  missing/nonfinite values, and diagnostic metric-drift overrides cannot
  promote artifacts. Wheel/sdist validation rejects undeclared production
  assets and, when a default is declared, requires exactly that bundle.
  Production training records the canonical name-tuple and ORCID prefix-count
  data SHA-256 values in the feature contract; bundle export and load require
  exact matches, and the linker binding covers both through its ordered
  feature-contract digest. The production-bundle schema is version 5 and the
  clusterer-config schema is version 5; older bundles are rejected rather than
  adapted. Historical commit `e54c6ba` documents the published v1.21 loader's
  explicit clustering-threshold override to `0.65` for versions `1.2`/`1.21`;
  the stored threshold is stale. The current canonical loader has no legacy
  override, so a v1.21 baseline must use that compatible historical runtime.
  Newly tuned canonical bundles use their own `clusterer.json` value.
- The release workflow can build and install Python and Rust wheel candidates
  outside the source tree. A synthetic
  `Clusterer.predict_incremental_from_arrow_paths` smoke has passed from the
  installed wheels and itself carries the strict canonical Arrow manifest.
  Rust-enabled CI fails hard on import/ABI drift, and Windows/macOS jobs execute
  their built wheels. The Rust build-system floor and release action are aligned
  at Maturin 1.14.1. The current publish controls are not authorized for v1.3:
  B26 still requires one digest-pinned evidence archive, protected release gate,
  and publication of the exact reviewed candidate bytes, and B16 requires a
  real external v1.3 bundle smoke.
- Arrow training iterates record batches and avoids duplicate full-table
  materialization. Paper-author inputs reject duplicate positions, empty names,
  and dangling references consistently in Python and Rust. Python subblocking
  rejects duplicate IDs with explicit runtime invariants.
- Canonical-v2 removes legacy pickle/count-dictionary loading and redundant
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
- Production linker finalization trains the final booster on train plus weighted calibration splits, then calibrates the final logistic gate on the held-out test split.
- Incremental name compatibility now accepts joined and first-token aliases in addition to exact first-name tuples.
- Artifact cache entries are keyed by validator type. Raw-ETag cache filenames are no longer probed.
