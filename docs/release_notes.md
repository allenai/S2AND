# Release Notes

## 0.60.0

- **Unreleased migration state:** the artifact-independent canonical-v2 code
  and release hardening are implemented, but canonical name counts, canonical
  ORCID prefix counts, benchmark-name re-export, the v1.3 retrain, and
  release-candidate quality/scale measurements are still pending. The legacy
  v1.21/v1.0-v1.2 models are not packaged and are rejected by the canonical
  loader, so 0.60.0 is not yet a usable production release. See
  [work_plan.md](work_plan.md).
- Breaking: Python and Rust now share one `canonical_v2` name contract and one
  versioned feature contract. `FEATURIZER_VERSION` is 10. Titles retain letters
  and digits, CLD2 runs in explicit plain-text mode, malformed email is missing
  evidence, query-author text uses canonical fields only, and incremental
  six-decimal values use ties-to-even rounding. Deterministic parity is pinned
  at `1e-6` (exact for discrete/count/boolean features).
- Breaking: runtime legacy-normalization shims, Sinonym rewriting, fastText,
  and reference features are removed. `s2and-rust` is a required runtime
  dependency. Production Rust featurization and ingest enter through validated
  Arrow IPC artifacts; classic JSON/`ANDData` remains a Python compatibility
  and reference surface.
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
- Name-count pickle, binary index, Arrow inputs, pairwise boosters, and linker
  metadata now carry and verify normalization, generation, size, SHA-256, and
  feature-contract provenance. Manifest-relative paths cannot escape their
  authority or depend on process CWD. Equal-size/equal-mtime mutation of
  altered-profile/disallow inputs is detected before altered-presplit reuse.
  Models selecting name-count features
  compare the exact four-field generation binding at Python, Arrow, and
  prebuilt-Rust-featurizer boundaries before feature work.
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
- Name-count source and binary-index publication remain two sequential,
  individually atomic generation publishes. A crash between them can leave a
  detected binding mismatch; rerunning with `--overwrite` repairs it. Warehouse
  access requires an explicit full-run flag and local fixtures are bounded.
  ORCID counts now use one direct JSON file plus an adjacent metadata sidecar,
  with no pointer manifest, cross-process lock, fsync protocol, retry loop, or
  legacy fallback. Both paths remain excluded from distributions until the
  approved canonical generation replaces the checked-in legacy JSON.
- Pairwise bundles and linker artifacts validate in sibling staging
  directories and publish with one rename into a new path; finalization never
  mutates the pairwise source in place. Metric promotion rejects
  missing/nonfinite values, and diagnostic metric-drift overrides cannot
  promote artifacts. Wheel/sdist validation rejects undeclared production
  assets and, when a default is declared, requires exactly that bundle.
  Production training records the canonical name-tuple and ORCID prefix-count
  data SHA-256 values in the feature contract; bundle export and load require
  exact matches, and the linker binding covers both through its ordered
  feature-contract digest. The production-bundle and clusterer-config schemas
  are version 4; older bundles are rejected rather than adapted.
- The release workflow builds and installs the exact Python and Rust wheels
  outside the source tree. A synthetic public
  `Clusterer.predict_incremental_from_arrow_paths` smoke has passed from the
  installed wheels and itself carries the strict canonical Arrow manifest.
  Rust-enabled CI fails hard on import/ABI drift, and Windows/macOS jobs execute
  their built wheels. The Rust build-system floor and release action are aligned
  at Maturin 1.14.1. A real declared-bundle smoke remains an external gate on
  the new v1.3 artifacts.
- Arrow training iterates record batches and avoids duplicate full-table
  materialization. Paper-author inputs reject duplicate positions, empty names,
  and dangling references consistently in Python and Rust. Python subblocking
  rejects duplicate IDs with explicit runtime invariants.
- Canonical-v2 removes legacy pickle/count-dictionary loading and redundant
  artifact plumbing. Bounded native name-count lookups, optimized float32
  scoring, and deterministic publication preserve output hashes while meeting
  the recorded memory and throughput gates.

## 0.51.1

- Ships the package as `0.51.1` and pins optional Rust installs to `s2and-rust==0.51.1`.
- Makes Arrow IPC the production Rust runtime boundary. Direct prediction now uses `Clusterer.predict_from_arrow_paths(...)` or Arrow-routed `Clusterer.predict(...)` over `signatures`, `papers`, `paper_authors`, selected `specter`, raw-planner batch indexes, and shared `name_counts_index`. JSON/`ANDData` remains available for compatibility, training, fixtures, and parity checks.
- Adds canonical Arrow runtime contracts and tooling: `s2and.arrow_inputs`, `s2and/arrow_schema_contract.json`, `scripts/convert_to_arrow.py`, `scripts/arrow_conversion_helpers.py`, local Arrow release validation, and bounded parity/quality verification scripts. The documented public data release is now `s2and-release-arrow`; the legacy JSON/pickle release remains for paper-era inputs.
- Tightens production validation. Missing or malformed Arrow artifacts now raise structured `MissingArrowArtifactError` failures, Rust production routes fail fast instead of silently falling back to `ANDData`, unsupported name-alias path keys are rejected, and direct Arrow prediction refuses models that require reference features.
- Reworks promoted incremental linking around Arrow/Rust. The promoted path reads base Arrow artifacts, query signatures, cluster seeds, cluster seed disallows, and altered-profile sidecars; Rust performs raw candidate planning and promoted row-signal construction; `batching_threshold` controls promoted Rust query batch size.
- Switches promoted linker replay/training to `--feature-mode arrow-rust` by default against the canonical `s2and_and_big_blocks_linker_dataset_20260525` Arrow+labels bundle. `precomputed-promoted` remains an explicit reuse mode.
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
