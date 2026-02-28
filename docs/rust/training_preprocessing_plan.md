# Training-Mode Rust Paper Preprocessing

Status (2026-02-28):
- Bundle 1 implementation is landed in the current working tree and locally verified.
- Bundle 2 dependency (Rust->LightGBM cleanup boundary) is now ported in production training scripts
  (`scripts/transfer_experiment_seed_paper.py`).

Execution update (local evidence, 2026-02-28):
- Build + core parity slice passed:
  - `uv run maturin develop -m s2and_rust/Cargo.toml --release`
  - `uv run pytest -q tests/test_rust_from_dataset_contract.py tests/test_preprocess_papers_parallel_defaults.py tests/test_rust_lifecycle.py tests/test_rust_capabilities.py`
- Transfer gate artifact:
  - `scratch/profile_transfer_mini_bundle1_4_20260228.json`
  - python `296.597s` / `5.491 GB`; rust `176.469s` / `4.794 GB`; quality triplets unchanged at shown precision
- Phase-0 microbench artifact:
  - `scratch/bench_paper_preprocess_bundle1_20260228.log` (kisti preprocess pool benchmark replay)

## Problem

In inference mode, `from_json_paths` does all paper preprocessing natively in Rust:
`normalize_text`, `get_text_ngrams` (words + chars), `detect_language`.

Important nuance: `from_json_paths` parallelizes the heavy normalization/ngram work via Rayon
inside a `py.allow_threads` phase, but language detection is performed in the serial extraction
loop (mirrors the current implementation and avoids thread-safety questions around the FastText model).
Python paper preprocessing is completely skipped (`skip_python_paper_preprocess=True`).

In training mode, `from_dataset` reads pre-computed Python `Paper` namedtuples and
extracts fields directly — it does zero paper preprocessing. Python must run
`preprocess_paper_1` first, which is GIL-bound string work (unidecode, regex, Counter
construction) that doesn't parallelize well on any platform:

- Windows: serial is fastest (threads add GIL contention, spawn overhead wipes out
  process gains). See `docs/preprocessing_parallelism.md`.
- Linux: processes get ~1.4x with 8 workers (limited by pickle serialization of Counter
  return values through pipes).

Meanwhile, Rust already has all the preprocessing functions and runs the heavy normalization/ngram
work in parallel via Rayon in `from_json_paths`. The code exists — it's just not reachable from
`from_dataset`.

## Scale of the gap

`preprocess_papers_parallel` timing on kisti (36k papers):
- Windows serial: 10.7s
- Linux processes x8: 8.7s (best case)

In end-to-end `transfer-mini` runs on Windows where `n_jobs>1` is used, this stage is often
*slower than the serial microbenchmark* because `UniversalPool` selects threads on Windows and
the work is GIL-bound. A concrete example (kisti) shows:

- `Telemetry stage: stage=anddata_preprocess_papers seconds=13.323 papers=36447`
  (`scratch/profile_transfer_mini_phase0_memacc_20260225_105000_0b3e877.log`).

For larger datasets (aminer: 157k papers), this is proportionally longer and becomes a
meaningful fraction of training wall time, especially when repeated across multiple
datasets in transfer experiments.

Additional large-dataset evidence (inspire, 100k-signature slice; captured 2026-02-28):

- Command:
  `uv run --with psutil python scripts/rust_suite.py compare --mode compare --dataset inspire --limit 100000 --pair-count 1000 --n-jobs 4 --require-rust-release 1 --write-json scratch/compare_inspire_100k_anddata_build.json`
- Artifact: `scratch/compare_inspire_100k_anddata_build.json`
- `anddata_build_seconds`:
  - python: `316.705s`
  - rust: `33.781s`
  - delta: `-89.33%` (`9.38x` faster build)
- Same run totals:
  - python total: `317.503s`, peak RSS: `16.267 GB`
  - rust total: `71.615s`, peak RSS: `9.563 GB`

## Where this fits in the current optimization execution plan

This item is intended to land after (or at least be evaluated alongside) the current low-risk
wins tracked in `docs/rust/roadmap.md`:

1. **L1b** (cleanup boundary before LightGBM/hyperopt) — reduces risk that increased
   `from_dataset` allocations worsen post-Rust Python-only stages.
2. **P1 (3a)** (constraint lookup improvements) — reduces per-pair overhead in clustering/eval.
3. **Training paper preprocessing defer** (this doc) — highest payoff, highest effort.

You do **not** need to wait for all of L1b/P1 to finish to start the *design + contract-test*
work here, but you should validate “net win” on the maintained gate artifacts
(`scripts/rust_suite.py transfer-mini`) before promoting any new default behavior.

## Options considered

### Option A: Standalone Rust paper preprocessing function

Export a new `preprocess_papers_rust(papers_dict) -> papers_dict` that takes raw papers
from Python, preprocesses them in Rust, and returns preprocessed Paper namedtuples back
to Python.

Pros:
- Clean API boundary.
- Python training code stays mostly unchanged (swap one function call).

Cons:
- Requires a new FFI roundtrip: Python dict -> Rust -> processed Python dict.
- Serialization cost of passing Counter objects back through PyO3 may negate the speedup
  (same pickle problem as ProcessPoolExecutor, but through FFI instead of pipes).
- New API surface to maintain alongside `from_dataset` and `from_json_paths`.

### Option B: Training writes JSON, uses `from_json_paths`

Have the training path write papers/signatures to temporary JSON files, then call
`from_json_paths` instead of `from_dataset`.

Pros:
- Reuses the fully-optimized inference path with zero new Rust code.
- Gets Rayon parallelism, language detection, and all preprocessing for free.

Cons:
- Requires serializing the entire dataset to disk (or tmpfs) as JSON.
- `from_json_paths` was designed for the inference contract (raw JSON files at known
  paths). Training datasets are often constructed programmatically in Python and don't
  naturally exist as JSON files.
- Awkward for interactive/notebook workflows where data is already in memory.
- Breaks the mental model: training "writes to disk then reads back" is surprising.

### Option C: Extend `from_dataset` to preprocess papers when fields are None (recommended)

When `from_dataset` reads a `Paper` NamedTuple and finds preprocessing-dependent fields are
missing, compute them in Rust from the raw strings (`title`, `venue`, `journal_name`,
`authors[*].author_name`).

This mirrors the existing inference JSON-ingest shape: `from_json_paths` already buffers raw
inputs under the GIL, then runs normalization + ngram computation in `py.allow_threads` using
Rayon.

Pros:
- Follows the existing precedent in `from_dataset`: signatures already defer missing fields to
  Rust.
- No new API surface. No JSON roundtrip. No Python↔Rust Counter roundtrips.
- Lets training skip `preprocess_papers_parallel` entirely (the bottleneck documented in
  `docs/preprocessing_parallelism.md`).
- Enables Rayon parallelism **if** paper preprocessing is implemented in an `allow_threads +
  par_iter()` phase (the current `from_dataset` paper loop is serial under the GIL).
- Incremental: can be done field-by-field with parity checks at each step.

Cons:
- `from_dataset` becomes "smarter" — it now does conditional work rather than being a
  pure reader. This is already true for signatures, so the precedent exists.
- Requires restructuring `from_dataset` paper ingest: today it extracts papers serially and
  stores paper authors before any parallel step.
- Must preserve **paper author normalization**: signature deferred-field logic derives coauthor
  sets/blocks/ngrams from `Paper.authors`, and Python normalizes `Author.author_name` in
  `preprocess_paper_1`. Skipping Python paper preprocessing without normalizing authors in Rust
  will drift.
- Must preserve `predicted_language` + `is_reliable` semantics; leaving them unset changes both
  feature values and constraint behavior.
- Must be stage-safe: only skip Python paper preprocessing when downstream stages won’t need
  Python-preprocessed paper fields (i.e. Rust pair featurization is active, and
  `compute_reference_features=False` unless we also port reference preprocessing).

## Recommended approach: Option C

### Paper preprocessing contract (Python ground truth)

`preprocess_paper_1` currently does:

- If `paper.in_signatures`: `detect_language(paper.title)` → sets `predicted_language`,
  `is_reliable` (and `is_english`, unused in Rust features).
- Always: `normalize_text(title)` + `get_text_ngrams_words(title)` and normalizes every
  `Author.author_name`.
- If `preprocess=True`: normalizes `venue`/`journal_name` (regardless of `in_signatures`).
  If `preprocess=True` and `paper.in_signatures`: computes `title_ngrams_chars`, `venue_ngrams`,
  `journal_ngrams` via `get_text_ngrams(..., use_bigrams=True)`.

`preprocess_paper_2` is a separate second pass that builds `reference_details` and is gated by
`compute_reference_features`.

### What changes in Rust (`s2and_rust/src/lib.rs`)

**Key correction:** `from_dataset` currently extracts papers in a serial loop under the GIL, and
only signatures have an `allow_threads + Rayon` compute phase. If we compute paper ngrams “inline”
in the current paper loop, we will *not* get Rayon speedups.

To get the same shape as `from_json_paths`, restructure paper handling into two phases:

1) **Extract under the GIL (serial):**
   - Read dataset-level resources needed for paper preprocessing (matching `from_json_paths`):
     - `STOPWORDS` and `VENUE_STOP_WORDS` from `s2and.text`
     - `unidecode` from `s2and.text` for building `unidecode_char_map`
   - Extract existing Paper fields as today:
     `title_ngrams_words`, `title_ngrams_chars`, `venue_ngrams`, `journal_ngrams`,
     `predicted_language`, `is_reliable`, `authors`, `references`, `year`, `has_abstract`, `paper_id`.
   - Detect whether a paper needs deferred preprocessing:
     - Any ngram field is `None` **or**
     - Any paper-author name needs normalization (training-mode raw `Paper` objects have unnormalized
       authors) **or**
     - `predicted_language` is `None`.
     - **Note:** do **not** treat `is_reliable=False` as “missing”. `detect_language(...)` legitimately
       returns `is_reliable=False` for unknown/ambiguous cases, and `predicted_language is None` is the
       reliable marker for “not computed yet”.
   - For papers that need work, buffer a `PaperInput` containing raw strings:
     `raw_title`, `raw_venue`, `raw_journal_name`, `raw_authors`.
   - Build/extend `unidecode_char_map` using `ensure_unidecode_for_text(...)` over these raw strings.

   **NamedTuple fast-path note:** `from_dataset` currently validates and uses a Paper fast-path that
   only checks indices for *preprocessed* fields. If we want to keep tuple indexing for raw strings
   too, add constants for the raw indices (`title`, `venue`, `journal_name`) and include them in
   `PAPER_FASTPATH_REQUIRED_FIELDS`. Otherwise, fall back to attribute access for the raw strings
   when needed.

2) **Compute in `py.allow_threads` (parallel):**
   - Run `paper_inputs.par_iter()` inside `install_with_optional_rayon_pool(num_threads, ...)`.
   - For each buffered paper, compute missing fields using the same helpers as `from_json_paths`:
     - `normalize_text_compat_from_map`
     - `word_ngrams_counter_python_compat` for title word ngrams
     - `char_ngrams_counter_python_compat` for title/venue/journal char ngrams (with the correct
       stopword set: `STOPWORDS` for title chars, `VENUE_STOP_WORDS` for venue/journal)
   - Normalize `raw_authors[*].author_name` with `normalize_text_compat_from_map` and return the
     normalized author list so `paper_authors_by_id` matches the Python-preprocessed contract.

3) **Language fields (must be handled):**
   - In training-mode raw Papers, `predicted_language`/`is_reliable` start as `None`.
   - If we skip Python `preprocess_paper_1`, Rust must populate these to preserve parity.
   - Rust already has `LanguageDetectorCompat` used by `from_json_paths`. It is Rust-native (FastText
     crate + CLD2 crate); it does **not** call back into Python for detection.
   - Practical sequencing: follow `from_json_paths` and run `detector.detect(&raw_title)` in the
     serial extraction phase (simpler if the FastText model isn’t `Sync`), while keeping the heavy
     ngram/normalize work parallel.

4) **`dataset.preprocess` parity:**
   - `from_dataset` currently does not read `dataset.preprocess`.
   - For full parity with `preprocess_paper_1`, the deferred paper logic should mirror the Python
     contract (always compute title word ngrams + normalize authors; only compute venue/journal and
     char ngrams when `preprocess=True`).

5) **Memory scaling (prefer chunked compute):**
   - Avoid buffering `PaperInput` for *all* papers at once on very large datasets.
   - Prefer processing deferred papers in bounded chunks (by paper-count or estimated raw-string bytes),
     each chunk computed in `allow_threads + Rayon`, then merged into the Rust-side `papers` map.
   - If chunking is deferred initially for simplicity, at minimum record peak RSS deltas on
     `transfer-mini` and treat meaningful regressions as a blocker.

**Implementation note:** `extract_counter(...)` already returns `Option<CounterData>` and treats
`None`/empty dict as `Ok(None)`, so no new `extract_counter_opt` helper is required.

### What changes in Python

**`s2and/rust_lifecycle.py`:**

Expand the existing `skip_python_paper_preprocess` policy to cover training-mode `from_dataset`
builds **only when** the Rust extension supports deferred paper preprocessing in `from_dataset`.

Guard options:
- Version-gate: only enable when `s2and_rust.__version__` is at/above the first version that
  includes the feature.
- Capability marker: add an explicit API marker on `RustFeaturizer` and include it in
  `s2and/rust_capabilities.py` checks.

Also gate on:
- `compute_reference_features=False` (until reference preprocessing is ported or otherwise handled).
- Rust pair featurization enabled (stage-safe; don’t skip Python paper preprocessing if Python is
  going to consume those fields).
- Optional initial rollout gate: `preprocess=True` (start narrow; once parity is proven, consider also
  enabling skip for `preprocess=False`, since Python still normalizes title/authors + computes title word ngrams).

**`s2and/data.py`:**

When `skip_python_paper_preprocess=True`, skip `preprocess_papers_parallel` entirely.
Pass raw (unpreprocessed) Paper objects to `from_dataset`. Rust handles the rest.

This is a one-line change in the training path — the lifecycle policy already gates
whether Python preprocessing runs.

### What about `preprocess_paper_2`?

`preprocess_paper_2` builds `reference_details` (reference-derived Counters + block keys) and is
gated by `compute_reference_features`.

This plan does **not** port reference preprocessing to Rust. Until it does, the safest policy is:
only skip Python paper preprocessing when `compute_reference_features=False`.

## Language detection: the tricky part

`from_json_paths` already handles language detection by instantiating `LanguageDetectorCompat`:
it is Rust-native (FastText crate + CLD2 crate) and returns `(is_reliable, is_english, language)`.
Python is only used to resolve the FastText model path (via `s2and.file_cache.cached_path`) at
construction time.

```rust
// lib.rs ~line 3016
let detector = LanguageDetectorCompat::new(py);
// lib.rs ~line 3185
let (is_reliable, _is_english, language) = detector.detect(&raw_title);
```

For `from_dataset`, we need the same behavior. Two sub-options:

1. **Compute language in Rust (recommended for full skip):**
   - Run `detector.detect(&raw_title)` in the serial paper-extraction phase (mirrors
     `from_json_paths` and avoids any thread-safety questions around the FastText model).
   - Keep normalization + ngram computation in the parallel `allow_threads + Rayon` phase.

2. **Keep language detection in Python (fallback / partial skip):**
   - Python runs a lightweight pass to fill `predicted_language`/`is_reliable` only.
   - Rust still computes ngrams + author normalization for missing fields.
   - This is simpler if language parity is problematic, but it is not a complete removal of
     Python preprocessing.

Either way, language detection is a small fraction of the wall time compared to normalization +
ngram computation, so keeping it serial is acceptable if needed.

## Verification plan

1. **Parity test:** Create a test that runs the same dataset through both paths:
   - Build A: Python-preprocessed papers (`preprocess_papers_parallel`) → `from_dataset`
   - Build B: raw papers (no paper preprocessing; all paper ngram fields `None`) → `from_dataset`
   - Assert: `RustFeaturizer.featurize_pair(...)` outputs are identical for representative pairs.
   - Include at least one case that exercises coauthor-derived features and language-derived features.

2. **Regression gate:** Existing `test_feature_port_parity.py` and
   `test_rust_from_json_paths.py` must still pass (they exercise `from_dataset` with
   pre-computed fields).

3. **Performance measurement:** Benchmark kisti and aminer training with Python
   preprocessing vs Rust deferred preprocessing. Measure:
   - Total training wall time
   - `from_dataset` build time (expect slight increase since it now does preprocessing)
   - `preprocess_papers_parallel` time (expect elimination)
   - Peak RSS

4. **Quality gate:** Same thresholds as existing Rust alignment (from
   `docs/normalization_migration.md`):
   - Pairwise: AUC delta <= 0.001, F1 delta <= 0.005
   - Clustering: B3 delta <= 0.005

## Practical next steps (implementation + measurement)

This section is the “do this next” checklist, with concrete evidence artifacts to collect.
The intent is to make this project easy to pursue (or abandon) based on measured deltas.

### Phase 0 — Record “before” artifacts (no code changes)

1) **Paper-preprocess baseline (Python-only):**
   - Run: `uv run --no-project python scripts/bench_preprocess_phases.py --dataset kisti --limit-signatures 0 --skip-paper2 --skip-signatures`
   - Record: platform, dataset, n_jobs, and the best timings (Windows serial vs Linux processes).

2) **Train/eval baseline (recommended primary gate):**
    - Build release extension: `uv run maturin develop -m s2and_rust/Cargo.toml --release`
    - Run maintained compare:
      `uv run --with psutil python scripts/rust_suite.py transfer-mini --mode compare --preset full --n-jobs 4 --n-train-pairs 10000 --n-iter 5 --target kisti --require-rust-release 1 --write-json scratch/profile_transfer_mini_before_paper_defer.json`
    - Evidence: JSON artifact contains `total_seconds`, `peak_rss_gb`, and clustering metrics.

Optional stability runs (perf-noise control):
- For benchmarks that complete in <5 minutes (typically `compare` and the paper-preprocess microbench),
  run 3× and report the median.

Optional but useful:
- Inference comparator sanity (should be unaffected, but keeps confidence high):
  `uv run --no-project python scripts/rust_suite.py compare --mode compare --dataset inspire --limit 5000 --pair-count 5000 --n-jobs 8 --require-rust-release 1 --write-json scratch/compare_before_paper_defer.json`

### Phase 1 — Implement behind a capability gate

1) **Rust: deferred paper preprocessing in `from_dataset`:**
   - Implement the two-phase design described above:
     - Extract raw strings + existing fields under the GIL (serial).
     - Compute normalization + ngrams + author normalization in `py.allow_threads` with Rayon.
     - Populate `predicted_language`/`is_reliable` using `LanguageDetectorCompat` when missing.
     - Respect `dataset.preprocess` (mirror Python contract).

2) **Python: lifecycle + stage-safety gating:**
   - Add a capability marker (or version threshold) that specifically represents:
     “`from_dataset` can safely fill missing Paper preprocessing fields + normalize authors.”
   - Only set `skip_python_paper_preprocess=True` when:
     - Rust pair-featurization is enabled (so Python won’t consume paper ngrams),
     - `compute_reference_features=False` (until reference preprocessing is ported),
     - optionally `preprocess=True` for the initial rollout (see note above),
     - and the Rust capability marker is present.

### Phase 2 — Parity tests (must land with the behavior change)

Add a regression test that proves:
- raw papers (no Python paper preprocessing) → `RustFeaturizer.from_dataset` produces identical features
  to the existing Python-preprocessed-paper path.

Minimum test coverage:
- A case where `Author.author_name` normalization changes the coauthor-derived features.
- A case where language fields are material (`predicted_language`, `is_reliable`) and affect both
  features and constraints.

Commands:
- Fast local slice: `uv run pytest -q tests/test_rust_from_dataset_contract.py`
- Core parity slice: `uv run pytest -q tests/test_feature_port_parity.py tests/test_rust_from_json_paths.py`

### Phase 3 — Measure deltas and decide go/no-go

Run the same gates as Phase 0, producing “after” artifacts:

1) `scripts/bench_preprocess_phases.py` (optional re-run; the goal is that training no longer
   pays this stage at all).
2) `transfer-mini` compare:
   - `uv run --with psutil python scripts/rust_suite.py transfer-mini --mode compare --preset full --n-jobs 4 --n-train-pairs 10000 --n-iter 5 --target kisti --require-rust-release 1 --write-json scratch/profile_transfer_mini_after_paper_defer.json`
3) Inference comparator (optional): `scratch/compare_after_paper_defer.json`

Optional gate (baseline vs current):
- `uv run --with psutil python scripts/rust_suite.py transfer-mini --mode gate --baseline-json scratch/profile_transfer_mini_before_paper_defer.json --current-json scratch/profile_transfer_mini_after_paper_defer.json --gate-run-label rust`

What to record (minimum evidence table):

- **Eliminated Python paper preprocessing time**
  - Source: `Telemetry stage: stage=anddata_preprocess_papers seconds=...` log line.
  - Expectation: near-zero in Rust runs where the skip is enabled.
- **Shifted Rust featurizer build cost**
  - Source: `Telemetry: rust_core_build ... pre=... ffi=... post=...` log line.
  - Expectation: `ffi_seconds` increases (paper preprocessing moved into the Rust build).
- **End-to-end training wall time**
  - Source: `scratch/profile_transfer_mini_*` JSON `total_seconds`.
  - Expectation: net improvement (target: ≥5–10% on kisti-scale workloads; otherwise deprioritize).
- **Peak RSS**
  - Source: `scratch/profile_transfer_mini_*` JSON `peak_rss_gb` and `stage_rss_gb`.
  - Expectation: no meaningful regression (prefer flat or improved; investigate if +>~5%).
- **Quality parity**
  - Source: `transfer-mini` JSON (B3 / cluster / cluster_macro) + existing parity tests.
  - Expectation: identical at displayed precision; otherwise block on parity investigation.

Stop criteria (deprioritize this project if any holds):
- End-to-end transfer-mini improvement is <5% and featurizer build gets meaningfully slower.
- Any parity drift in coauthor or language-derived features/constraints.
- Peak RSS regresses beyond the repo’s standard non-regression tolerance without a clear mitigation.

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Ngram parity drift between Python and Rust implementations | Medium | `from_json_paths` helpers already parity-tested; add raw→computed `from_dataset` parity test |
| Paper-author normalization drift (coauthor features) | Medium | Normalize `Author.author_name` in Rust deferred paper path; include coauthor-focused parity cases |
| Paper preprocessing remains serial (no speedup) | Medium | Implement the two-phase (`extract` then `allow_threads + Rayon`) design; avoid heavy compute in the GIL loop |
| Peak RSS spike from buffering raw paper strings | Medium | Prefer chunked deferred-paper compute (bounded chunks); validate via `transfer-mini` peak RSS + stage RSS snapshots |
| Increased `from_dataset` allocations worsen post-Rust Python-only stages (L1b) | Medium | Land/validate the L1b cleanup boundary and reprofile `transfer-mini` (watch `union_*_fit_seconds` deltas + RSS snapshots) |
| Language detection drift | Low | Reuse `LanguageDetectorCompat` (same as `from_json_paths`); gate/monitor via parity tests |
| Stage-safety violation (Python featurizer still needs preprocessed fields) | Medium | Only set `skip_python_paper_preprocess` when Rust pair featurization is active |
| `compute_reference_features=True` produces missing/incorrect `reference_details` | Medium | Gate skip to `compute_reference_features=False` until reference preprocessing is ported |
| `from_dataset` becomes harder to reason about (conditional vs pure reader) | Low | Precedent exists for signatures; document the contract |
| Old Rust extension + new Python that skips preprocessing = broken | Medium | Version-gate: `skip_python_paper_preprocess` only when Rust version >= threshold |
| Normalization migration interaction | Low | This is a pure-performance change; normalization policy is unchanged (same functions, same outputs). Revisit when normalization canonical cutover happens. |

## Scope and non-goals

In scope:
- Defer paper ngram computation (title_ngrams_words, title_ngrams_chars, venue_ngrams,
  journal_ngrams) and normalize paper authors to Rust in `from_dataset`.
- Update lifecycle policy to skip Python `preprocess_papers_parallel` when Rust handles it.
- Parity + performance tests.

Not in scope (future work):
- `preprocess_paper_2` / `reference_details` preprocessing in Rust.
- Signature preprocessing changes (already handled).
- Normalization policy changes (tracked separately in `docs/normalization_migration.md`).

## Dependency on existing work

- Requires all preprocessing helper functions (`normalize_text_compat_from_map`,
  `word_ngrams_counter_python_compat`, `char_ngrams_counter_python_compat`) to already
  exist in Rust and be parity-validated. They do — they're used by `from_json_paths`.
- Requires access to `STOPWORDS` and `VENUE_STOP_WORDS` in `from_dataset` (already present in
  `s2and.text`; already used by `from_json_paths`).
- No hard dependency on L0–L6 lifecycle work or P0–P4 optimizations, but note that making
  `from_dataset` do more work increases the value of correct featurizer reuse (L0) on workloads
  that build repeatedly.
- No dependency on normalization migration (same functions, same behavior).
