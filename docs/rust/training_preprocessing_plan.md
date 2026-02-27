# Training-Mode Rust Paper Preprocessing

## Problem

In inference mode, `from_json_paths` does all paper preprocessing natively in Rust:
`normalize_text`, `get_text_ngrams` (words + chars), `detect_language` — all parallelized
via Rayon across papers. Python is completely skipped (`skip_python_paper_preprocess=True`).

In training mode, `from_dataset` reads pre-computed Python `Paper` namedtuples and
extracts fields directly — it does zero paper preprocessing. Python must run
`preprocess_paper_1` first, which is GIL-bound string work (unidecode, regex, Counter
construction) that doesn't parallelize well on any platform:

- Windows: serial is fastest (threads add GIL contention, spawn overhead wipes out
  process gains). See `docs/preprocessing_parallelism.md`.
- Linux: processes get ~1.4x with 8 workers (limited by pickle serialization of Counter
  return values through pipes).

Meanwhile, Rust already has all the preprocessing functions and runs them in parallel via
Rayon in `from_json_paths`. The code exists — it's just not reachable from `from_dataset`.

## Scale of the gap

`preprocess_papers_parallel` timing on kisti (36k papers):
- Windows serial: 10.7s
- Linux processes x8: 8.7s (best case)

For larger datasets (aminer: 157k papers), this is proportionally longer and becomes a
meaningful fraction of training wall time, especially when repeated across multiple
datasets in transfer experiments.

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

When `from_dataset` reads a Paper namedtuple and finds ngram fields are `None`, it
computes them from raw title/venue/journal strings — exactly like it already does for
signatures.

Pros:
- Follows the existing pattern in `from_dataset` (signatures already do this at
  `lib.rs:2707-2873`).
- No new API surface. No JSON roundtrip. No extra serialization.
- Python can skip `preprocess_paper_1` entirely and pass raw Paper objects.
- Rayon parallelism for the compute-heavy ngram/normalize work.
- Incremental: can be done field-by-field with parity checks at each step.

Cons:
- `from_dataset` becomes "smarter" — it now does conditional work rather than being a
  pure reader. This is already true for signatures, so the precedent exists.
- Language detection callout: `from_json_paths` calls Python's `detect_language` via
  `LanguageDetectorCompat` during preprocessing. `from_dataset` would need the same
  callout if `predicted_language` is None. This is the one field that calls back into
  Python (pycld2 via FFI).

## Recommended approach: Option C

### What changes in Rust (`s2and_rust/src/lib.rs`)

**In `from_dataset`, after extracting paper fields (~line 2500–2666):**

Currently:
```rust
let title_words = extract_counter(paper_tuple, PAPER_IDX_TITLE_NGRAMS_WORDS, ...)?;
let title_chars = extract_counter(paper_tuple, PAPER_IDX_TITLE_NGRAMS_CHARS, ...)?;
let venue_ngrams = extract_counter(paper_tuple, PAPER_IDX_VENUE_NGRAMS, ...)?;
let journal_ngrams = extract_counter(paper_tuple, PAPER_IDX_JOURNAL_NGRAMS, ...)?;
```

After:
```rust
let title_words = extract_counter_opt(paper_tuple, PAPER_IDX_TITLE_NGRAMS_WORDS, ...)?;
let title_chars = extract_counter_opt(paper_tuple, PAPER_IDX_TITLE_NGRAMS_CHARS, ...)?;
let venue_ngrams = extract_counter_opt(paper_tuple, PAPER_IDX_VENUE_NGRAMS, ...)?;
let journal_ngrams = extract_counter_opt(paper_tuple, PAPER_IDX_JOURNAL_NGRAMS, ...)?;

// If any ngram field is None, read raw strings and compute in Rust
let need_paper_preprocess = title_words.is_none()
    || title_chars.is_none()
    || venue_ngrams.is_none()
    || journal_ngrams.is_none();

if need_paper_preprocess {
    let raw_title = extract_string(paper_tuple, PAPER_IDX_TITLE, ...)?;
    let raw_venue = extract_string(paper_tuple, PAPER_IDX_VENUE, ...)?;
    let raw_journal = extract_string(paper_tuple, PAPER_IDX_JOURNAL_NAME, ...)?;

    // Normalize (reuse normalize_text_compat_from_map from from_json_paths)
    let title_norm = normalize_text_compat_from_map(&raw_title, ...);
    let venue_norm = normalize_text_compat_from_map(&raw_venue, ...);
    let journal_norm = normalize_text_compat_from_map(&raw_journal, ...);

    // Compute ngrams (reuse word_ngrams_counter_python_compat, char_ngrams_counter_python_compat)
    if title_words.is_none() {
        title_words = Some(word_ngrams_counter_python_compat(&title_norm, &stop_words));
    }
    if title_chars.is_none() {
        title_chars = Some(char_ngrams_counter_python_compat(&title_norm, ...));
    }
    if venue_ngrams.is_none() {
        venue_ngrams = Some(char_ngrams_counter_python_compat(&venue_norm, ...));
    }
    if journal_ngrams.is_none() {
        journal_ngrams = Some(char_ngrams_counter_python_compat(&journal_norm, ...));
    }
}
```

The Rayon parallelization happens naturally: this deferred compute can run inside the
existing per-paper loop that's already dispatched to the thread pool.

**Language detection:** Same pattern — if `predicted_language` is None, call back to
Python's `detect_language` via the existing `LanguageDetectorCompat` bridge. This is the
same mechanism `from_json_paths` already uses (it calls Python for language detection
too). The `LanguageDetectorCompat` needs to be plumbed into `from_dataset`, which
currently doesn't instantiate one.

**New raw-string Paper namedtuple indices:** `from_dataset` currently reads only
preprocessed fields. To read raw strings when needed, it needs access to the raw
`title`, `venue`, `journal_name` fields from the Paper namedtuple. These fields already
exist on the namedtuple — they're just not currently extracted in `from_dataset`. Add
extraction only when `need_paper_preprocess` is true.

### What changes in Python

**`s2and/rust_lifecycle.py`:**

Add a new policy flag: `skip_python_paper_preprocess_for_from_dataset`. When
`use_rust=True` and the Rust extension supports the new deferred-preprocess capability,
set this to True even for `from_dataset` builds.

Alternatively (simpler): just expand the existing `skip_python_paper_preprocess` flag to
cover `from_dataset` when the Rust extension version is new enough. Guard with a version
check so old Rust extensions that expect pre-computed fields still get them.

**`s2and/data.py`:**

When `skip_python_paper_preprocess=True`, skip `preprocess_papers_parallel` entirely.
Pass raw (unpreprocessed) Paper objects to `from_dataset`. Rust handles the rest.

This is a one-line change in the training path — the lifecycle policy already gates
whether Python preprocessing runs.

### What about `preprocess_paper_2`?

`preprocess_paper_2` (abstract word ngrams, detailed language stats) runs in a second
pass. It's only used when `preprocess=True` and papers have abstracts. This is a separate
concern and can remain in Python initially — it's not on the critical path for the same
reason (it's lighter work and less frequently triggered). Extend to Rust later if needed.

## Language detection: the tricky part

`from_json_paths` already handles this by instantiating `LanguageDetectorCompat` which
calls back to Python's pycld2:

```rust
// lib.rs ~line 3016
let detector = LanguageDetectorCompat::new(py);
// lib.rs ~line 3185
let (predicted_language, is_reliable) = detector.detect(&raw_title)?;
```

For `from_dataset`, the same bridge is needed. Two sub-options:

1. **Plumb `LanguageDetectorCompat` into `from_dataset`** — straightforward but requires
   the GIL for each `detect` call (it's a Python callout). This limits Rayon parallelism
   for the language detection step specifically. The ngram computation can still run in
   parallel; only the detection callout is serialized.

2. **Skip language detection in Rust, keep it in Python** — only defer ngram computation
   to Rust. Python runs `detect_language` as part of a lightweight first pass, then passes
   `predicted_language`/`is_reliable` on the Paper namedtuple. Rust only fills in the
   None ngram fields. This avoids the GIL callout issue entirely and still captures
   ~90%+ of the preprocessing CPU time (ngram computation dominates).

Sub-option 2 is simpler and lower risk. Language detection is ~5% of `preprocess_paper_1`
wall time; ngram computation and normalize_text are ~95%.

## Verification plan

1. **Parity test:** Create a test that runs the same dataset through both paths:
   - Path A: Python `preprocess_paper_1` -> `from_dataset` (current behavior)
   - Path B: raw Papers (ngrams=None) -> `from_dataset` with deferred compute
   - Assert: all extracted features are identical.

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

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Ngram parity drift between Python and Rust implementations | Medium | Already validated by `test_feature_port_parity.py`; add specific raw->processed parity test |
| Language detection GIL bottleneck | Low | Use sub-option 2 (keep detection in Python, only defer ngrams) |
| `from_dataset` becomes harder to reason about (conditional vs pure reader) | Low | Precedent exists for signatures; document the contract |
| Old Rust extension + new Python that skips preprocessing = broken | Medium | Version-gate: `skip_python_paper_preprocess` only when Rust version >= threshold |
| Normalization migration interaction | Low | This is a pure-performance change; normalization policy is unchanged (same functions, same outputs). Revisit when normalization canonical cutover happens. |

## Scope and non-goals

In scope:
- Defer paper ngram computation (title_ngrams_words, title_ngrams_chars, venue_ngrams,
  journal_ngrams) and normalize_text to Rust in `from_dataset`.
- Update lifecycle policy to skip Python `preprocess_papers_parallel` when Rust handles it.
- Parity + performance tests.

Not in scope (future work):
- `preprocess_paper_2` (abstract ngrams, detailed language stats).
- Language detection in Rust (keep in Python for now).
- Signature preprocessing changes (already handled).
- Normalization policy changes (tracked separately in `docs/normalization_migration.md`).

## Dependency on existing work

- Requires all preprocessing helper functions (`normalize_text_compat_from_map`,
  `word_ngrams_counter_python_compat`, `char_ngrams_counter_python_compat`) to already
  exist in Rust and be parity-validated. They do — they're used by `from_json_paths`.
- Requires `extract_counter_opt` (or equivalent None-tolerant extraction) for Paper
  fields. The pattern already exists for signature fields.
- No dependency on L0–L6 lifecycle work or P0–P4 optimizations (orthogonal).
- No dependency on normalization migration (same functions, same behavior).
