# Preprocessing parallelism (papers + signatures)

## Summary

S2AND has three main Python preprocessing phases that can dominate end-to-end runtime:

1. **Papers 1/2**: `preprocess_paper_1` (title/author normalization + word ngrams; and venue/journal normalization when `preprocess=True`)
2. **Papers 2/2**: `preprocess_paper_2` (reference-details ngrams + block counts)
3. **Signatures**: `ANDData.preprocess_signatures` (normalization + feature creation)

**Production default behavior (as of 2026-02-27):**

- **Linux / WSL2**: use a process pool for **Papers 1/2** when `n_jobs > 1`; run **Papers 2/2** serial; run **Signatures** serial.
- **Windows/macOS (native)**: run **Papers 1/2** serial (even if `n_jobs > 1`); run **Papers 2/2** serial; run **Signatures** serial.

Rationale (high level): `preprocess_paper_1` is CPU-bound and benefits from `fork` multiprocessing on Linux, while `spawn` platforms (Windows/macOS) pay heavy import/pickle overhead. For `preprocess_paper_2` and signature preprocessing, the overhead of shipping large Python objects around dominates, so pooling is net negative.

## Implementation notes

- `preprocess_paper_1` takes an explicit `preprocess=...` flag (spawn-safe; no worker globals).
- `preprocess_papers_parallel` uses `UniversalPool` only for the **Papers 1/2** phase on Linux; **Papers 2/2** always runs serial.
- `UniversalPool` remains platform-aware when used elsewhere: processes on Linux (`fork`), threads on Windows/macOS by default.

## Benchmark script

`scripts/bench_preprocess_phases.py` benchmarks the three phases separately.

## Rust `from_dataset` bypass (Bundle 1)

When Rust deferred paper preprocessing is active (Bundle 1 in `docs/work_plan.md`),
`preprocess_papers_parallel` is **skipped entirely** in training mode. Rust's `from_dataset`
handles paper normalization, ngrams, and language detection natively via Rayon parallelism,
making the Python parallelism discussion above moot for Rust-enabled training runs.

Conditions for the bypass:
- Rust extension supports `SUPPORTS_FROM_DATASET_PAPER_PREPROCESS`
- Rust pair featurization is enabled
- `compute_reference_features=False`

See `docs/rust/training_preprocessing_plan.md` for full details and verification plan.

## Date

2026-02-28
