# Threading and parallelism

S2AND uses multiple libraries that can each create their own thread pools (Rust Rayon, LightGBM/OpenMP, BLAS, etc.).
If those pools are configured independently, runs can oversubscribe CPU cores and show higher-than-expected CPU usage.

This doc describes the intended “single knob” behavior and the practical rules to keep thread counts predictable.

## Single knob: `n_jobs`

Within the Python API, treat `n_jobs` as the canonical concurrency setting for a run:

- **Rust backend**: Python passes `num_threads=n_jobs` into the Rust extension for batch constraints + featurization.
- **LightGBM inference**: `Clusterer.n_jobs` propagates into the underlying estimators, and prediction passes
  `num_threads=n_jobs` when supported.
- **Python preprocessing**: `ANDData(n_jobs=...)` controls the limited (and platform-dependent) pooling used in some
  preprocessing phases.

## Recommended run setup

1. **Pick one `n_jobs`** and use it everywhere:
   - `ANDData(..., n_jobs=N)`
   - `clusterer.n_jobs = N` (or pass `n_jobs=N` when constructing `Clusterer`)

2. **Set thread env vars before importing compute-heavy libraries** (especially on Windows):
   - `OMP_NUM_THREADS=N` (OpenMP; LightGBM, some clustering libs)
   - `MKL_NUM_THREADS=N`, `OPENBLAS_NUM_THREADS=N`, `NUMEXPR_NUM_THREADS=N` (if your NumPy/SciPy stack uses them)
   - Optional: `RAYON_NUM_THREADS=N` (only affects Rust code that uses Rayon’s global pool; S2AND’s Rust extension
     primarily uses explicit `num_threads` arguments instead)

   Many OpenMP runtimes read environment variables at first use / first load; setting them after importing `lightgbm`
   is not reliable.

## Avoiding nested parallelism

Oversubscription usually comes from *stacking* an outer pool (threads/processes) on top of an inner thread pool
(Rayon/OpenMP/BLAS).

Rules of thumb:

- If you use an **outer process pool** with `W` workers, cap **inner threads** to ~`floor(N / W)` (often `1`) inside each
  worker.
- Prefer **one parallelism layer per phase**:
  - Rust batch featurization: Rayon handles parallelism; avoid wrapping it in additional thread pools.
  - LightGBM inference: let LightGBM/OpenMP use threads; avoid concurrent `predict_proba()` calls from multiple workers.

## Rust Rayon pool lifetime

The Rust extension caches Rayon thread pools by thread count for reuse. Those worker threads stay alive for the process
lifetime, even between calls. This should not consume CPU when idle, but it does mean “thread count” tools may show more
threads than expected.

If you need to guarantee that all worker threads fully exit between runs, use process boundaries (run each workload in a
fresh Python process).

## Date

2026-03-01

