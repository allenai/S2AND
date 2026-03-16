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
  preprocessing phases (see “Python preprocessing parallelism” below).

## Python preprocessing parallelism

S2AND has three main Python preprocessing phases that can dominate end-to-end runtime:

1. **Papers 1/2**: `preprocess_paper_1` (title/author normalization + word ngrams; and venue/journal normalization when `preprocess=True`)
2. **Papers 2/2**: `preprocess_paper_2` (reference-details ngrams + block counts)
3. **Signatures**: `ANDData.preprocess_signatures` (normalization + feature creation)

**Production default behavior (as of 2026-02-27):**

- **Linux / WSL2**: use a process pool for **Papers 1/2** when `n_jobs > 1`; run **Papers 2/2** serial; run **Signatures** serial.
- **Windows/macOS (native)**: run **Papers 1/2** serial (even if `n_jobs > 1`); run **Papers 2/2** serial; run **Signatures** serial.

Rationale (high level): `preprocess_paper_1` is CPU-bound and benefits from `fork` multiprocessing on Linux, while
`spawn` platforms (Windows/macOS) pay heavy import/pickle overhead. For `preprocess_paper_2` and signature preprocessing,
the overhead of shipping large Python objects around dominates, so pooling is net negative.

Implementation notes:

- `preprocess_paper_1` takes an explicit `preprocess=...` flag (spawn-safe; no worker globals).
- `preprocess_papers_parallel` uses `UniversalPool` only for the **Papers 1/2** phase on Linux; **Papers 2/2** always runs serial.
- Production `UniversalPool` call sites pass explicit `use_threads=...` so pool mode does not rely on implicit defaults.
- `UniversalPool` remains platform-aware for helpers/callers that do not pass `use_threads`: processes on Linux (`fork`), threads on Windows/macOS by default.

Benchmark script:

- `scripts/bench_preprocess_phases.py` benchmarks the three phases separately.

Rust `from_dataset` bypass (Bundle 1):

When training/eval runs use the Rust backend, paper preprocessing can be deferred to Rust (see
`docs/rust/runtime.md` section "Training-mode deferred paper preprocessing"). In that mode,
`preprocess_papers_parallel` is **skipped entirely**, and Rust’s `from_dataset` handles paper
normalization, ngrams, and language detection via Rayon parallelism, making the Python parallelism
discussion above moot for Rust-enabled training runs.

Conditions for the bypass:

- Backend resolves to Rust (`S2AND_BACKEND=rust` or `auto` resolved to Rust)
- `preprocess=True`
- Rust extension supports `SUPPORTS_FROM_DATASET_PAPER_PREPROCESS`
- `compute_reference_features=False`

## Recommended run setup

1. **Pick one `n_jobs`** and use it everywhere:
   - `ANDData(..., n_jobs=N)`
   - `clusterer.n_jobs = N` (or pass `n_jobs=N` when constructing `Clusterer`)

2. **Set thread env vars before importing compute-heavy libraries** (especially on Windows):
   - `OMP_NUM_THREADS=N` for OpenMP users such as LightGBM
   - Usually `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`
   - Optional: `RAYON_NUM_THREADS=N` (only affects Rust code that uses Rayon’s global pool; S2AND’s Rust extension
     primarily uses explicit `num_threads` arguments instead)

   Why are the BLAS / NumExpr knobs usually pinned to `1` while OpenMP / Rayon may be set to `N`?

   - In typical S2AND runs, the main parallel work is Rust featurization / constraint resolution and LightGBM inference.
   - MKL, OpenBLAS, and NumExpr can create their own thread pools for helper operations inside NumPy / SciPy expressions.
   - If those helper libraries also fan out to `N` threads, a single S2AND process can end up with nested parallelism
     (`Rayon x BLAS`, `OpenMP x BLAS`, etc.), which usually hurts end-to-end throughput through oversubscription.
   - So the safe default is: let the intended top-level engine use threads, and keep BLAS / NumExpr at `1` unless a
     profile shows a BLAS-heavy phase dominates and benefits from a different setting.

   Many OpenMP runtimes read environment variables at first use / first load; setting them after importing `lightgbm`
   is not reliable.

3. **Choose the env pattern that matches your deployment shape**:

   - **One S2AND process should use the machine**:

     ```bash
     export PYTHONUNBUFFERED=1
     export S2AND_BACKEND=rust
     export OMP_NUM_THREADS=36
     export MKL_NUM_THREADS=1
     export OPENBLAS_NUM_THREADS=1
     export NUMEXPR_NUM_THREADS=1
     export RAYON_NUM_THREADS=36

     uv run python your_script.py --n_jobs 36
     ```

     In this setup, do **not** leave `OMP_NUM_THREADS=1` if you want LightGBM / OpenMP inference to use the available
     cores.

   - **An outer scheduler / worker pool is already parallelizing the job**:

     ```bash
     export PYTHONUNBUFFERED=1
     export S2AND_BACKEND=rust
     export OMP_NUM_THREADS=1
     export MKL_NUM_THREADS=1
     export OPENBLAS_NUM_THREADS=1
     export NUMEXPR_NUM_THREADS=1
     export RAYON_NUM_THREADS=1

     uv run python your_worker_script.py --n_jobs 1
     ```

     If an outer launcher runs `W` workers on a machine with `N` cores, size each worker to roughly `floor(N / W)`
     inner threads, often `1`.

4. **Treat `PYTHONUNBUFFERED=1` as a logging knob**:
   - It helps flush logs promptly.
   - It does not make compute faster.

## Avoiding nested parallelism

Oversubscription usually comes from *stacking* an outer pool (threads/processes) on top of an inner thread pool
(Rayon/OpenMP/BLAS).

Rules of thumb:

- If you use an **outer process pool** with `W` workers, cap **inner threads** to ~`floor(N / W)` (often `1`) inside each
  worker.
- Prefer **one parallelism layer per phase**:
  - Rust batch featurization: Rayon handles parallelism; avoid wrapping it in additional thread pools.
  - LightGBM inference: let LightGBM/OpenMP use threads; avoid concurrent `predict_proba()` calls from multiple workers.
- BLAS / NumExpr are usually **not** the parallelism layer you want to scale first in S2AND. Leave
  `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` unless profiling shows otherwise.

## Rust Rayon pool lifetime

The Rust extension caches Rayon thread pools by thread count for reuse. Those worker threads stay alive for the process
lifetime, even between calls. This should not consume CPU when idle, but it does mean “thread count” tools may show more
threads than expected.

If you need to guarantee that all worker threads fully exit between runs, use process boundaries (run each workload in a
fresh Python process).

## Date

2026-03-16
