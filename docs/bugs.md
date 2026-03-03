# Bug Audit — `some_tweaks` vs `main` (second pass, 2026-03-02)

## Actionable bugs (verified)

### 1) Rust cluster-seed sync could go stale after in-place seed edits (high)

- **Impact:** Rust constraint evaluation could use stale `cluster_seeds_require` / `cluster_seeds_disallow` after in-place edits, producing incorrect constraints during clustering.
- **Root cause:** `_sync_rust_cluster_seeds` skip logic relied on `(version, id, len)` and could miss value-only in-place changes.
- **Repro (pre-fix):**
  - Build a dataset object with seed containers.
  - Call `_sync_rust_cluster_seeds(...)` once.
  - Mutate `dataset.cluster_seeds_require["s2"]` in place without changing length.
  - Call `_sync_rust_cluster_seeds(...)` again and observe `update_rust_cluster_seeds` not called.
- **Fix:**
  - Added mutation-tracked seed containers in `s2and/model.py`:
    - `_VersionedClusterSeedDict`
    - `_VersionedClusterSeedSet`
    - `_ensure_cluster_seed_version_tracking`
  - `_sync_rust_cluster_seeds` now ensures tracked containers before applying unchanged-skip logic.
- **Regression coverage:**
  - `tests/test_regression_fixes.py::test_sync_rust_cluster_seeds_detects_in_place_seed_mutation`
  - `tests/test_regression_fixes.py::test_sync_rust_cluster_seeds_skips_when_unchanged`
- **Validation command/output:**
  - `uv run pytest -q tests/test_regression_fixes.py::test_sync_rust_cluster_seeds_skips_when_unchanged tests/test_regression_fixes.py::test_sync_rust_cluster_seeds_detects_in_place_seed_mutation`
  - Result: `2 passed`

### 2) `UniversalPool.imap` silently dropped all work for non-positive knobs (high)

- **Impact:** Passing `chunksize <= 0` or `max_prefetch <= 0` yielded no results and no error, silently masking caller bugs.
- **Root cause:** `_streaming_imap` assumes positive values; `imap` did not validate either parameter.
- **Repro (pre-fix):**
  - `list(pool.imap(lambda x: x + 1, [1, 2, 3], chunksize=0))` returned `[]`.
  - `list(pool.imap(lambda x: x + 1, [1, 2, 3], chunksize=1, max_prefetch=0))` returned `[]`.
- **Fix:**
  - Added explicit validation in `s2and/mp.py::UniversalPool.imap`:
    - raise `ValueError` for `chunksize < 1`
    - raise `ValueError` for `max_prefetch < 1`
- **Regression coverage:**
  - `tests/test_mp.py::test_streaming_imap_rejects_non_positive_chunksize`
  - `tests/test_mp.py::test_streaming_imap_rejects_non_positive_max_prefetch`
- **Validation command/output:**
  - `uv run pytest -q tests/test_mp.py`
  - Result: all tests passed, including new validation tests.

## Non-actionable items from the initial draft (kept for traceability)

- Rust JSON ingest arg order: contract matches current Rust signature.
- Name-count last-first-initial semantics: artifact + runtime behavior match.
- Sinonym given-name joining: behavior is intentional and tested.
- Incremental clustering return payload: telemetry payload by default is intentional.
- Linux-only stage-1 multiprocessing in paper preprocessing: intentional and tested.
