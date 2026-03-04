# Environment Variables

Centralized reference for all S2AND environment variables.

---

## Runtime Backend

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `S2AND_BACKEND` | `python`, `rust`, `auto` | `auto` | Controls which backend is used for featurization and constraints. `auto` resolves to Rust when the extension is available and core-capable; otherwise Python. |

---

## Cache Configuration

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `S2AND_CACHE` | `<path>` | `~/.s2and` | Cache root directory (only used when `use_cache=True`). |
| `S2AND_RUST_FEATURIZER_MAX_INMEM` | `<int>` | unbounded | Cap in-memory Rust featurizer entries (`0` = unbounded). Use `1` for single-dataset-per-process workloads; `2-3` if alternating among a few datasets. Only matters when `use_cache=True`. |

---

## Artifact Paths

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `S2AND_RUST_NAME_COUNTS_JSON` | `<path>` | none | Artifact-backed name-count lookups for Rust JSON ingest (`from_json_paths`). Used when dataset signature-level name counts are not available. |

---

## Normalization Compatibility

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `S2AND_NORMALIZATION_VERSION` | `<string>` | `legacy_compat` | Normalization version expected by artifact-backed name-count ingest. |
| `S2AND_ALLOW_NORMALIZATION_VERSION_MISMATCH` | `0`, `1` | `0` | Allow artifact-backed name-count ingest with missing/mismatched normalization metadata. |

---

## Testing & Benchmarking

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `S2AND_SKIP_FASTTEXT` | `0`, `1` | `0` | Skip FastText loading (useful for tests/benchmarks that don't need language detection). |

---

## Threading & Parallelism

These variables control thread counts for various libraries. Set them **before importing** compute-heavy libraries.

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `RAYON_NUM_THREADS` | `<int>` | auto | Rust-side thread count (standard Rayon env var). S2AND's Rust extension primarily uses explicit `num_threads` arguments, so this mainly affects Rayon's global pool. |
| `OMP_NUM_THREADS` | `<int>` | auto | OpenMP thread count (affects LightGBM and some clustering libs). |
| `MKL_NUM_THREADS` | `<int>` | auto | Intel MKL thread count (if your NumPy/SciPy stack uses MKL). |
| `OPENBLAS_NUM_THREADS` | `<int>` | auto | OpenBLAS thread count (if your NumPy/SciPy stack uses OpenBLAS). |
| `NUMEXPR_NUM_THREADS` | `<int>` | auto | NumExpr thread count. |

See `docs/threading.md` for detailed guidance on avoiding nested parallelism and oversubscription.

---

## CI-Specific

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `S2AND_CI_TY_PLATFORM` | `linux`, `windows`, etc. | `linux` | Override platform emulation for local `ty` checks. By default, local CI runs use `--python-platform linux` to match GitHub Linux runners. |

---

## Notes

- **Rust batch mode** uses Rayon internally for parallelism; Python process pools are not used when Rust is enabled.
- **Thread env vars** (OMP, MKL, etc.) are typically read at library load time. Setting them after importing `lightgbm` or similar is unreliable.
- **Windows memory budgeting** uses `GlobalMemoryStatusEx` for total RAM and `GetProcessMemoryInfo` for RSS when `psutil` is unavailable.
