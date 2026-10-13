# Environment Variables

Centralized reference for supported S2AND environment variables.

---

## Runtime and telemetry

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `S2AND_MEMORY_TELEMETRY_JSONL` | `<path>` | unset | Sole library authority for appending structured memory-telemetry JSONL. Parent directories are created. Prefer a fresh run-specific path; records append under an in-process lock. |

---

## Artifact Paths

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `S2AND_PATH_CONFIG` | `<path>` | `s2and/data/path_config.json` | Path to the JSON data-path config. Use when data lives outside the package default path. |

Artifact compatibility is not configurable through the environment.
Models declare supported format and feature-contract versions plus ordered feature names.
The generating runtime records provenance.
Arrow and name-count data use public format `1`, independently of the package version.
Installed Python and Rust package versions still match for their internal interfaces.

## Threading & Parallelism

These variables control thread counts for various libraries. Set them **before importing** compute-heavy libraries.

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `RAYON_NUM_THREADS` | `<int>` | auto | Rust-side thread count (standard Rayon env var). S2AND's Rust extension primarily uses explicit `num_threads` arguments, so this mainly affects Rayon's global pool. |
| `OMP_NUM_THREADS` | `<int>` | auto | OpenMP thread count for Python LightGBM training and other OpenMP libraries. Production model scoring runs in Rust and follows `n_jobs` instead. Most S2AND scripts (including `train_pairwise.py`, `eval_prod_models.py`, and `profile_promoted_incremental_arrow.py`) and linker-training data loading overwrite this variable from their own `n_jobs` setting. |
| `MKL_NUM_THREADS` | `<int>` | auto | Intel MKL thread count (if your NumPy/SciPy stack uses MKL). |
| `OPENBLAS_NUM_THREADS` | `<int>` | auto | OpenBLAS thread count (if your NumPy/SciPy stack uses OpenBLAS). |
| `NUMEXPR_NUM_THREADS` | `<int>` | auto | NumExpr thread count. |

See [threading.md](threading.md) for detailed guidance on avoiding nested parallelism and oversubscription.

---

## CI-Specific

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `S2AND_CI_TY_PLATFORM` | `linux`, `windows`, etc. | `linux` | Override platform emulation for local `ty` checks. By default, local CI runs use `--python-platform linux` to match GitHub Linux runners. |

The full test suite requires the matching Rust extension and PyArrow without an
environment opt-in. Tests default to Python orchestration; native cases select
their runtime explicitly. See [the test suite guide](../tests/README.md).

---

## Notes

- **Rust Arrow routes** parallelize through Rayon using `n_jobs`. Python worker pools (`UniversalPool`) belong to classic `ANDData` paper preprocessing and Python featurization.
- **Thread env vars** (OMP, MKL, etc.) are typically read at library load time. Setting them after importing `lightgbm` or similar is unreliable.
- **Windows memory budgeting** uses `GlobalMemoryStatusEx` for total RAM and `GetProcessMemoryInfo` for RSS when `psutil` is unavailable.
- **Import path policy**: avoid using `PYTHONPATH` for normal repo scripts because it can shadow an installed package or compiled extension. CI/test commands may set it only when intentionally testing the checkout source tree.
