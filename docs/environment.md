# Environment Variables

Centralized reference for supported S2AND environment variables.

---

## Runtime Backend

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `S2AND_BACKEND` | `python`, `rust` | `python` | Backend used when a caller builds a runtime context without an explicit backend. Invalid values fail immediately. Public APIs have fixed routes: classic `ANDData` construction and prediction use Python, while Arrow-training and `*_from_arrow_paths` APIs use Rust. Rust requires the exact version pinned by the project metadata; there is no silent fallback. |

---

---

## Artifact Paths

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `S2AND_PATH_CONFIG` | `<path>` | `s2and/data/path_config.json` | Path to the JSON data-path config. Use when data lives outside the package default path. |

Normalization version is an artifact/model contract, not an environment
override. Canonical-v2 consumers reject missing or mismatched provenance rather
than allowing process configuration to relabel an artifact.

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
- **Import path policy**: avoid using `PYTHONPATH` for normal repo scripts because it can shadow an installed package or compiled extension. CI/test commands may set it only when intentionally testing the checkout source tree.
