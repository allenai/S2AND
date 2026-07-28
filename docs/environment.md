# Environment Variables

Centralized reference for supported S2AND environment variables.

---

## Runtime and telemetry

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `S2AND_BACKEND` | `python`, `rust` | `python` | Backend used when a caller builds a runtime context without an explicit backend. Invalid values fail immediately. Public APIs have fixed routes: classic `ANDData` construction and prediction use Python, while APIs taking an open `ArrowDataset` use Rust. Rust requires the exact version pinned by the project metadata; there is no silent fallback. |
| `S2AND_MEMORY_TELEMETRY_JSONL` | `<path>` | unset | Sole library authority for appending structured memory-telemetry JSONL. Parent directories are created. Prefer a fresh run-specific path; records append under an in-process lock. |

---

## Artifact Paths

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `S2AND_PATH_CONFIG` | `<path>` | `s2and/data/path_config.json` | Path to the JSON data-path config. Use when data lives outside the package default path. |

Artifact compatibility is not configurable through the environment. Model
bundles record the exact generating runtime, and independently readable Arrow
and name-count data declares public format `1`.

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
| `S2AND_TEST_REQUIRE_RUST` | truthy/falsey string | unset/false | CI/test-only guard. Truthy values (`1`, `true`, `yes`, `on`) require the installed Rust extension instead of allowing Rust-dependent tests to skip. Normal applications should not set it. |

---

## Notes

- **Rust batch mode** uses Rayon internally for parallelism; Python process pools are not used when Rust is enabled.
- **Thread env vars** (OMP, MKL, etc.) are typically read at library load time. Setting them after importing `lightgbm` or similar is unreliable.
- **Windows memory budgeting** uses `GlobalMemoryStatusEx` for total RAM and `GetProcessMemoryInfo` for RSS when `psutil` is unavailable.
- **Import path policy**: avoid using `PYTHONPATH` for normal repo scripts because it can shadow an installed package or compiled extension. CI/test commands may set it only when intentionally testing the checkout source tree.
