# Development

This document collects the repo-level development commands and workflows that do not need to stay in the root README.

## Core commands

For a fresh development environment, install the `dev` extra and build the matching Rust extension using the
[local development build instructions](../s2and_rust/README.md#local-dev-build). The full pytest suite requires that
native runtime. The [local CI wrapper](#local-ci-mirror) prepares it, while the
[static-check fast path](#static-check-fast-path) supports lint/type checks without a native build.

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format .
uv run ty check s2and --ignore unresolved-import --ignore unused-type-ignore-comment --ignore possibly-missing-attribute --ignore unresolved-global
```

## Scoped searches

Choose the relevant source roots for the task. For a search spanning maintained code and docs, these PowerShell
examples reuse exclusions for virtual environments, data, and generated output. The `.venv` glob also excludes
directories such as `.venv-dev` and `project.venv-cache`, including nested paths.

```powershell
$searchRoots = @('s2and', 's2and_rust/src', 'scripts', 'tests', 'docs')
$searchGlobs = @(
  '--glob', '!**/*.venv*/**',
  '--glob', '!**/.git/**',
  '--glob', '!**/data/**',
  '--glob', '!**/dist/**',
  '--glob', '!**/scratch/**',
  '--glob', '!**/target/**'
)
rg -n --hidden @searchGlobs 'pattern' @searchRoots
rg --files --hidden @searchGlobs @searchRoots

# When a root content search is necessary, keep the exclusions and cap file size.
rg -n --hidden @searchGlobs --max-filesize 4M 'pattern' .
```

For intentional data inspection, choose the exact file or directory and bound the output (for example,
`rg -n --max-count 5 --glob '!**/*.venv*/**' 'pattern' data/specific-file.json`).

## Local CI mirror

Run the full local CI wrapper:

```bash
uv run --no-project python scripts/run_ci_locally.py
```

`scripts/run_ci_locally.py` mirrors `.github/workflows/main.yaml` by running:

- lint (`scripts/sync_version.py --check`, `scripts/sync_feature_schema.py --check`,
  `ruff check`, and `ruff format --check`)
- one required-runtime `typecheck-and-test` job
- native `cargo fmt`, correctness/suspicious `clippy`, and Rust library-test
  gates
- a local Maturin build, installed-Rust API smoke, `ty`, and the full pytest
  suite with `S2AND_BACKEND=python`, required native dependencies, and branch coverage

PR CI runs Python 3.11–3.13 on Ubuntu and Python 3.11 on Windows. Both the local
runner and hosted jobs use the combined 80% coverage floor in `pyproject.toml`.
See [the test suite guide](../tests/README.md) for shared fixtures, test selection,
and the evidence standard.

The lint job runs the version and generated feature-schema checks without a
project environment and executes the exact Ruff pin from the `dev` extra in an
isolated uv tool environment. It does not install the runtime or ML dependency stack.

Hosted CI invokes this same script. Run one job independently with:

```bash
uv run --no-project python scripts/run_ci_locally.py lint
uv run --no-project python scripts/run_ci_locally.py typecheck-and-test
```

The current runtime requires the Rust-backed name-count index even when Python orchestration is selected. The runner builds
the exactly pinned `s2and-rust` dependency once, requires it to import, and then runs full-suite coverage of the Python
route.

By default, local `ty` checks use `--python-version 3.11 --python-platform linux` to match GitHub Linux runners.

To override the local platform emulation:

- set `S2AND_CI_TY_PLATFORM`, for example `windows`

## Static-check fast path

If you want to skip Rust extension compilation while iterating, run only the static checks:

```bash
uv run --no-project python scripts/run_ci_locally.py lint
uv sync --active --extra dev --frozen --no-install-package s2and-rust
uv run --active --no-project ty check s2and --ignore unresolved-import --ignore unused-type-ignore-comment --ignore possibly-missing-attribute --ignore unresolved-global --python-version 3.11 --python-platform linux
uv run --active --no-project ty check scripts --exclude scripts/archive --ignore unresolved-import --ignore unused-type-ignore-comment --ignore possibly-missing-attribute --ignore unresolved-global --ignore unresolved-reference --ignore unresolved-attribute --python-version 3.11 --python-platform linux
```

The scripts check above covers active top-level and nested scripts but deliberately
excludes retired reference code in `scripts/archive`; hosted CI currently type-checks
only top-level scripts. The explicit platform flags match the CI default; replace
`linux` with `windows` for a Windows-specific check. These ignores are limited
to resolution/attribute diagnostics from optional and native modules omitted
from the static-only environment. Type errors remain blocking; warnings remain
visible.

The full pytest suite requires the native name-count index. Build the extension first or use
`uv run --no-project python scripts/run_ci_locally.py`.

Do not set `PYTHONPATH` for normal repo scripts; it can shadow the installed package or compiled Rust extension. Test
and CI commands may set it only when they are intentionally exercising the checkout source tree.

## Runtime routing verification

After building the matching extension with the
[local development build instructions](../s2and_rust/README.md#local-dev-build),
run the focused constructor and backend-routing tests:

```bash
uv run pytest -q tests/test_runtime.py tests/test_data.py tests/test_arrow_training_ingestion.py
```

The [inference guide](production_inference.md#explicit-execution-routes) owns
the runtime contract. Feature parity, subblocking quality, packaging, and
installed-runtime checks use the focused commands in the
[production command reference](../scripts/production/README.md#evaluation).

## Version bumping

Versioning is centralized in `VERSION`.

This synchronizes Python/Rust package manifests mechanically. The release fixes
both package versions at `1.0.0`; the production model and public
data remain release version `1.3`, and the Arrow/name-count representation uses
public format `1`. At runtime the installed Python and Rust package versions
must match exactly. The release policy and publication gates are in
[release.md](release.md).

Recommended one-time hook setup:

```bash
git config core.hooksPath .githooks
```

Version bump workflow:

```bash
# 1) edit VERSION to the new semantic version

# 2) sync manifests
uv run python scripts/sync_version.py

# 3) regenerate lockfiles
uv sync --extra dev
uv run --active --no-project cargo generate-lockfile --manifest-path s2and_rust/Cargo.toml
```

Notes:

- The pre-commit hook only runs when `VERSION` is staged.
- The hook auto-syncs manifests and regenerates lockfiles when needed.
- `uv.lock` and `s2and_rust/Cargo.lock` are generated files.

## Related docs

- Docs index: [docs/README.md](README.md)
- Runtime contract: [production_inference.md](production_inference.md#explicit-execution-routes)
- Promoted incremental performance profiling:
  [production command reference](../scripts/production/README.md#promoted-incremental-performance-report)
- v1.3 release operator runbook: [release.md](release.md)
