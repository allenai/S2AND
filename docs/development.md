# Development

This document collects the repo-level development commands and workflows that do not need to stay in the root README.

## Core commands

Follow the [checkout setup](install.md#repo-checkout) to install the `dev` extra
and build the matching Rust extension. The full pytest suite requires the
extension, including the native name-count index used by Python orchestration.
The [local CI wrapper](#local-ci-mirror) prepares the test environment.
Use the [static-check fast path](#static-check-fast-path) for lint and type checks.

```bash
uv run pytest -q -n auto --dist loadfile
uv run ruff check .
uv run ruff format .
uv run ty check s2and
```

For a quick library check while you edit, skip the `tooling` tests. Those tests
cover `scripts/` release, data-prep, training, and dev tooling, and take about
two-thirds of the suite time. CI and the local CI wrapper still run every test.

```bash
uv run pytest -q -n auto --dist loadfile -m "not tooling"
```

Mark a new test file `pytestmark = pytest.mark.tooling` when it tests `scripts/` tooling.

`ty` reads its defaults from `[tool.ty]` in `pyproject.toml`: Python 3.11, the
`linux` platform used by the GitHub runners, and ignores for resolution and
attribute diagnostics from optional and native modules that static-only
environments omit. Type errors remain blocking. Pass
`--python-platform windows` for a Windows-specific check.

## Native build troubleshooting

The user-facing recovery commands are in
[install.md](install.md#build-the-rust-extension-from-source). This section
explains why they are sometimes needed.

Syncing `uv run` decides whether to rebuild `s2and` from the `cache-keys`
list in the root `pyproject.toml`. Packaged license and notice edits also
invalidate the cached distribution. Files outside that list, such as editor
swap files and build outputs, do not invalidate uv's cached wheel. Cargo still
watches source directories to detect new modules when a native build runs.

uv's file-glob keys track the newest matching timestamp, not file contents or
the complete file inventory. Deleting an older file, adding a backdated file,
or changing content while preserving timestamps can therefore miss a rebuild.

The extension embeds a content digest of its native build inputs (the Cargo
manifests, root lockfile, build scripts, and Rust/C++ sources; not the Python
wrapper or `README.md`). `tests/conftest.py` recomputes it from the checkout
and stops before collection when they differ, for example after
`uv run --no-sync` or a direct interpreter call following a Rust change. The
digest records inputs seen by the build script; it does not prove that Cargo
recompiled every dependency, which is why a Cargo clean is sometimes needed.
Participating native input paths must be valid UTF-8 and source directories
must not be symlinks or junctions; unsupported paths fail with a diagnostic
instead of producing different Python and Rust digests.

`uv run --active --no-project maturin develop -m s2and_rust/Cargo.toml` still
installs an editable build; the next syncing `uv sync` or `uv run` replaces it
with the locked build.

Delete any `s2and.egg-info` left in the checkout root by older setuptools builds.
Maturin never refreshes it, and code run from the root reads its stale version.
The native version check then fails. `tests/conftest.py` stops when it finds that copy.

## Local CI mirror

Run the full local CI wrapper:

```bash
uv run --no-project python scripts/dev/run_ci_locally.py
```

`scripts/dev/run_ci_locally.py` mirrors `.github/workflows/main.yaml` by running:

- lint (`scripts/dev/sync_version.py --check`, `scripts/dev/sync_feature_schema.py --check`,
  `ruff check`, and `ruff format --check`)
- one required-runtime `typecheck-and-test` job, which in order:
  - syncs the locked environment, building `s2and` from source;
  - runs native `cargo fmt`, correctness/suspicious `clippy`, and Rust
    library-test gates;
  - runs the installed-Rust API smoke and `ty`; and
  - runs the full pytest suite with coverage (branch coverage is configured in
    `pyproject.toml`) under pytest-xdist with `-n auto`, capped at eight
    workers. Pass
    `--pytest-workers N` to choose a worker count, or `--pytest-workers 0` to
    run it in one process for debugging.

PR CI runs Python 3.11–3.13 on Ubuntu and Python 3.11 on Windows. Both the local
runner and hosted jobs use the combined 80% coverage floor in `pyproject.toml`.
See [the test suite guide](../tests/README.md) for shared fixtures, test selection,
and the evidence standard.

The lint job runs the version and generated feature-schema checks without a
project environment and executes the exact Ruff pin from the `dev` extra in an
isolated uv tool environment. It does not install the runtime or ML dependency stack.

Hosted CI invokes this same script. Run one job independently with:

```bash
uv run --no-project python scripts/dev/run_ci_locally.py lint
uv run --no-project python scripts/dev/run_ci_locally.py typecheck-and-test
```

The CI runner's `ty` checks use Python 3.11 and the `linux` platform to match
the GitHub Linux runners. Set `S2AND_CI_TY_PLATFORM` (for example, `windows`)
to change the platform it emulates.

## Static-check fast path

If you want to skip Rust extension compilation while iterating, run only the static checks:

```bash
uv run --no-project python scripts/dev/run_ci_locally.py lint
uv sync --active --extra dev --frozen --no-install-package s2and
uv run --active --no-project ty check s2and
uv run --active --no-project ty check scripts --ignore unresolved-reference --ignore unresolved-attribute
```

The local `ty check scripts` command scans the whole `scripts/` tree and can
include untracked subdirectories. Hosted CI checks top-level `scripts/*.py` and the
stage and shared directories listed in `ACTIVE_NESTED_SCRIPT_DIRS`.
Both checks use the `[tool.ty]` defaults. The scripts check also ignores
unresolved references and attributes from optional and native modules. Type
errors remain blocking. Warnings remain visible.

## Runtime routing verification

Run the focused constructor and backend-routing tests from the development environment.

```bash
uv run pytest -q tests/test_runtime.py tests/test_data.py tests/test_arrow_training_ingestion.py
```

The [inference guide](production_inference.md#explicit-execution-routes) owns
the runtime contract. The production command reference covers
[feature parity and subblocking quality](../scripts/release/README.md#acceptance-checks),
[packaging](../scripts/release/README.md#distribution-and-arrow-checks), and
[installed-runtime checks](../scripts/release/README.md#installed-release-smoke).

## Implementation notes

These notes describe the internal boundaries of production inference.

### Import boundaries

Importing `s2and.production_model` does not load evaluation, plotting, or
Hyperopt modules. Pure clustering metrics live in `s2and.metrics` and are also
importable from `s2and.eval`. Evaluation loads plotting libraries
when rendering and scopes its plot styling to that operation.

Constructing a `Clusterer` or loading a production bundle does not create an
optimizer search space or load Hyperopt. With the default `search_space=None`,
the attribute remains `None` until `Clusterer.fit()` or Arrow validation
calibration creates the default uniform EPS space. Explicit search spaces are
used as given. Genie loads only when its subblocking algorithm is used.

### Prediction state ownership

Prediction orchestration keeps effective seed assignments, disallow pairs,
altered-profile overrides, and internal diagnostics in request-owned state.
Arrow subblocking builds synthetic seeds from its completed subblocks and
keeps them in request-owned state. Nested predictions and failed subblock calls
leave the original seed assignments intact.

Rust feature data can be reused across predictions. Python orchestration
passes each prediction's seed constraints at the Rust boundary. Applications
control the thread safety of their own classifiers and caches.

Per-call `cluster_model_params` apply throughout bulk prediction on both
routes, including initial-only attachment and residual clustering on Arrow.
Each block fits a fresh clustering estimator. Explicit `preserve_input`
choices survive nested calls, and the shared model's stored parameters are
unchanged. Incremental altered-profile splitting uses the configured clustering
model. Its reusable cache distinguishes the model parameters. Graph subblocking
uses its own configuration.

Seed-link application is a pure decision phase in promoted Arrow incremental
prediction. It accepts ordered seed memberships, link decisions, altered-profile
mappings, and first-name metadata. It returns owned cluster lists, residual
signatures, and rejected links. It preserves split-specific name checks and
restores original profile IDs.

The completion component uses that result to group residuals, allocate numeric
cluster IDs, and record request telemetry. It receives lazy first-name and
ORCID views, explicit policy and supervision, and a typed residual-clustering
operation. The orchestration boundary resolves aliases, logs rejected links,
and binds backend execution, constraints, and memory limits. The lazy views
read name metadata from the prepared dataset in place.

## Bounded model runs

Use `scripts/dev/run_bounded.py` for long model jobs with shared thread budgets and
durable logs. Follow the [bounded model runs runbook](bounded_runs.md).

## Version bumping

Versioning is centralized in `VERSION`.

This synchronizes the root Python version, Cargo version, and both lockfiles.
The single `s2and` distribution contains Python and native code.
The loader rejects a native binary whose version differs from the installed package.
Model/data release `1.3` and public format `1` remain unchanged.

Recommended one-time hook setup:

```bash
git config core.hooksPath .githooks
```

Version bump workflow:

```bash
# 1) edit VERSION to the new semantic version

# 2) sync manifests and both lockfiles
uv run python scripts/dev/sync_version.py

# 3) rebuild the environment for the new version
uv sync --extra dev
```

Notes:

- The pre-commit hook only runs when `VERSION` is staged; it runs steps 2 and 3
  and stages the updated files.
- `scripts/dev/sync_version.py` rewrites the version in `pyproject.toml` and
  `s2and_rust/Cargo.toml` and updates the package entries in `s2and_rust/Cargo.lock`
  and `uv.lock`. `cargo metadata --locked` and `uv lock --check` validate the
  updated files without re-resolving other dependencies.
- `uv.lock` and `s2and_rust/Cargo.lock` are generated files.

## Related docs

- Docs index: [docs/README.md](README.md)
- Runtime contract: [production_inference.md](production_inference.md#explicit-execution-routes)
- Promoted incremental performance profiling:
  [production command reference](../scripts/release/README.md#promoted-incremental-performance)
- v1.3 release operator runbook: [release.md](release.md)
