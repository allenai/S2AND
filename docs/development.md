# Development

This document collects the repo-level development commands and workflows that do not need to stay in the root README.

## Core commands

For a fresh development environment, install the `dev` extra and build the matching Rust extension using the
[local development build instructions](../s2and_rust/README.md#local-dev-build). The full pytest suite requires that
native runtime. The [local CI wrapper](#local-ci-mirror) prepares it, while the
[static-check fast path](#static-check-fast-path) supports lint/type checks without a native build.

```bash
uv run pytest -q -n auto
uv run ruff check .
uv run ruff format .
uv run ty check s2and
```

`ty` reads its defaults from `[tool.ty]` in `pyproject.toml`: Python 3.11, the
`linux` platform used by the GitHub runners, and ignores for resolution and
attribute diagnostics from optional and native modules that static-only
environments omit. Type errors remain blocking. Pass
`--python-platform windows` for a Windows-specific check.

## Native build troubleshooting

The user-facing recovery commands are in
[install.md](install.md#build-the-rust-extension-from-source). This section
explains why they are sometimes needed.

Syncing `uv run` decides whether to rebuild `s2and-rust` from the `cache-keys`
list in `s2and_rust/pyproject.toml`. Files outside that list, such as editor
swap files and build outputs, do not invalidate uv's cached wheel. Cargo still
watches source directories to detect new modules when a native build runs.

uv's file-glob keys track the newest matching timestamp, not file contents or
the complete file inventory. Deleting an older file, adding a backdated file,
or changing content while preserving timestamps can therefore miss a rebuild.

The extension embeds a content digest of its Rust build inputs (the Cargo
manifests, root lockfile, build script and Rust sources; not the Python
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

## Local CI mirror

Run the full local CI wrapper:

```bash
uv run --no-project python scripts/run_ci_locally.py
```

`scripts/run_ci_locally.py` mirrors `.github/workflows/main.yaml` by running:

- lint (`scripts/sync_version.py --check`, `scripts/sync_feature_schema.py --check`,
  `ruff check`, and `ruff format --check`)
- one required-runtime `typecheck-and-test` job, which in order:
  - syncs the locked environment, building `s2and-rust` from source;
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
uv run --no-project python scripts/run_ci_locally.py lint
uv run --no-project python scripts/run_ci_locally.py typecheck-and-test
```

The current runtime requires the Rust-backed name-count index even when Python orchestration is selected. The runner builds
the exactly pinned `s2and-rust` dependency once, requires it to import, and then runs full-suite coverage of the Python
route.

The CI runner's `ty` checks use Python 3.11 and the `linux` platform to match
the GitHub Linux runners. Set `S2AND_CI_TY_PLATFORM` (for example, `windows`)
to change the platform it emulates.

## Static-check fast path

If you want to skip Rust extension compilation while iterating, run only the static checks:

```bash
uv run --no-project python scripts/run_ci_locally.py lint
uv sync --active --extra dev --frozen --no-install-package s2and-rust
uv run --active --no-project ty check s2and
uv run --active --no-project ty check scripts --ignore unresolved-reference --ignore unresolved-attribute
```

The scripts check above covers the top-level and nested scripts; hosted CI checks the same
set. Both checks use the `[tool.ty]` defaults; the scripts
check additionally ignores unresolved references and attributes from optional
and native modules. Type errors remain blocking; warnings remain visible.

The full pytest suite requires the native name-count index. `uv sync --extra dev` builds and installs the extension;
`uv run --no-project python scripts/run_ci_locally.py` does the same before testing.

## Runtime routing verification

After building the matching extension with the
[local development build instructions](../s2and_rust/README.md#local-dev-build),
run the focused constructor and backend-routing tests:

```bash
uv run pytest -q tests/test_runtime.py tests/test_data.py tests/test_arrow_training_ingestion.py
```

The [inference guide](production_inference.md#explicit-execution-routes) owns
the runtime contract. The production command reference covers
[feature parity and subblocking quality](../scripts/production/README.md#evaluation),
[packaging](../scripts/production/README.md#distribution-and-arrow-checks), and
[installed-runtime checks](../scripts/production/README.md#installed-release-smoke).

## Bounded model runs

Detect available logical CPUs from process affinity before each CPU-intensive launch.
Reserve two logical CPUs. Share the remaining `k - 2` thread budget across concurrent jobs.
Apply a lower total limit when the user requests one.
For a 12-thread total limit, allocate four threads to each of three workers.
Apply each allocation to model prediction, OpenMP, and Rayon. Keep secondary BLAS pools at one thread.
Use disjoint CPU affinity leases as a backstop for parallel library pools.
Record the detected capacity and allocations. Do not give each worker the complete shared budget.

For comparisons of complete production models, complete one model before more
variants. Require a strict reload of the complete production bundle. Then run a
bounded public prediction on one representative full block. Verify its outputs
before the next variant. For example, the giant-block count study uses full
Silva at EPS 0, .55, and 1. An interim pairwise export does not meet this gate.

Run one serial milestone at a time. Do not queue later variants before this gate
passes. Use a bounded representative sample before an expensive milestone.
Record its command, inputs, resource limits, expected duration, and required
outputs.

Before a threshold sweep, identify work that does not depend on the threshold.
Reuse that work across cuts after exact output checks. Keep threshold-dependent
constraints, temporary seeds, attachment, and residual prediction in the public
production path. Cache pristine arrays before a consumer can change them. Bind
the cache to fixed inputs, models, counts, source, native code, and resource
settings. Measure cold and warm runs before the full runtime estimate.

Wrap each milestone with `scripts/run_bounded.py`. Set `--output-dir` and
`--max-rss-gib`. Select `--max-seconds SECONDS` or `--no-deadline`.
Use `--no-deadline` when the approved workload has no wall deadline.
The memory limit and process ownership remain active.
The records store `max_seconds` as `null` for this choice.

Repeat `--expected-output PATH` for each
required output file. Put the child command after `--`. Use a fresh output
directory for each attempt.

Use expected output files that do not exist before launch. Do not use runner
records as expected outputs. The runner checks file presence and records file
size. Validate artifact contents separately.

Preserve `run.json`, `events.jsonl`, `stdout.log`, and `stderr.log`. The terminal
`result.json` and validated outputs establish completion. Require `status` equal
to `complete` and child exit code zero. Process disappearance, stale status,
and file presence alone do not establish success. Preserve partial outputs and
worker logs after failure.

Use `supervisor_pid` in `run.json` to identify the actual supervisor. Outer `uv`
and environment launcher PIDs can differ.

Use one timed query for each progress check. Do not start parallel status
monitors. On timeout, stop only the bounded command's owned process tree.
Preserve unrelated workers. The runner uses a 0.1-second resource poll interval.
Samples can miss short memory peaks.

On Windows, a Job Object contains the supervisor and descendants from launch.
Actual supervisor death closes its private handle and stops remaining
descendants. On POSIX, ownership covers the command's process group. A
descendant can start a new session outside this group. Hard supervisor death
can leave group members alive.

A remaining descendant keeps the run open after the command exits. One Windows
exception exists: MSVC `link.exe` starts `vctip.exe` to upload telemetry, and it
can linger after a Rust build. Ten seconds after the command exits, the runner
checks the remaining job members. If each one is `vctip.exe` inside a
`Microsoft Visual Studio` folder or the command's console host (`conhost.exe`),
it stops the `vctip.exe` processes and lists them in `telemetry_cleanup` in
`result.json`. The console host then exits by itself. Any other remaining
process still keeps the run open.

On Windows, a reader without delete sharing can block atomic file replacement.
Prefer immutable records and append-only telemetry. If a status file requires
replacement, close readers promptly and permit delete sharing. Report optional
telemetry failures without termination of an active worker. Failure to publish a
required artifact must fail the milestone.

Before an expensive launch, run these Git checks in order. Each command has its
own time and memory limit.

```bash
uv run --no-sync python scripts/run_bounded.py --output-dir scratch/git-head-check --max-seconds 15 --max-rss-gib 1 -- git rev-parse --verify "HEAD^{commit}"
uv run --no-sync python scripts/run_bounded.py --output-dir scratch/git-tree-check --max-seconds 15 --max-rss-gib 1 -- git cat-file -e "HEAD^{tree}"
uv run --no-sync python scripts/run_bounded.py --output-dir scratch/git-ref-check --max-seconds 15 --max-rss-gib 1 -- git for-each-ref --sort=-committerdate "--format=%(refname)%00%(objectname)%00%(*objectname)"
```

These three checks cover the HEAD commit, root tree, and ref enumeration. They
do not establish complete repository health. Stop the launch if any check fails
or exceeds its limit. Preserve the checkout and request a separate Git
diagnosis. Do not reset or delete repository files. Do not run unbounded Git
checks.

Keep active checkouts and their Git directories outside cloud-sync folders.
Transfer commits with Git push and fetch. The [Git FAQ](https://git-scm.com/docs/gitfaq#_transfers)
states that cloud sync can cause missing objects and broken refs. This risk
does not establish the cause of any specific failure. Do not move an active
checkout while jobs use its files or environment.

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

# 2) sync manifests and both lockfiles
uv run python scripts/sync_version.py

# 3) rebuild the environment for the new version
uv sync --extra dev
```

Notes:

- The pre-commit hook only runs when `VERSION` is staged; it runs steps 2 and 3
  and stages the updated files.
- `scripts/sync_version.py` rewrites the version in `pyproject.toml`,
  `s2and_rust/pyproject.toml`, `s2and_rust/Cargo.toml`, and the package entries
  in `s2and_rust/Cargo.lock` and `uv.lock`. The result passes
  `cargo metadata --locked` and `uv lock --check`, so a bump does not re-resolve
  other dependencies.
- `uv.lock` and `s2and_rust/Cargo.lock` are generated files.

## Related docs

- Docs index: [docs/README.md](README.md)
- Runtime contract: [production_inference.md](production_inference.md#explicit-execution-routes)
- Promoted incremental performance profiling:
  [production command reference](../scripts/production/README.md#promoted-incremental-performance-report)
- v1.3 release operator runbook: [release.md](release.md)
