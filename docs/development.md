# Development

This document collects the repo-level development commands and workflows that do not need to stay in the root README.

## Core commands

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format .
uv run ty check s2and
```

## Local CI mirror

Run the full local CI wrapper:

```bash
uv run python scripts/run_ci_locally.py
```

`scripts/run_ci_locally.py` mirrors `.github/workflows/main.yaml` by running:

- lint (`scripts/sync_version.py --check`, `ruff check`, and `ruff format --check`)
- `typecheck-and-test` matrix lanes (`py-only`, then `rust-enabled`)
- Rust parity guardrail tests in the `rust-enabled` lane

The local runner passes `-ra` to pytest so skip reasons are printed under each lane. Rust-only tests may skip in the
`py-only` lane because that lane intentionally syncs without the `rust` extra and forces `S2AND_BACKEND=python`; the
same tests must run in the later `rust-enabled` lane after `maturin develop` builds the extension.

By default, local `ty` checks use `--python-version 3.11 --python-platform linux` to match GitHub Linux runners.

To override the local platform emulation:

- set `S2AND_CI_TY_PLATFORM`, for example `windows`

## Python-only fast path

If you want to skip Rust extension compilation while iterating:

```bash
uv sync --active --extra dev --frozen
uv run --active --no-project ruff format --check s2and scripts/*.py
uv run --active --no-project ty check s2and --ignore unresolved-import --ignore unused-type-ignore-comment --ignore possibly-missing-attribute --ignore unresolved-global
uv run --active --no-project ty check scripts/*.py --ignore unresolved-import --ignore unused-type-ignore-comment --ignore possibly-missing-attribute --ignore unresolved-global --ignore unresolved-reference --ignore unresolved-attribute
```

Pytest with coverage:

```bash
uv run --active --no-project pytest tests/ --cov=s2and --cov-report=term-missing --cov-fail-under=40
```

Do not set `PYTHONPATH` for normal repo scripts; it can shadow the installed package or compiled Rust extension. Test
and CI commands may set it only when they are intentionally exercising the checkout source tree.

## Version bumping

Versioning is centralized in `VERSION`.

Recommended one-time hook setup:

```bash
git config core.hooksPath .githooks
```

Version bump workflow:

```bash
# 1) edit VERSION
echo 0.51.1 > VERSION

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
- Rust runtime contract: [rust/runtime.md](rust/runtime.md)
- Rust promotion baselines: [rust/baselines.md](rust/baselines.md)
