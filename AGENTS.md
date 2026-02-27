# AGENTS.md

This file defines working agreements for coding agents in this repo.
Optimize for verifiability: tests + measurable evidence (logs/metrics/profiles) over intuition.

## Non-negotiables
1) Always use `uv` for Python (setup, running, tooling). Never invoke raw `pip`.
   - Setup: `uv sync`
   - Run: `uv run ...`
   - Python: `uv python install <ver>` / `uv python pin <ver>` (if needed)
   - Use `ruff` and `ty`. Do not use `mypy` or `black`.
2) Prefer verifiable changes.
   - For behavior changes: add/update `pytest`.
   - For performance claims: include profiling evidence and the delta.
   - For data pipelines: include spot-check metrics and counts.
3) Expensive pipelines (time or money).
   - First run components on tiny fixtures / small samples.
   - If the test will be useful again, convert it into a `pytest` regression test.
4) Error handling must surface failures.
   - Catch only narrow exceptions you can handle.
   - When catching: log context, then re-raise or return an explicit typed error result.
   - Retries must be bounded and instrumented (attempt count + final failure).
5) Cost and safety guardrails.
   - Default to small samples; require an explicit `--limit` (or similar) for large runs.
   - For paid APIs: estimate cost and stop if cost might exceed a reasonable threshold.
   - Ask before actions that can delete data, change schemas, rotate secrets, or trigger large spend.
6) Search must explicitly exclude virtualenv paths.
   - Always exclude any folder that has `.venv` in its path.

## Search Performance Guardrails (Important in this repo)

This repo includes multi-GB files under `data/` (notably `data/inventors/*.json`).
Unscoped root searches like `rg ... .` can scan tens of GB and make the machine sluggish,
especially in cloud-synced folders.

- Do not run unscoped content search from repo root by default.
- Prefer searching code directories directly: `s2and`, `scripts`, `tests`, `docs`.
- Default content search pattern:

```powershell
rg -n --hidden `
  --glob '!**/.venv/**' `
  --glob '!**/.git/**' `
  --glob '!data/**' `
  --glob '!dist/**' `
  --glob '!scratch/**' `
  "pattern" s2and scripts tests docs
```

- If you must search from `.`, include exclusions plus a file-size cap:

```powershell
rg -n --hidden `
  --glob '!**/.venv/**' `
  --glob '!**/.git/**' `
  --glob '!data/**' `
  --max-filesize 4M `
  "pattern" .
```

- For filename discovery, scope first:

```powershell
rg --files s2and scripts tests docs --hidden --glob '!**/.venv/**'
```

- Only search `data/` intentionally, with an explicit path and bounded output (`--max-count`).

## Verification Requirements
- Unit tests: add/modify `pytest` for each behavior change.
- CLI verification: provide exact command(s) and expected output shape.
- UI changes: provide a concrete screenshot-based or DOM-based check when applicable.
- Performance work:
  1) profile on a realistic workload to get hotspots
  2) change a few things
  3) re-profile
  4) report delta
  5) repeat until plateau

## Python Commands (Canonical)
- Tests: `uv run pytest -q`
- Single test: `uv run pytest -q path/to/test_file.py::test_name`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Typecheck (if configured): `uv run ty ...` or `uv run pyright ...`

## Repository Hygiene
- Keep changes small and reviewable.
- Prefer adding tests alongside code changes.
- Avoid new dependencies unless necessary; ask before adding production deps.
- Update docs when changing public behavior.

## Ask-First Triggers
Stop and ask before:
- Changing schemas, public APIs, or serialization formats
- Large refactors or broad deletions/migrations
- Introducing new production dependencies
- Running large jobs (time-intensive) or paid API jobs that might be costly
