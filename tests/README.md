# Test suite

Use the matching native extension and the `dev` extra; see the
[development guide](../docs/development.md). Rust and PyArrow are required
dependencies. A missing or incompatible runtime must fail instead of silently
skipping native coverage. Shell tests require Bash (native Git Bash on Windows);
platform-specific file-lock tests remain explicitly marked.

```bash
uv run pytest -q
uv run pytest -q tests/test_arrow_inputs.py
uv run pytest -q tests/test_predict_and_combine.py -k real_native
uv run pytest -q --cov=s2and --cov-branch --cov-report=term-missing --durations=25
uv run ruff check s2and scripts tests
```

The default invocation collects the entire suite. There is no implicit slow-test
or native-test exclusion. Pytest validates configuration and marker names and
reports skips. Large datasets, paid APIs, and production training are separate
release checks, not prerequisites for these bounded tests.

The shared CI runner measures branches and enforces the 80% combined coverage
floor from `pyproject.toml`. PR CI runs Python 3.11–3.13 on Ubuntu and Python 3.11
on Windows, so Windows file-lock and open-file cleanup contracts are exercised.

## Shared infrastructure

| Module | Responsibility |
| --- | --- |
| `conftest.py` | Python backend default and restoration of global Python/NumPy RNG state after each test |
| `helpers.py` | Tiny datasets, name counts, Arrow conversion, query/cluster values, and pairwise CLI arguments |
| `raw_arrow_helpers.py` | Tiny raw Arrow bundles, retained readers, and native planner construction |
| `promoted_linking_helpers.py` | Deterministic tiny boosters, pairwise bundles, and logistic gate configuration |
| `model_helpers.py` | Constant pair probabilities for tests of clustering policy |
| `training_helpers.py` | Classic training, calibration, and holdout populations |
| `shell_helpers.py` | Bounded Bash execution with captured diagnostics and native Git Bash selection |
| `linker_row_feature_reference.py` | Independent row-wise reference for native feature comparisons |
| `fixtures/rust_lightgbm/` | Serialized models and independent prediction fixtures |

Test modules must not import other collected test modules. Put a builder used by
multiple domains in the appropriate helper module; keep single-use fixtures near
their tests. Builders construct inputs, while assertions stay with the test.
Keep identity/expected-result oracles independent from the production algorithm.

Ordinary tests use Python orchestration even if the invoking shell sets
`S2AND_BACKEND=rust`. Native tests select explicit runtime contexts or use
`monkeypatch.setenv`. Tests of absent-environment defaults use
`monkeypatch.delenv`. Prefer local seeded RNGs; global RNG restoration protects
legacy tests without replacing assertions about a function's own RNG effects.
Use `tmp_path` for mutable artifacts and `tmp_path_factory` for immutable shared
fixtures. Patch the failing IO/scorer boundary, not the function under test.

## Evidence standard

- State the input, expected output or failure, and user-visible consequence.
- Prefer exact partitions, selected IDs, hand-calculated metrics, and real tiny
  serialization/scoring round trips over mocks that return the asserted result.
- Parameterize independent cases so a failure identifies its input and does not
  prevent other cases from running.
- Test failure cleanup as well as rejection: preserve the previous artifact,
  release handles/locks, and prove a retry works.
- Keep meaningful boundary checks (checksums, label independence, holdout
  separation, import isolation, release gates). Source substrings alone do not
  establish these contracts.
- For significant new assertions, demonstrate a plausible fault that they catch.
  Coverage helps locate gaps; it does not establish correctness by itself.
- Before deleting overlapping tests, identify the retained assertion that catches
  the same fault. Prefer explicit boundary and interaction cases to Cartesian
  products of independent options; retain an independent expected-result oracle.

## Restricted Windows environments

If an existing user cache or pytest temporary root is inaccessible, use writable
workspace paths. Give each concurrent run its own fresh `--basetemp`; pytest
cleans that directory, so never point it at valuable files.

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) 'scratch/uv-cache'
uv run --no-sync pytest -q --basetemp=scratch/my-test-run-temp -o cache_dir=scratch/pytest-cache
```

`--no-sync` is for an already prepared environment. Normal setup and CI should
still synchronize dependencies and build the matching native extension.
