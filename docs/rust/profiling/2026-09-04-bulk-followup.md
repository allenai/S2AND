# Bulk Arrow specialization follow-up

This follow-up measures candidates against the already optimized bulk path
in commit `a5bfdeb`. The acceptance gates remain: at least 10% less time in
the affected category, unchanged outputs, and at most 10% more peak working
set and peak committed memory. All native runs use a verified release build
and `n_jobs=10`.

## Accepted results

Both implemented candidates pass all three gates. Three sequential requests
per implementation in fresh baseline/candidate processes gave:

| Measurement | Before | After | Reduction |
|---|---:|---:|---:|
| Full request, median | 47.762 s | 45.657 s | 4.41% |
| Constraints, isolated median | 0.603 s | 0.270 s | 55.27% |
| Feature construction, isolated median | 4.094 s | 3.429 s | 16.24% |
| Full-process peak working set | 2,988,351,488 B | 2,917,179,392 B | 2.38% |
| Full-process peak committed memory | 2,638,503,936 B | 2,560,892,928 B | 2.94% |

Full baseline seconds: `48.183, 47.762, 47.701`. Full candidate seconds:
`45.664, 45.657, 45.011`. Individual category memory controls pass too:
values-only changes peak working set/commit by `-0.42%/-0.51%`; compact-only
by `-0.08%/-0.09%`. Even comparing the largest candidate high-water mark to
the smallest baseline high-water mark, individual growth stays below 0.44%.
High-water marks accumulate across three repetitions in each fresh process.

The factorial controls also pass when each change is added after the other:
constraints improve 50.75%, and features improve 19.15%. In actual complete
scoring runs, constraints improve 45.98% and features 18.46%. The isolated
coordinate control discards unused lists inside the measured helper, so its
timing includes disposal that the original bulk caller performed later.
Both comparisons comfortably exceed the category threshold.

The overall improvement is smaller because model prediction still occupies
38.465 s, about 84% of the final request. These changes remove overhead;
they do not reduce the required tree evaluations.

All three real-scoring repetitions preserve every label/distance digest,
cluster key, membership, and list order. A separate full-workload comparison
checks the bits of every consumed main/nameless feature, with a second
candidate check against dense features. It bypasses model scoring to keep
this additional verification inexpensive; real model output parity comes
from the full scoring runs. The exact cluster dictionary SHA256 remains
`5aa0b47cf472b7e9ab531fb8bff764f2da74fe14df825e1f05a259134075eb04`.

Machine-readable evidence and acceptance checks:
`scratch/bulk_followup_20260904/acceptance.json` and
`summarize_followup.py`. Every benchmark has `report.json`, `phases.log`,
`code_identity.json`, and `completed.json`; all completion records report
exit zero and no Python source changes during their run.

## Candidates and rationale

**Constraint values without coordinates.** FastCluster consumes condensed
distances in triangle order. Returning two Python coordinate lists serves no
purpose for this consumer. A private native entry point shares validation,
pair traversal, constraint arithmetic, and ordered parallel collection with
the existing public method, but omits its coordinate vectors and conversion
to Python integers. The public method remains necessary for square-matrix
consumers and still returns exactly the same tuple.

**Keep supervised features compact.** The previous sparse builder computed
only unconstrained pairs, scattered them into a full NaN-filled matrix,
projected full matrices, then gathered the same scored rows for prediction.
That bridge initially preserved the established scorer interface and its
batching behavior. The private compact mode projects only scored rows and
retains the full label and prediction vectors. It still calculates classifier
call boundaries using the original copy-budget formula, passes independent
writable NumPy-owned copies, and scatters predictions to the same positions.
The dense route is unchanged. This preserves generic classifiers whose
outputs depend on batch boundaries or that mutate their inputs.

Two unconditional shortcuts were rejected at the output-contract gate:

- Transferring owned Rust feature storage to NumPy changes `OWNDATA`, `base`,
  and whether array resizing succeeds. A tiny probe compared the actual
  current feature output with the existing scorer's Rust-storage transfer
  as an ownership proxy; a feature candidate was not rebuilt or timed.
  The current copy provides NumPy-owned storage.
- Caching a failed `psutil` import prevents recovery after a temporary import
  failure. A tiny import-hook probe returns `None`, then valid RAM/RSS values
  with the current code; negative caching keeps returning `None`.
  Repeated probes permit recovery and continue to read fresh memory values.

Probe scripts and JSON evidence are in
`scratch/bulk_followup_20260904/rejected_contract_probes.*`.

## Measurement contract

The workload is the same five largest natural AMiner blocks, 9,267
signatures and 8,762,170 pairs, using the historical v1.21 pairwise models.
The explicit `--limit 2500` truncates none of the selected blocks.
This does not evaluate current-model quality.

The baseline Python files are immutable exports from `a5bfdeb` under
`scratch/bulk_followup_20260904/baseline_current`. Both routes use the same
new release runtime; the baseline invokes the preserved public coordinate
method. This isolates Python path selection and avoids different compiler
or model builds between comparisons.

Whole-path timing includes equal tracing machinery. Category timing comes
from aggregated `rust_featurizer_predict` telemetry, not the last block's
`rust_featurizer_make_dists` entry. Memory comes from Windows
`GetProcessMemoryInfo` high-water counters, not S2AND's fallback estimates.
Exact-feature comparison runs are separate from memory acceptance runs.

All commands use a repository-local uv cache in this sandbox:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) 'scratch/uv-cache'
uv run --no-sync python scratch/bulk_followup_20260904/profile_compare.py --dataset aminer --limit 2500 --blocks 5 --repeat 3 --runtime scratch/bulk_followup_20260904/runtime --baseline --output scratch/bulk_followup_20260904/full_baseline
```

Omit `--baseline` for the candidate. `--full-features` restores the expanded
sparse matrices; `--coordinates` restores unused coordinate generation.
Their combinations isolate the two changes. `--skip-scoring` removes model
time from category experiments. `--capture-consumed --check-features`
enables the separate bit-exact feature checks.

## Verification and build

The release wheel was built with:

```powershell
uv run --no-sync maturin build --release --manifest-path s2and_rust/Cargo.toml --out scratch/bulk_followup_20260904/wheels
```

It is extracted under `scratch/bulk_followup_20260904/runtime`; the default
local extension was not replaced. Tests explicitly preload this runtime
before importing S2AND. The final regression run passed **219 tests** in
10.07 s. Its exact test selection is the nine files covering bulk sparse
features, constraint values/labels, distance orchestration/parity, tiled
LightGBM scoring/parity, Rust batch chunking, and feature-port parity.
`regression_final.log` records the outcome. The initial broader run exposed
one stale mock targeting the old coordinate helper; it was updated to test
the same constraint-count rejection through the new values-only boundary.

Ruff lint and format checks, `ty check s2and/model.py s2and/rust_calls.py`,
`cargo fmt --check`, and `git diff --check` pass. Subsequent production-file
edits after benchmarks were comments and formatting only.
