# Release, artifact, and test simplification

Status date: 2026-07-27

Status: approved for implementation. This is a temporary implementation plan,
not a release authority. Delete it after the final runbook and release note
exist.

## Outcome

One trusted owner should be able to build, evaluate, diagnose, and publish
v1.3 from a clean checkout without learning a private protocol.

The implementation should shrink. Record additions and deletions against the
`origin/main` merge base after every step. A simplification step that grows the
tree materially stops for review; tests and a small explicit handle may add
lines, but new frameworks, compatibility layers, ledgers, and evidence schemas
are out of scope.

## Rules

Keep a check only when removing it can cause:

1. a silently wrong scientific result;
2. an unsafe binary read; or
3. an irreversible bad publication.

Validate an independently replaceable artifact when it is produced, installed,
or first opened. Reuse the validated object. Do not repeatedly hash or rescan
it while that object lives.

Keep:

- held-out isolation and validation-only EPS selection;
- canonical-name, Arrow, name-count, feature-order, and model bindings;
- linker binding to the pairwise feature contract and both booster digests;
- Python/Rust prediction parity;
- quality, subblocking, runtime, RSS, packaging, and installed-runtime gates;
- the external complete-v5 model bundle; and
- `scripts/run_ci_locally.py` as the executable CI authority.

Do not add:

- a generic artifact or schema framework;
- a second slow/parity lane;
- transported evidence protocols;
- compatibility readers for unreleased formats; or
- another status ledger.

## Fixed decisions

- Complete the simplification before generating final production artifacts.
- Keep separate `s2and` and `s2and-rust` distributions.
- Support Python 3.11, 3.12, and 3.13 in both distributions.
- Replace six release inputs with three write-once plans.
- Introduce one fixed-root `ArrowDataset` handle with no compatibility layer.
- Retain the Arrow manifest's `artifact_generation.generation_id`; remove only
  duplicate caller/cache identity.
- Preserve historical profiling snapshots verbatim.
- Run an independent audit agent after every completed step.

## Step 1: correct the release plan

The separate blocker ledger is retired; remaining work lives in the runbook.

Keep only four unfinished prerequisites:

1. join canonical benchmark names into the fields consumed by `ANDData`;
2. generate split assignments grouped by `base_group_id`;
3. generate and package the final ORCID runtime assets; and
4. make pairwise training fail without finite selected validation metrics.

Treat version selection, historical baseline measurement, Arrow regeneration,
final linker fitting, installed smoke, and release evaluation as ordinary
runbook steps.

Delete two false blockers:

- EPS changes only cluster configuration. It does not invalidate linker
  features or require a new linker fit.
- `pys2` is an operator-side query client, not a runtime or correctness
  contract. Use any working authenticated query/export path and validate the
  returned columns, bounds, counts, and spot checks.

Acceptance:

- one dependency-ordered runbook;
- no blocker ledger or blocker IDs;
- no EPS-to-linker invalidation claim;
- no mandatory database-client rewrite; and
- all local links and anchors resolve.

## Step 2: freeze a reproducible baseline

Record an exact source snapshot before behavioral simplification:

- source/tree identity and `uv.lock` SHA-256;
- Python, Rust, LightGBM, CPU, and thread settings;
- merge-base additions/deletions and total tracked/untracked text lines;
- exact commands and complete logs.

Run:

```powershell
uv lock --check
uv run pytest -q --durations=40
```

The test run is long. Launch it with a durable log, PID/completion record, and
bounded monitoring. This is a simplification baseline, not the final release
commit.

## Step 3: bound the parity workload

Fix the cap-followed-by-unbounded-union defect in
`tests/test_rust_lightgbm_booster_parity.py`.

The default workload must:

- enforce hard split and row budgets;
- cover represented combinations of missing type, default direction,
  zero-threshold regime, and reachable below/equal/above neighbors;
- spread remaining selections deterministically;
- pin explicit LightGBM and Rust thread counts; and
- retain one focused one-thread-versus-four-thread identity assertion.

Use the same bounded workload locally, in CI, and for release validation. Do
not add a marker, default exclusion, environment flag, or second exhaustive
lane. Replace the oversized pair-sampling self-oracle with a tiny test of the
exact train, validation, and test arguments routed to the sampler.

Rerun the focused tests and full duration command. Report measured, not
predicted, deltas.

## Step 4: delete completed Rust migration machinery

Keep the promoted incremental Arrow profiler because it produces the release
performance report. Move it beside the other verification report producers and
inline only the helpers it uses.

Delete the dispatcher, migration comparisons, transfer/large-block/stress
tools, memory calibrators, telemetry summarizer, tool-only tests, and live docs
that teach those retired commands. Keep `s2and/memory_budget.py` runtime
telemetry and the public workload helper used by retained tests.

Do not edit dated profiling snapshots. Append a new retirement note to release
notes rather than rewriting history.

Acceptance:

- retained profiler schema still passes its consumer tests;
- no live import or operator doc references a retired command; and
- the step is strongly net-negative in lines.

## Step 5: use three write-once release plans

Use a fresh run directory:

```text
run/
  release.json
  model_plan.json
  evaluation_plan.json
  stages/
  reports/
  final/
```

- `release.json` is the reviewed owner input.
- `model_plan.json` contains only training/validation inputs and
  validation-only EPS policy.
- `evaluation_plan.json` contains held-out inputs, populations, baselines,
  gates, and the performance Arrow root and workload.

Preparation may inspect held-out identity only for overlap checks. Training and
calibration of the pairwise model receive only `model_plan.json`; they contain
no code that probes its sibling files. The linker's separate source-bundle and
target contract is unchanged. The complete model is fixed and reloaded before
held-out pairwise and cluster evaluation receives `evaluation_plan.json`.

All three are fresh snapshots. Changes start a new run; later stages never
reread mutable source configuration.

Delete the six old operator-input schemas and their bespoke error-string tests.
Keep small typed loaders local to their commands and normative maximum gates in
code.

## Step 6: introduce one Arrow dataset handle

The public shape is:

```python
with ArrowDataset.open(root) as dataset:
    ...
```

The immutable handle owns validated Arrow readers, batch indexes, native
name-count state, and deterministic cleanup. Concurrent reads are supported;
close during active use is rejected.

Training, full prediction, incremental prediction, evaluation, validators, and
both installed smokes consume the handle. Request-local seed sidecars remain
explicit arguments.

Predictions must use retained validated state, not reopen mutable paths.
Migrate every internal caller before deleting:

- caller-assembled base path maps;
- caller-supplied generation identity;
- `_RUNTIME_ARROW_GENERATION_CACHE`; and
- unsupported path aliases.

Keep first-open schema, normalization, content-identity, batch-index, name-count
layout, and model-binding checks. Leave the separate `NameCountsIndex` cache
alone.

Verify cold open, repeated use, six-root lifecycle, cleanup, concurrency,
post-open path replacement, corruption, same-size substitution, p50 runtime,
and RSS.

## Step 7: finish packaging support

Keep the two distributions and their real platform build boundaries. Confirm
Python 3.11-3.13 in metadata, dependency resolution, CI, installed smokes, and
documentation.

Preserve Rust-before-Python publication, the tested `main`-ref guard, and the
protected PyPI environment. Simplify duplicated commands and prose, not the
platform matrix.

## Step 8: finish the four prerequisites

Use bounded fixtures before full data or training work.

1. Canonical benchmark export:
   join by signature ID into the exact `ANDData` name fields; reject duplicate
   or missing keys; verify `ANDData -> Arrow -> featurization`; review join and
   divergence counts.
2. Leakage-free assignments:
   assign whole `base_group_id` values to one split; freeze one CSV; run the
   existing disjointness validator.
3. Runtime assets:
   generate the tuple-bound ORCID JSON and minimal manifest; declare them as
   package data; verify a real wheel and sdist.
4. Pairwise selection:
   fail on missing/non-finite selected validation metrics and emit only the
   selected metrics, parameters, and row counts. Keep search/resource detail in
   ordinary logs.

No new provenance schemas or evidence protocols.

## Step 9: consolidate and verify

Create `docs/release.md` as the sole release policy and runbook. Move durable
normalization rules to their owning format/runtime documents, accounting for
every heading before deleting the migration document. Retain manual
adjudication evidence and historical snapshots.

Delete:

- `docs/1_3_release_todo.md`;
- `docs/normalization_migration_blocked.md`; and
- this plan after its outcome is recorded in release notes.

Final verification:

```powershell
uv lock --check
uv run --no-project python scripts/run_ci_locally.py
uv run pytest -q --durations=40
git diff --check
```

Run the promoted performance evaluator on the exact candidate. Resolve the two
unconditional production-evaluation skips with canonical-v2 assertions or
equivalent unskipped release-smoke assertions.

Finally, have a cold reader follow only `docs/release.md` on bounded fixtures.
Every required clarification is a documentation defect.

## Invalidation

- Data changes invalidate consumers of that data.
- Pairwise booster, feature-contract, linker-source, or linker-target changes
  require fresh linker materialization and fitting.
- EPS-only changes do not invalidate linker features or the linker fit.
- Population, metric, baseline, threshold, or performance-workload changes
  invalidate their reports; changing them after seeing held-out results
  requires a genuinely untouched holdout.
- Package content or version changes require rebuilt distributions and both
  installed smokes.
- Behavior-changing source changes invalidate every affected downstream stage.

## Completion

The work is complete when the nine audited steps pass, final artifacts are
generated only from the simplified interfaces, the release gates pass, one
document explains the release, and additions have not replaced the complexity
we removed.
