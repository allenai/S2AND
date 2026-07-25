# Rust Operational Baselines

Status date: 2026-05-28 (latest snapshot: [`profiling/2026-05-28-promoted-incremental-arrow.md`](profiling/2026-05-28-promoted-incremental-arrow.md))

Command review date: 2026-07-24.

This doc is the operator guide for rerunning Rust promotion gates.
Local JSON artifacts under `scratch/` are the raw evidence for a run but are
gitignored. A dated Markdown snapshot is the durable record of the command,
environment, metrics, and interpretation worth retaining.

The retained 2026-05-28 snapshot came from a dirty worktree and is historical
evidence, not the pinned v1.3 Stage-8 baseline. For the release, use the
commands here but write JSON, logs, environment, and hardware evidence to the
durable reports root required by
[../1_3_release_todo.md](../1_3_release_todo.md), then bind it into the
immutable quality report. The release runbook owns acceptance and promotion.

---

## Artifact conventions

- Development gate JSON may be local raw evidence under `scratch/` (gitignored).
- Write non-release outputs under `scratch/baselines_YYYYMMDD/`. Release
  outputs go under the runbook's durable `$Reports` root.
- Promotion-grade runs should include `workload_id` when present (in the JSON).
- Release-grade promotion still requires `run_metadata.git_dirty=false`.

---

## Profiling snapshots (historical evidence)

Profiling snapshots are dated Markdown files under `profiling/YYYY-MM-DD.md`. Each snapshot captures
one gate refresh at a point in time (environment, commands, artifact paths, and any noteworthy interpretation).

When refreshing gates:
1. Write development JSON under `scratch/baselines_YYYYMMDD/`, or release JSON
   under the durable runbook reports root.
2. Add a new snapshot file under `profiling/YYYY-MM-DD.md` referencing those artifacts.
3. Update the `Status date` (and `latest snapshot` link) at the top of this doc.

### Snapshots

| Date | Highlights |
|---|---|
| [2026-05-28](profiling/2026-05-28-promoted-incremental-arrow.md) | Release-build comparison retaining the measured rationale for memory-mapping the name-count index; not a release-grade promotion result because the worktree was dirty. |

---

## Canonical gate commands

Build first (develop mode is slower, so use release mode for gates):
```
uv run maturin develop -m s2and_rust/Cargo.toml --release
```

Replace `REVIEWED_ARTIFACT_ROOT` below with `scratch` for a development run or
with the durable v1.3 `$Reports\rust` directory for a release run. Record the
expanded literal paths in the launch record.

Capture run logs and structured memory telemetry with rust-suite global options
(these must appear before the command name):
```
--log-file REVIEWED_ARTIFACT_ROOT/baselines_YYYYMMDD/<run>_YYYYMMDD.log
--memory-telemetry-jsonl REVIEWED_ARTIFACT_ROOT/baselines_YYYYMMDD/<run>_memory_telemetry_YYYYMMDD.jsonl
```

Optional: summarize memory prediction telemetry from the JSONL artifact:
```
uv run python scripts/rust_suite.py summarize-memory-telemetry REVIEWED_ARTIFACT_ROOT/baselines_YYYYMMDD/<run>_memory_telemetry_YYYYMMDD.jsonl --write-json REVIEWED_ARTIFACT_ROOT/baselines_YYYYMMDD/<run>_memory_telemetry_YYYYMMDD.json
```

**1. Promoted incremental Arrow profile**
```
uv run python scripts/rust_suite.py promoted-incremental-arrow-profile \
  --arrow-root /path/to/s2and_and_big_blocks_linker_dataset_v1_3 \
  --dataset pubmed --model-path /path/to/production_model_v1.3 \
  --query-limit 25 --max-seed-clusters 25 --runs 5 \
  --synthetic-seeds-when-clusters-missing \
  --require-rust-release \
  --output-dir REVIEWED_ARTIFACT_ROOT/promoted_incremental_arrow_profile \
  --write-json REVIEWED_ARTIFACT_ROOT/promoted_incremental_arrow_profile/pubmed.json
```

This is the current production-inference workload shape, not a frozen numeric
target. Build the Rust extension in release mode, use a clean worktree, record
hardware and input identities, and compare against the owner-approved pinned
baseline before treating numbers as release-grade. The v1.3 command is not
runnable until B09 regenerates the canonical linker replay Arrow root.

**2. Inference comparator**
```
uv run --no-project python scripts/rust_suite.py compare \
  --dataset inspire --limit 5000 --pair-count 5000 --n-jobs 4 \
  --require-non-dev-rust 0 --require-rust-release 1 \
  --write-json REVIEWED_ARTIFACT_ROOT/baselines_YYYYMMDD/compare_inspire_5k_YYYYMMDD.json
```

**3. Transfer-mini full**
```
uv run --with psutil python scripts/rust_suite.py transfer-mini \
  --mode compare --preset full --target kisti \
  --n-jobs 4 --n-train-pairs 10000 --n-iter 5 --require-rust-release 1 \
  --write-json REVIEWED_ARTIFACT_ROOT/baselines_YYYYMMDD/profile_transfer_mini_full_YYYYMMDD.json
```

**4. Stress rebuild (6x)**
```
uv run --with psutil python scripts/rust_suite.py stress-rebuild \
  --dataset aminer \
  --repeats 6 --num-threads 1 --rss-sample-ms 50 --require-rust-release 1 \
  --write-json REVIEWED_ARTIFACT_ROOT/baselines_YYYYMMDD/stress_rust_from_arrow_paths_aminer_6x_YYYYMMDD.json
```

`stress-rebuild` is Arrow-only; classic `ANDData` lifecycle comparisons should
use Python featurization tests.
