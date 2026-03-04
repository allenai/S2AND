# Rust Operational Baselines

Status date: 2026-03-02 (latest snapshot: [`profiling/2026-03-02.md`](profiling/2026-03-02.md))

This doc is the operator guide for rerunning Rust promotion gates.
The JSON artifacts under `scratch/` are the source of truth; avoid copying full metric tables into Markdown.

---

## Artifact conventions

- Gate JSON artifacts are local evidence under `scratch/` (gitignored).
- Write outputs under `scratch/baselines_YYYYMMDD/`.
- Promotion-grade runs should include `workload_id` when present (in the JSON).
- Release-grade promotion still requires `run_metadata.git_dirty=false`.

---

## Profiling snapshots (historical evidence)

Profiling snapshots are dated Markdown files under `profiling/YYYY-MM-DD.md`. Each snapshot captures
one gate refresh at a point in time (environment, commands, artifact paths, and any noteworthy interpretation).

When refreshing gates:
1. Write JSON artifacts under `scratch/baselines_YYYYMMDD/`.
2. Add a new snapshot file under `profiling/YYYY-MM-DD.md` referencing those artifacts.
3. Update the `Status date` (and `latest snapshot` link) at the top of this doc.

### Snapshots

| Date | Highlights |
|---|---|
| [2026-03-02](profiling/2026-03-02.md) | Gate rerun refresh snapshot (commands + artifact paths). |

---

## Canonical gate commands

Build first (develop mode is slower, so use release mode for gates):
```
uv run maturin develop -m s2and_rust/Cargo.toml --release
```

Capture a log for every gate run (stdout + stderr):
- PowerShell: append `*> scratch/baselines_YYYYMMDD/<run>_YYYYMMDD.log`
- bash/zsh: append `> scratch/baselines_YYYYMMDD/<run>_YYYYMMDD.log 2>&1`

Optional: summarize memory prediction telemetry from a run log:
```
uv run python scripts/rust_suite.py summarize-memory-telemetry scratch/baselines_YYYYMMDD/<run>_YYYYMMDD.log --write-json scratch/baselines_YYYYMMDD/<run>_memory_telemetry_YYYYMMDD.json
```

**1. Inference comparator**
```
uv run --no-project python scripts/rust_suite.py compare \
  --dataset inspire --limit 5000 --pair-count 5000 --n-jobs 4 \
  --require-non-dev-rust 0 --require-rust-release 1 \
  --write-json scratch/baselines_YYYYMMDD/compare_inspire_5k_YYYYMMDD.json
```

**2. Transfer-mini full**
```
uv run --with psutil python scripts/rust_suite.py transfer-mini \
  --mode compare --preset full --target kisti \
  --n-jobs 4 --n-train-pairs 10000 --n-iter 5 --require-rust-release 1 \
  --write-json scratch/baselines_YYYYMMDD/profile_transfer_mini_full_YYYYMMDD.json
```

**3. Stress rebuild (6x)**
```
uv run --with psutil python scripts/rust_suite.py stress-rebuild \
  --dataset aminer --build-path from_json_paths \
  --repeats 6 --num-threads 1 --rss-sample-ms 50 --require-rust-release 1 \
  --write-json scratch/baselines_YYYYMMDD/stress_rust_from_json_paths_aminer_6x_YYYYMMDD.json
```
