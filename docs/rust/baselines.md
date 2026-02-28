# Rust Operational Baselines

Status date: 2026-02-28

This is the source of truth for reproducible Rust gate artifacts.

## Evidence policy (local-only for now)

Baseline JSONs and logs are treated as **local evidence** and should live under `scratch/`
(already gitignored).

"Promoting" a baseline means:
- Rerun the canonical gate command (below) on a clean tree with a release build of the Rust extension.
- Record: command line, `workload_id` (when present), and key deltas (seconds/RSS/parity) here.
- Optionally record the local `scratch/...json` filename for convenience.

See also: `docs/work_plan.md` (ops quick-hits section).

## Active baselines (2026-02-27 refresh)

| Gate | Artifact(s) | Result summary | Status |
| --- | --- | --- | --- |
| Inference comparator parity + latency + RSS | `scratch/baselines_20260227/compare_inspire_5k_20260227.json` | python `18.724s` / `1.510 GB`; rust `4.734s` / `0.917 GB`; feature parity pass; speedup `3.955x`; RSS delta `-39.27%` | Active |
| Maintained transfer-mini full workload (acceptance baseline) | `scratch/baselines_20260227/profile_transfer_mini_full_20260227.json` | workload_id `3291489010b58481a15d209d5c5bb3ed764af109709d4d6ffc4c1ed617a95128`; python `304.639s` / `5.477 GB`; rust `159.928s` / `5.654 GB`; B3 F1 `0.960` both; speedup `1.905x`; RSS delta `+3.23%` | Active |
| Transfer-mini smoke workload (sanity only) | `scratch/baselines_20260227/profile_transfer_mini_smoke_20260227.json` | workload_id `3f09cde4b4eb4ed4956f6b147fe0c68a82eebc550e7ad5c84cc0c58a43eb3ec2`; python `113.046s` / `5.265 GB`; rust `47.753s` / `5.076 GB`; B3 F1 `0.291` both | Active (sanity) |
| Largest-block compare smoke (canonical CLI path + output-shape gate) | `scratch/baselines_20260227/largest_block_compare_smoke_200_20260227.json` | `dataset=aminer`, `block='j wang'`, `max_block_size=200`; cluster equivalent `True`, signature diff `0/200`; python `375.726s` / `13.652 GB`; rust `326.999s` / `9.855 GB`; predict speedup `4.051x` | Active |
| Stress rebuild RSS-series baseline (`from_json_paths`) | `scratch/baselines_20260227/stress_rust_from_json_paths_aminer_6x_20260227.json` | `6/6` success; `rss_peak_gb_by_iteration=[6.836578, 6.855633, 6.862076, 6.86729, 6.868664, 6.87191]`; `rss_growth_fraction=0.005168` | Active |
| Stress rebuild threshold-enforcement check | `scratch/baselines_20260227/stress_rust_from_json_paths_aminer_2x_gate_20260227.json` | `--rss-growth-max-fraction 0.05` pass; `rss_growth_fraction=0.002552`; `rss_growth_gate_pass=True` | Active (verification) |

## Supporting evidence (non-gating)

These artifacts are useful for design justification and capacity planning, but are not
promotion gates.

| Probe | Artifact(s) | Result summary | Status |
| --- | --- | --- | --- |
| Inspire 100k-slice ANDData build probe (training-preprocess defer justification) | `scratch/compare_inspire_100k_anddata_build.json` | `--dataset inspire --limit 100000 --pair-count 1000 --n-jobs 4`; `anddata_build_seconds`: python `316.705s`, rust `33.781s` (`-89.33%`, `9.38x` faster); total runtime `317.503s -> 71.615s`; peak RSS `16.267 GB -> 9.563 GB` | Supporting (2026-02-28) |
| Bundle 1/2 transfer-mini validation (full preset) | `scratch/profile_transfer_mini_bundle1_4_20260228.json` | workload_id `3291489010b58481a15d209d5c5bb3ed764af109709d4d6ffc4c1ed617a95128`; python `296.600s` / `5.491 GB`; rust `176.469s` / `4.794 GB`; `post_rust_cleanup` present (`4.188 GB`); pairwise-fit ratios pass (`1.163`, `1.053` <= `1.25`); RSS cleanup delta `0.113 GB` | Supporting (2026-02-28) |
| Transfer-mini latency diagnostics matrix (cleanup/preprocess toggles) | `scratch/diagnostics/transfer_mini_diag_{default,default_r2,default_r3,no_cleanup,force_py_papers,force_py_papers_no_cleanup}_20260228.json` | Default deferred-paper runs keep peak RSS stable (`4.787-4.795 GB`) but show large single-run latency spread (`153.828s`, `172.326s`, `200.574s`) driven by LightGBM trial wall time; parameter hashes + fitted tree counts remain constant. Forcing Python paper preprocessing raises build time (`~8.7s -> ~27.8-29.4s`), peak RSS (`4.795 GB -> 5.601-5.662 GB`), and runtime (`207.732-211.735s`). | Supporting (2026-02-28) |
| Bundle 4 calibration broadening (Phase A + Rust batch) | `scratch/calibrate_phase_a_shape_{4000_l5_pairbuffix,10000_p2p4,14995_l5_pairbuffix}_20260228.json`, `scratch/calibrate_rust_batch_shape_{4000_l5_overhead,4000_l5_pairbuffix,14995_l5_pairbuffix}_20260228.json` | Phase-A recommended bytes `163`, `192`, `151` (legacy-overhead outlier file reports `461`); Rust-batch recommended persistent row overhead bytes `37`, `37`, `49` | Supporting (2026-02-28) |
| Bundle 3 (P3/P4) independent validation — transfer-mini | `scratch/profile_transfer_mini_p3p4_20260228.json` | workload_id `3291489010b58481a15d209d5c5bb3ed764af109709d4d6ffc4c1ed617a95128`; python `299.0s` / `5.520 GB`; rust `177.0s` / `4.790 GB`; quality parity (B3 `0.960`, Cluster `0.976`, ClusterMacro `0.933` identical); `1.69x` speedup, `-13.2%` RSS | Supporting (2026-02-28) |
| Bundle 3 (P3/P4) independent validation — inference | `scratch/compare_p3p4_inspire5k_20260228.json` | python `21.671s` / `1.506 GB`; rust `5.014s` / `0.922 GB`; feature parity pass; `4.32x` speedup, `-38.8%` RSS | Supporting (2026-02-28) |

## Policy updates (effective 2026-02-27)

1. Transfer-mini acceptance decisions (RSS/quality/latency) must use the full workload baseline only.
2. Transfer-mini smoke remains a fast sanity run and is not acceptance evidence.
3. Largest-block gates are run only via `scripts/rust_suite.py`.
4. Stress RSS growth claims must cite artifacts with `rss_peak_gb_by_iteration` and `rss_growth_fraction`.

## Workload identity enforcement

Transfer-mini artifacts now include `workload` and `workload_id`.

Observed gate behavior:
- Mismatched workload IDs hard-fail:
  - baseline: `profile_transfer_mini_smoke_20260227.json`
  - current: `profile_transfer_mini_full_20260227.json`
  - error: `Workload mismatch between baseline and current artifacts...`
- Matching workload IDs pass:
  - baseline/current: `profile_transfer_mini_full_20260227.json`
  - output: `violations: 0`

## Build and release pinning

Rust-backed artifacts above include `rust_extension_identity` with:
- module file path
- sha256 / size / mtime
- `build_info` (`debug_assertions`, version, profile metadata)

Baseline capture policy:
1. Build extension in release mode:
   - `uv run maturin develop -m s2and_rust/Cargo.toml --release`
2. Run rust-backed gates with:
   - `--require-rust-release 1`
3. Do not promote artifacts where `build_info.debug_assertions=true`.

## Canonical gate commands

Build prep:
- `uv run maturin develop -m s2and_rust/Cargo.toml --release`

1. Inference comparator:
   - `uv run --no-project python scripts/rust_suite.py compare --dataset inspire --limit 5000 --pair-count 5000 --n-jobs 4 --require-non-dev-rust 0 --require-rust-release 1 --write-json scratch/baselines_20260227/compare_inspire_5k_20260227.json`
2. Transfer-mini smoke sanity:
   - `uv run --with psutil python scripts/rust_suite.py transfer-mini --mode compare --preset smoke --target kisti --n-jobs 2 --n-train-pairs 300 --n-iter 1 --require-rust-release 1 --write-json scratch/baselines_20260227/profile_transfer_mini_smoke_20260227.json`
3. Transfer-mini full acceptance baseline:
   - `uv run --with psutil python scripts/rust_suite.py transfer-mini --mode compare --preset full --target kisti --n-jobs 4 --n-train-pairs 10000 --n-iter 5 --require-rust-release 1 --write-json scratch/baselines_20260227/profile_transfer_mini_full_20260227.json`
4. Largest-block compare smoke:
   - `uv run --no-project python scripts/rust_suite.py largest-block --mode compare --dataset aminer --block "j wang" --n-jobs 4 --max-block-size 200 --timeout-hours 0.5 --require-rust-release 1 --write-json scratch/baselines_20260227/largest_block_compare_smoke_200_20260227.json`
5. Stress rebuild RSS-series baseline:
   - `uv run --with psutil python scripts/rust_suite.py stress-rebuild --dataset aminer --build-path from_json_paths --repeats 6 --num-threads 1 --rss-sample-ms 50 --require-rust-release 1 --write-json scratch/baselines_20260227/stress_rust_from_json_paths_aminer_6x_20260227.json`
6. Stress rebuild threshold-enforcement check:
   - `uv run --with psutil python scripts/rust_suite.py stress-rebuild --dataset aminer --build-path from_json_paths --repeats 2 --num-threads 1 --rss-sample-ms 50 --rss-growth-max-fraction 0.05 --require-rust-release 1 --write-json scratch/baselines_20260227/stress_rust_from_json_paths_aminer_2x_gate_20260227.json`
7. Supporting only (not an acceptance gate): inspire 100k ANDData-build probe
   - `uv run --with psutil python scripts/rust_suite.py compare --mode compare --dataset inspire --limit 100000 --pair-count 1000 --n-jobs 4 --require-rust-release 1 --write-json scratch/compare_inspire_100k_anddata_build.json`

## Artifact provenance policy

Required checks for promotion:
1. `run_metadata.git_commit` is non-null.
2. `run_metadata.git_dirty` is `false` for release-grade baselines.
3. `run_metadata.env` captures non-default runtime knobs used in the run.

Note: the refreshed artifacts above were captured from a dirty working tree (`git_dirty=true`) and are verification-grade, not release-grade.
