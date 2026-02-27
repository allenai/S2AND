# Rust Operational Baselines

Status date: 2026-02-26

This is the single source of truth for Rust runtime benchmark baselines.

## Active baselines

| Gate | Artifact(s) | Result summary | Status |
| --- | --- | --- | --- |
| Inference comparator parity + latency + RSS | `scratch/compare_investigate_20260224.json` | python `96.410s` / `1.509 GB`; rust `53.480s` / `1.028 GB`; parity pass; speedup `1.803x`; RSS delta `-31.88%` | Active |
| Maintained train/eval RSS gate (3 seeds, `n_jobs=4`) | `scratch/profile_transfer_mini_compact_cd_seed1_njobs4_20260224.json`<br>`scratch/profile_transfer_mini_compact_cd_seed2_njobs4_20260224.json`<br>`scratch/profile_transfer_mini_compact_cd_seed3_njobs4_20260224.json` | Seed deltas: `+2.2%`, `-1.5%`, `+1.0%` Rust RSS vs Python; quality parity pass | Active |
| Big-block phase-split parity | `scratch/big_block/compare_phase_split_10k_seed43_python_20260224.json` | `cluster_equivalent=True`, partition diff `0/10000` | Active |

## Latest checks (2026-02-26, not promoted)

These are current-code verification runs. They are intentionally separate from the active baseline table until all maintained gates are green together.

| Gate/check | Artifact(s) | Result summary | Promotion status |
| --- | --- | --- | --- |
| Phase 0 fixed workload (telemetry semantics gate) | `scratch/big_block/phase0_memacc_rust_20260225_104213_0b3e877.json`<br>`scratch/big_block/phase0_memacc_rust_20260225_104213_0b3e877.log` | `phase_split_phase_a`: `prediction_error_ratio=0.700`, `underpredicted=False` (passes target). Rust-batch ratios: min/max/avg `0.877/1.030/0.919` (one tiny-sample underprediction). | Candidate |
| Big-block memory gate (post follow-up, 14995 sigs) | `scratch/big_block/memacc_l5_pairbuffix_14995_rust_20260226.json`<br>`scratch/big_block/memacc_l5_pairbuffix_14995_rust_20260226.log`<br>`scratch/big_block/phase_a_calibration_l5_pairbuffix_14995_20260226.json`<br>`scratch/big_block/rust_batch_calibration_l5_pairbuffix_14995_20260226.json` | Phase A: `prediction_error_ratio=0.852`, `underpredicted=False`. Rust batch: ratio min/p50/p95/max `0.316/0.924/0.968/0.976`, underpredicted `0/102`. Peak RSS `2.489 GB` (`n_jobs=8`). | Candidate |
| Big-block phase-split parity (post follow-up, 10k) | `scratch/big_block/compare_phase_split_l5_mem_tune_20260226.json` | `cluster_equivalent=True`, partition diff `0/10000`. Baseline: `3221.148s` / `10.815 GB`; candidate: `3574.904s` / `1.922 GB`; runtime delta `+353.756s`, peak RSS delta `-8.893 GB`. | Candidate |
| Inference comparator parity + latency + RSS (post L5) | `scratch/compare_l5_mem_tune_final_20260226.json` | python `27.425s` / `1.509 GB`; rust `16.454s` / `0.921 GB`; parity pass; speedup `1.667x`; RSS delta `-38.97%`. | Candidate |
| Startup 3-probe calibration smoke (forced threshold) | `scratch/compare_probe3_calibration_smoke_20260226.json`<br>`scratch/compare_probe3_calibration_smoke_20260226_b.json` | parity pass in both. Latest rerun (`..._b`): python `22.820s` / `1.510 GB`; rust `19.281s` / `0.924 GB`; speedup `1.184x`; RSS delta `-38.81%`. | Informational |
| L2 save-outside-lock comparator before/after (`inspire`, 5k/5k) | `scratch/compare_save_outside_lock_before.json`<br>`scratch/compare_save_outside_lock_after.json` | Parity pass in both (`feature_parity.pass=True`). Before: python `22.410s` / `1.507 GB`, rust `13.621s` / `0.923 GB`. After: python `18.496s` / `1.508 GB`, rust `11.961s` / `0.918 GB`. | Candidate |
| Featurizer reuse microbench (post L5, `kisti`, repeats=3) | `scratch/profile_rust_featurizer_reuse_l5_mem_tune_final_20260226.json` | `same_object.iterations[*].featurizer_build_count=[1,1,1]`; same-object mean `16.149s`; reinstantiated mean `20.013s`; reinstantiation penalty `+3.864s`. | Candidate |
| Maintained mini-transfer gate (post L5, `kisti/arnetminer/zbmath`) | `scratch/profile_transfer_mini_l5_mem_tune_final_20260226.json` | Quality triplets equal at shown precision. Python `417.353s` / `5.506 GB`; Rust `258.829s` / `5.620 GB` (speedup `1.612x`, RSS `+2.07%`). | Candidate |
| Maintained mini-transfer gate (phase0_memacc snapshot, historical) | `scratch/profile_transfer_mini_phase0_memacc_20260225_105000_0b3e877.json`<br>`scratch/profile_transfer_mini_phase0_memacc_20260225_105000_0b3e877.log` | Quality parity equal, but regression vs Python: runtime `259.672s -> 486.143s` (`+87.2%`), peak RSS `4.798 GB -> 5.564 GB` (`+16.0%`). | Blocked |
| Big-block parity command smoke fallback | `scratch/big_block/compare_phase_split_small_phase0_memacc_retry_20260225_0b3e877.json` | 4k fallback sanity run: `cluster_equivalent=True`, partition diff `0/4000`, runtime delta `-15.456s`, peak RSS delta `-1.501 GB`. | Informational |

Command drift/fail logs captured during the same update:
- `scratch/big_block/compare_phase_split_phase0_memacc_20260225_110303_0b3e877.log` (legacy subset CLI arg no longer supported).
- `scratch/big_block/compare_phase_split_phase0_memacc_20260225_110315_0b3e877.log` (`--full-run` required for `>4000` signatures).

## Drift controls

1. Do not treat `docs/archive/*` as release-gating evidence.
2. Do not treat `scratch/*now*`, `scratch/*investigate*`, or ad-hoc files as active baselines unless promoted here.
3. Promotion to active baseline requires:
   - matching gate command and workload shape,
   - parity/latency/RSS gate outcomes recorded,
   - artifact path added to this file,
   - Various MD docs updated.

## Artifact provenance policy

Benchmark JSON artifacts produced by these scripts now include `run_metadata`:
- `scripts/rust_suite.py compare`
- `scripts/rust_suite.py transfer-mini`
- `scripts/rust_suite.py prod-inference`

Required checks for new baseline promotions:
1. `run_metadata.git_commit` is non-null.
2. `run_metadata.git_dirty` is `false` for release-grade baselines.
3. `run_metadata.env` captures non-default runtime knobs used in the run.

Legacy note: current active artifacts were created before `run_metadata` was added. Refresh them on the next promotion cycle.

## Canonical gate commands

Build-prep (required before recording baselines):
- `uv run maturin develop -m s2and_rust/Cargo.toml --release`

1. Inference comparator:
   - `uv run --no-project python scripts/rust_suite.py compare --dataset inspire --limit 5000 --pair-count 5000 --n-jobs 8 --require-non-dev-rust 0 --require-rust-release 1 --write-json scratch/compare_<label>.json`
2. Maintained mini-transfer gate:
   - `uv run --with psutil python scripts/rust_suite.py transfer-mini --mode compare --preset full --n-jobs 4 --n-train-pairs 10000 --n-iter 5 --target kisti --require-rust-release 1 --write-json scratch/profile_transfer_mini_<label>.json`
3. Big-block parity gate:
   - `uv run --no-project python scripts/rust_suite.py big-block-incremental --mode compare_phase_split --backend python --subset-dir scratch/inventors_topblock_15k --total-signatures 10000 --seed-signatures 7500 --seed-cluster-count 1200 --batching-threshold 7500 --n-jobs 8 --random-seed 43 --require-rust-release 1 --full-run --write-json scratch/big_block/compare_phase_split_<label>.json`

## Historical archive

- `docs/archive/README.md`

