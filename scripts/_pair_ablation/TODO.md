# TODO

## Promote `big7_1250_v1` into production training

- [x] Declare the complete recipe in `scripts/production/model/train_pairwise.py`.
- [x] Move deterministic balanced-linker selection and no-heldout recipe
  resolution into `s2and/pairwise_training.py`, shared with the ablation
  assembler.
- [x] Verify the production realization contract: seven 100,000-pair gold
  sources plus 625 positive and 625 negative pairs from each of seven big
  linker blocks, for 708,750 nominal rows.
- [ ] When the updated canonical Arrow datasets and name-count generation land,
  route the exact resolved rows through maintained Arrow featurization.
- [ ] Record the resolved recipe in `pairwise_training_config.json` and the
  realized source counts, selection audit, and pair digest in
  `pairwise_training_summary.json`.
- [ ] Run a tiny fixture/smoke training first, then the full production job.
  The extant legacy big-block bundle is for practice only and must not be used
  to build a release.

## Gate the seed-3333 launch after the frozen seed-2222 run

- [x] Launch seed 3333 only through `scratch/launch_pair_source_ablation_all15_seed3333_strict_gate_v4_20260714.ps1`, which must pass all three read-only checks immediately before invoking the existing wrapper:

  1. `audit_completed_pair_ablation_run_v4_20260712.py` reports `PASS`, exactly `225` result folds, and process exit code `0` for seed 2222.
  2. `verify_pair_ablation_sequence_v4_20260712.ps1` reports `PAIR_ABLATION_V4_SEQUENCE_OK`.
  3. `verify_pair_ablation_comparison_identity_v4_20260712.ps1` still reports comparison SHA `61825e5066546ea03deb9fb07ab2fe80487c34d7035611ac3e1dc64f5318eb30`.

  The current generic wrapper verifies the terminal status, exact result grid, required artifact existence, and comparison identity, but it does not invoke the full completed-run audit or sequence verifier. The additive strict-gate launcher fails closed, writes per-step logs plus `gate_status.json`, and enforces the stronger checks without changing any frozen runner source.

## Find a B³-safe additive linker-pair dose

- [ ] Determine the largest number of linker-derived pairs that can be added to the production pairwise-training recipe without materially degrading gold S2AND performance.

  - Once the current study's terminal audit confirms the result, freeze `uniform_100k` as the no-linker production base. Keep every one of its pairs fixed and add linker pairs on top; do not replace S2AND or pairwise-only pairs.
  - Implement this sweep with a separate additive assembler; changing the existing exact-budget linker cap would replace base rows and would not test this question.
  - For every seed and held-out dataset, require the additive arm's recorded base-pair digest to equal the corresponding frozen production-base result, then prove `final rows = unchanged base rows + deduplicated linker rows`.
  - Use the corrected block-local linker labels and balance positives/negatives within each source.
  - Run an escalating dose sweep per linker source: `0`, `2,500`, `5,000`, and `10,000` pairs. Test `25,000` and `50,000` only if `10,000` passes the safety gates.
  - After the first failing dose, refine the bracket between the largest passing and smallest failing doses. Stop when adjacent tested caps differ by at most `500` pairs per source, and retain the largest passing dose.
  - Compare two source sets:
    1. all 13 linker sources;
    2. only the seven non-S2AND name blocks (`a_khan`, `a_silva`, `h_wang`, `j_smith`, `s_gupta`, `s_lee`, `s_park`).
  - Use leave-one-dataset-out training. When evaluating a linker block, exclude that block's pairs completely so the proxy metrics measure cross-block transfer.
  - Repeat with training seeds `1111`, `2222`, and `3333`; keep evaluation identities and model hyperparameters fixed.
  - Report AUROC/AUPRC for all 15 held-out datasets and B³ only for the seven cluster-gold datasets. Keep name-block proxy metrics out of release ranking.
  - Accept the largest dose satisfying all B³ safety gates versus the no-linker production base:
    - mean paired B³ F1 delta `>= 0`;
    - paired lower-tail (`q05`) B³ F1 delta `>= -0.002`;
    - worst held-out-domain B³ F1 delta `>= -0.010`.
  - If no positive dose passes all three gates, retain zero linker pairs in the production recipe. Report both the nominal per-source cap and the realized total linker-pair count/share for the largest passing dose.
  - Record nominal caps, realized pair counts by source/family, total linker share, held-out exclusions, and paired domain-level deltas.
  - Run a tiny smoke test first, then the full job sequentially with Rust, `n_jobs=20`, and a 200 GiB RAM cap. Do not alter or overlap the currently frozen 675-fold study.

Current three-seed fold evidence to anchor the sweep: a `10,000`-per-source replacement cap realized up to 112,958 linker rows (16.1% of a 700k fold). It improved cross-block proxy AUPRC by `0.01416` across the seven linker blocks, but reduced mean gold S2AND B³ by `0.00924` (q05 `-0.01900`; worst domain, ZBMath, `-0.04033`), so it failed the release safety gates. The initial additive production candidate is therefore `2,500` pairs per source, pending this experiment.

Increasing the replacement cap to `50,000` per source does not repair the problem: across three seeds, the linker-only arm reduced mean B³ by `0.00914`, with q05 `-0.01943` and worst held-out domain (ZBMath) `-0.04167`. It did strengthen genuine cross-block linker transfer to pooled proxy AUPRC `+0.01836` and AUROC `+0.02713`. Linker dose therefore buys proxy performance but not S2AND safety. Keep the additive sweep staged and do not run its `25,000`/`50,000` doses unless the additive `10,000` arm first passes every safety gate.

The matched proxy-negative-only control is not a viable recipe: across three seeds it reduced pooled cross-block proxy AUPRC by `0.05412` and AUROC by `0.08897`. It also reduced mean gold S2AND B³ by `0.02533` (q05 `-0.04981`; worst held-out domain, INSPIRE, `-0.09333`). Preserve balanced positive/negative linker supervision in every additive-dose arm; do not use proxy negatives alone.
