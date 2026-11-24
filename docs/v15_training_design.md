# S2AND 1.5 Training, Split, and Labeling Design

This design addresses the data-split, leakage, distribution, and release-engineering issues surfaced during the S2AND `1.4` build (`2026-11-20` to `2026-11-23`).

## Core rules

1. **One block key = one lifecycle role.** Assign every canonical block key a single role via seeded SHA-256 hash (`scripts/intake/assign_block_roles.py`), with multi-block author closure (`strict` union-find or `home_pool` mirroring `cross_holders.parquet` in Gold ORCID v2 to avoid giant-component percolation). A block key keeps the same role across classic benchmarks, Gold ORCID v2, production block exports, and LLM-adjudicated packets.
2. **Partition `orcid-blocks-v2` by block instead of cross-fitting the full pipeline.** Gold ORCID v2 provides 1.77M development blocks (~2,800 giants >10k) and 888k holdout blocks (1,443 giants). Block-disjoint pools keep every downstream stage strictly out-of-sample for upstream models without nested cross-fitting.
3. **Separate linker-table construction from model release.** Freeze validated linker/gate task tables before starting a release fit. Do not mix data refresh, model fitting, and new giant-block task design into one release run.
4. **Ship what was validated.** Do not pool `train + val` and refit after selection; refit seed noise (±0.4 F1 pts on small benchmarks) shifts downstream distances seen by the linker.
5. **Redact and gate once at intake.** Hash email fields, strip plaintext emails from free text (`affiliations`, `title`, `abstract`), filter non-S2AG papers, and redact unshareable abstracts to `"Has Abstract"` (preserving `has_abstract` and redacting correction history) *before* Arrow conversion or pair sampling. The `write_release_bundles.py` CLI scans final bundles automatically by default (deleting output on failure).
6. **Build linker candidates from batch-clustered profiles, after pairwise fit, and freeze native top-25 windows before any labeling.** Never train on historical fragmented production clusters, and never label candidate lists built with a different retrieval planner than the trainer uses.

## Development pools and sealed test halves

All legacy `1.4` giants (`a khan`, `a silva`, `h wang`, `j smith`, `s gupta`, `s lee`, `s park`, `s feldman`) and the 27 Batch 1–6 blocks are permanently assigned to `dev_test` (regression/iteration only). `f matsen` (1 scored identity) is excluded.

| Pool | Sources & allocation | Fits / tunes | Key guarantees |
| --- | --- | --- | --- |
| `pairwise_train` | Single clean 80% split of 7 benchmarks (reclaiming the ~9% `unused_internal_test` discarded by 1.4's double 80/10/10 split) + fresh augmentation (+ optional gated `>45yr` historical or ORCID dev pairs) | Main (33-feat) & Nameless (17-feat) pairwise LightGBM heads | Zero block and scored ORCID overlap with downstream or test pools. Freeze heads before generating downstream scores. |
| `pairwise_val` | 10% of 7 benchmarks (macro-7 + INSPIRE-capped) + 20–30 stratified `orcid-blocks-v2/dev` blocks (small/med/large/giant, incl. non-English & missing-SPECTER2) | Pairwise HPO / early stopping / feature & missingness safety checks | Catches giant/non-English regressions at Stage 3; no longer the sole EPS selector. |
| `linker_train` | 40–60 diverse `orcid-blocks-v2/dev` blocks (Western, Iberian, Slavic, CJK, South-Asian) + optional OOF benchmark slice | Retrieval weights (replace `h_wang_any_input_v2`) & 53-feature LightGBM ranker | Replays real `full_name` and real `initial_only` queries (**preserving real `author_middle`**; retire middle-wiped `visibleq:` clones). `no_safe` via deterministic **whole-author holdout before retrieval**. |
| `gate_fit` | 15–25 disjoint `orcid-blocks-v2/dev` blocks (~15k queries) | Gate medians, `StandardScaler`, feature selection (`k`), `C`, and logistic weights (grouped block CV) | 100% out-of-sample pairwise & ranker scores; enriched to **25–40% whole-author-holdout `no_safe`** across all size bands. Never pooled with `gate_thresh`. |
| `gate_thresh` | 10–15 disjoint `orcid-blocks-v2/dev` blocks at **natural production request frequencies** + a non-ORCID natural sample for blind review | Operating-point report and blind review of added and removed links | Uses natural requests, including retrieval misses, without constructed author holdouts. Keep the configured weight until explicitly changed. |
| `eps_cal` | ~40 disjoint `orcid-blocks-v2/dev` blocks across size × script/region bands (plus benchmark `pairwise_val` as a secondary diagnostic) | Production-route EPS (step `0.02`), close-pair bonus, subblocking & post-linkage corrections | Score verified identities in their home role. Keep raw signatures in the clustering input. |
| `dev_test` | ~40 disjoint `orcid-blocks-v2/dev` blocks + legacy giants + 27 Batch 1–6 blocks + fresh paired Sol disagreements | Model-vs-model iteration & go/no-go | Never fitted on. |
| `final_test_a` and `final_test_b` *(sealed)* | 130–300 fresh `orcid-blocks-v2/holdout` blocks in total (incl. 40–60 fresh giants) + a sealed blind panel + canonical classic outer test (regression report) | Evaluate the frozen candidate on A and keep B sealed for a subsequent candidate | Require verified `(corpus_paper_id, position)` ORCID label joins with `prepare_identity_labels.py --expected-split holdout`. |

### Feature contracts

Treat 1601 and 1800 as missing publication years when mining, labeling, or scoring pairs.
Pairwise year distance is missing if either paper has a placeholder. Retrieval omits placeholders from
candidate dates and year filters. Linker and gate year features use the existing missing-value behavior.
The source Arrow year stays available for provenance and the year-gap split's historical placement rule.
Feature contract 2 applies this rule to pairwise and linker inference. Loaded contract 1 bundles, including
1.4, keep the original feature and retrieval behavior. Finalizing a pairwise bundle preserves its contract.
Rebuild frozen linker windows when the recorded distance policy or package version differs. The manual EPS
curve reader also requires the current distance policy.
Rebuild feature tables and EPS distance caches created before this policy. EPS records distance policy 15.
The Arrow schema, feature names, and model bundle formats are unchanged.

### Intake production requests

Convert a directory or an explicit list of full production request bundles. Use the name-counts index
from the production model's inference bundle.

```sh
uv run python -m scripts.intake.requests_to_datasets requests/full \
  --output-root snapshot --name-counts-index inference/name_counts_index
```

This writes `snapshot/manifest.json` and one indexed dataset under `snapshot/datasets/<block_key>/`.
The dataset name uses underscores. Each dataset records its source bundle SHA-256 in
`request_source.json`. Re-run the same command to reuse validated datasets.

### Freeze roles

Pass `--blocks` the chosen block set as JSON or Parquet. Include `block_key`, `signature_count`,
`orcids` and `split`. Use the real signature count from the full Arrow export for each sealed block.
Supply `size_band` and `script_region` (or separate `script` and `region`) to define strata.
An explicit `stratum` overrides those fields. Otherwise, size defaults to the ≤1k, 1k–15k and >15k bands.

```sh
uv run python -m scripts.intake.assign_block_roles \
  --blocks blocks.parquet --targets-json targets.json --output-dir freeze \
  --closure-mode home_pool --enforce-source-splits --extra-pins-json reviewed-blocks.json
```

An aggregate `final_test` target produces two halves. Give each half room for the eligible strata.
The allocator reserves coverage before proportional allocation. Only one half can cover a stratum with a
single available block. Review the printed `stratum_shortfalls` and the per-role counts in
`assignments.json`. Use `--dev-test-blocks` to pin examined blocks and `--exclude-blocks` to record
exclusions, including blocks absent from the intake.

Pass the prior `assignments.json` to `--pinned-roles-json` when re-freezing. With `home_pool`, pass
block-role maps to `--extra-pins-json` on both initial and subsequent freezes. Keep prior pinned blocks
in the metadata. Provide a positive `orcid_mention_counts` value
for every listed eligible ORCID, or supply one list entry per mention. Unique lists give each known ORCID
one mention per block. Include every deposited and verified Gold ORCID that can appear in the block.

A shared ORCID scores only in its frozen home role. If it occurs in `dev_test` and a sealed half, home it
in `dev_test`. Keep raw signatures as clustering context. A nonzero raw overlap is expected in
`home_pool`. The assignment manifest must certify zero cross-role scoring overlap.

### Join Gold ORCID labels

```sh
uv run python -m scripts.intake.prepare_identity_labels \
  --arrow-dataset-dir snapshot/datasets/a_ross --gold-parquet gold/labels.parquet \
  --block-role-assignments freeze/assignments.json --expected-split development \
  --output-parquet labels/a_ross.parquet --audit-json labels/a_ross.audit.json
```

Use `--expected-split holdout` for either sealed half. The join keeps verified non-home and source-unscored
identities as `scored=False`. It exits with status 1 if it verifies zero rows. Use the audit to check
missing positions, masked rows and Arrow signature counts.

### Check sealed overlap with 1.4 training

```sh
uv run python -m scripts.intake.build_v14_training_inventory \
  --gold-parquet gold/labels.parquet --output inventory/v14_training.parquet
uv run python -m scripts.intake.check_v14_training_overlap \
  --block-role-assignments freeze/assignments.json --blocks-json sealed-blocks.json \
  --training-parquet inventory/v14_training.parquet
```

Pass each additional identified ORCID-bearing 1.4 training table to the builder with `--private-labels`.
Map every sealed block to its Arrow directory in `sealed-blocks.json`. The checker validates the
real-signature count against the freeze and compares signature IDs, available `(paper_id, position)` keys,
and deposited and frozen ORCIDs. It reports distinct valid training ORCIDs and fails on overlap or zero
training ORCIDs. Use `--allow-no-training-orcids` only for an intentionally signature-only check.
The 1.4 benchmark mentions cannot be matched to production mentions because their IDs are dataset-local
and their signature tables carry no ORCIDs.

### Build linker inputs

```sh
uv run python -m scripts.linker.build_linker_inputs \
  --block-role-assignments freeze/assignments.json --blocks-json linker-blocks.json \
  --model-path inference/production_model_v1.4 --output-dir linker-inputs --n-jobs N
```

Map each block to `arrow_dir` and `labels` in `linker-blocks.json`. The command samples scored mentions
using `--seed` and `--queries-per-block`. Supply a `gate_thresh` block's `incremental_request` path to
take its real unseeded signatures as queries. Use a seeded sample when the incremental request is
unavailable and record that choice in the run. Use natural request frequencies for a release gate review.

The builder predicts the remaining signatures on an ORCID-hidden copy with batching threshold 15,000.
It records query lists and seed components in `linker-inputs/inputs.json`. Re-run with the same inputs to
reuse completed blocks. Include `pairwise_train` blocks in the input map to write full-block 1.4 predictions
as `linker-inputs/<block_key>/assignments.parquet` for mining.

### Build linker tasks

```sh
uv run python -m scripts.linker.build_linker_task_tables \
  --block-role-assignments freeze/assignments.json --block-inputs linker-inputs/inputs.json \
  --output-dir linker-tasks --n-jobs N
```

The default requires all frozen `linker_train`, `gate_fit` and `gate_thresh` blocks. Use
`--blocks` and a fresh output directory to build a subset. The builder preserves stored first and middle
names and uses native query views (`full` or `initial_only`). It selects whole-author holdouts in
`linker_train` and `gate_fit` and excludes every known component of a holdout author before retrieval.
It uses all verified identities for seed purity. Mixed components and unresolved outcomes stay unscored.
It checks contiguous ranks and retains native short windows, including unknown outcomes.

Pass the complete `linker_tasks.json` and the same `--block-role-assignments` to
`train_linker_and_finalize.py --linker-task-manifest`. The trainer verifies table hashes and computes
features on the frozen windows. For the gate-only arm, pass `--fixed-ranker-bundle` and
`--expected-ranker-sha256`. The trainer verifies the source booster and its binding to the requested
pairwise heads and feature contract. Use `--use-legacy-gate-calibration` with the 1.4 ranker.
It uses linker contract 1, while frozen linker windows use contract 2. The gate-only report records the
fixed booster's tree count and test metrics.

### Mine pairwise training pairs

```sh
uv run python -m scripts.pairwise.mine_pairwise_pairs \
  --block-role-assignments freeze/assignments.json --blocks-json miner-blocks.json \
  --pairs-per-block 2000 --pairs-per-identity 25 --output mined/pairs.csv
```

Map each `pairwise_train` block to `arrow_dir`, joined `labels` and 1.4 `assignments` Parquet.
Keep source-unscored identities in the label table for masking. The miner excludes mentions whose Gold or
deposited ORCID has a different home role. Its strata are `false_merge`, `missed_link`, `merged_unlabeled`
and `random`. The block and identity caps apply to each stratum.
The manifest records the raw eligible population, capped frame size, sample count, input digests and seed.
The reported inclusion probability is conditional on the capped frame. Use it for inference within that frame.

### Sweep production EPS

```sh
uv run python -m scripts.eps.sweep_eps_on_linking_gold \
  --dataset a_ross --arrow-root snapshot --model-path inference/production_model_v1.4 \
  --block-role-assignments freeze/assignments.json --reference-labels labels/a_ross.parquet \
  --allow-all-reference-identities --route production --batching-threshold 15000 \
  --eps-step 0.02 --compute-missing-dists --allow-full-run --output-dir eps/a_ross --n-jobs N
```

Sweep every `eps_cal` block. Name each sweep folder exactly after the dataset as declared in the snapshot
manifest. For dataset `a_ross`, use `eps/a_ross`. Production prediction hides ORCID. The curves score
repeated and singleton-inclusive references with the frozen home-role mask.

### Apply reviewed EPS

```sh
uv run python -m scripts.eps.apply_reviewed_eps \
  --source-model inference/production_model_v1.4 --output-model reviewed-model \
  --sweep-root eps --datasets a_ross --expected-batching-threshold 15000 \
  --block-role-assignments freeze/assignments.json --eps 0.38 --precision-f1-tolerance 0.01 \
  --review-note "Record the review decision here." --output-report eps-review.json
```

List the complete reviewed `eps_cal` pool. Apply checks that every block-disjoint sweep used the exact
reviewed role manifest and that each curve's counts match the nonempty reference. Check the report and
reload the revised model to verify the applied EPS.

## First 1.5 experiment order

1. **Experiment 1:** Keep 1.4's pairwise heads and 53-feature candidate ranker fixed for the gate-only experiment. Replay M1 gate calibration with the 1.4 ranker. For a contract-2 ranker, refit the gate on `gate_fit` and inspect the `gate_thresh` report for link count, gold precision on scored links, and gold recall against known safe seed components before retrieval. Blind-judge the links each weight adds and removes. The report includes the configured weight and keeps it in production unless `classic.promoted_stratified_gate.false_abstain_weight` explicitly sets a new one. Measure cross-subblock recovery on unseen >15k blocks and non-ORCID precision. Across 14 blocks above 15k in Batches 1–6, 1.4 changed pooled F1 by −0.145 points [−0.609, +0.311] versus 1.3.2. The across-sub-block component was −0.456 points.
2. **Experiment 2:** Evaluate compact asymmetric middle & career-year gate features (`query_mid_x_cand_no_mid_x_size`, `query_mid_overlap_frac`, `query_mid_conflict_any`, `year_predates_log`) and retrieval re-tuning only after Experiment 1 establishes the clean baseline.
3. **Experiment 3 (conditional):** Lift `YEAR_DIFF_CAP=50` only if backed by targeted `>45yr` historical labeled pairs (`≥5,000` pairs so splits exceed `min_child_samples`).

## Sol 6.2 blind judging protocol & phased budget

Use ORCID Gold v2 and classic benchmarks for bulk pairwise training, ranker training, and modern STEM recall. Reserve `gpt-6.2-sol` (Sol 6.2) for ORCID's structural blind spots: non-ORCID / pre-2010 records, non-English & regional name regimes, institutional homonyms (e.g., BARC), non-ORCID marginal linker accepts/abstains, constructed-negative contamination, and post-linkage rule precision.

### Core budget (~6,600 packets / ~13,200 dual-pass judgments)
- **Phase 0 — Cross-family pilot (200 packets):** Use at least two distinct LLM families as the agreement anchor. Measure token cost, decided share, control accuracy and false-same versus false-different skew before scaling.
- **Phase 1A — Gate prevalence, contamination & reliability (1,800 packets):** 600 natural production requests across 3 size bands (`gate_thresh`), 500 constructed-negative contamination checks, 500 non-ORCID reliability checks, 200 seed-purity / one-sided middle-initial checks.
- **Phase 1B — Sealed final release panel (3,000 packets):** 1,800 probability-weighted natural production requests (600 each across `≤1k`, `1k–15k`, `>15k`; about ±1.74 percentage points per 600 at 5% error under simple random sampling) + 6 challenge cohorts × 200 (pre-2010/no-ORCID, non-English/missing-vector, East-Asian, South-Asian/BARC, marginal `initial_only`/abstains, residual splits/missed merges).
- **Phase 1C — Per-release paired disagreements (1,600 packets):** Fresh `1.5 != 1.4` disagreements (`600` bulk diffs accounting for ~51% "can't tell" on raw paper pairs, `600` linker attach/abstain diffs, `200` correction diffs, `200` controls).
- **Phase 2 (conditional):** `5,000` `>45yr` historical pairwise training pairs (only if lifting `YEAR_DIFF_CAP=50`); `1,200–3,000` gate hard-negative packets (only if Phase 1A reliability flags >5 pp error after reweighting); `700–1,500` one-time cross-family audit of AMiner v6 merges & unknown signatures.

### Non-negotiable packet rules (`scripts/judging/review_packets.py`)
- **Strict field allowlist:** Paper metadata only for pair packets (never predicted-cluster summaries, scores, ranks, split names, cluster IDs, or ORCIDs).
- **Isolated passes across judge families** with independently randomized A/B or candidate order. Report agreement and Cohen's κ on jointly decided items, the jointly decided share, and missing verdicts.
- **10–15% embedded ORCID gold controls** per packet (re-examine when the family fails the 90% control bar) + **10–20% cross-family model audit** on sealed test packets and classic benchmarks.
- **No "any abstain drops query" rule:** Never discard an entire 25-candidate query when a secondary candidate receives `abstain`.
- **Automatic ORCID selection-bias logging:** Every gate/EPS evaluation logs linked Gold-ORCID share and linked median publication year alongside B3 metrics.

Build controls with `review_packets.py build --block-role-assignments assignments.json --allowed-role gate_thresh`
for a gate threshold packet. The allowed roles apply to regular items and controls. Include the scoring
ORCIDs and any source `scored` flags on the private paper inputs. Controls require home-role evidence.
The builder derives pair and linker control truth from the verified ORCIDs and checks any supplied truth.
It balances same and different pair controls, or linkable and unlinkable linker requests, within available counts.
Pass `--balance-controls-by-score-band` and set private `metadata.score_band_v14` on each eligible control
to draw matched classes within supplied 1.4 score bands. Set `metadata.stratum` on every item.
Keep each answer key private.

Run `review_packets.py score --judge sol sol.jsonl sol_key.json --judge claude claude.jsonl claude_key.json`.
Each verdict row contains `item_id` and a pair `verdict` or a linker `candidate_verdicts` mapping from packet
slot to verdict. Use `same`, `different`, or `insufficient_evidence`. The scorer aligns independent packet orders
by private source and candidate IDs. A linker control request succeeds when every candidate verdict is correct.
The control bar requires at least 10 control requests, both request classes and at least 90% accuracy
overall and within each request and verdict class. The builder rejects a shortfall in balanced controls
when the requested control fraction is positive. The report includes overall, shard, and stratum accuracy
and directional error rates.
The scorer compares families on shared natural keys. It lists missing request and candidate keys for each
family, reports request coverage and counts zero-candidate requests. The unresolved share includes missing
verdicts and jointly covered requests with zero candidates. Cohen’s κ uses jointly decided verdicts.

### Score by frozen role

Run `uv run python -m scripts.eps.orcid_b3` with `--block-role-assignments`, `--role`, `--block`,
`--arrow-dir`, `--snapshot-root`, `--assignments-json` and `--eps`. Pass verified labels with
`--external-labels` when available. The scorer requires assignments for every reference signature and
reports B3 with and without singletons. It rejects external scored labels whose signatures are absent.
