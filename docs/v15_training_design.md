# S2AND 1.5 Training, Split, and Labeling Design

This design addresses the data-split, leakage, distribution, and release-engineering issues surfaced during the S2AND `1.4` build (`2026-11-20` to `2026-11-23`).

## Core rules

1. **One block key = one lifecycle role.** Assign every canonical block key a single role via seeded SHA-256 hash (`scripts/production/data/assign_block_roles.py`), with multi-block author closure (`strict` union-find or `home_pool` mirroring `cross_holders.parquet` in Gold ORCID v2 to avoid giant-component percolation). A block key keeps the same role across classic benchmarks, Gold ORCID v2, production block exports, and LLM-adjudicated packets.
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

### Freeze roles and join gold

Treat 1601 and 1800 as missing publication years when mining, labeling, or scoring pairs.
Pairwise year distance is missing if either paper has a placeholder. Retrieval omits placeholders from
candidate dates and year filters. Linker and gate year features use the existing missing-value behavior.
The source Arrow year stays available for provenance and the year-gap split's historical placement rule.
Feature contract 2 applies this rule to pairwise and linker inference. Loaded contract 1 bundles, including
1.4, keep the original feature and retrieval behavior. Finalizing a pairwise bundle preserves its contract.
Rebuild frozen linker windows when the recorded distance policy or package version differs. The manual EPS
curve reader also requires the current distance policy.
Rebuild feature tables and EPS distance caches created before this policy. EPS records distance policy 14.
The Arrow schema, feature names, and model bundle formats are unchanged.

Pass `--blocks` a JSON list or Parquet table with `block_key`, `signature_count`, `orcids` and `split`.
Supply `size_band` and `script_region` (or separate `script` and `region`) to define strata.
An explicit `stratum` overrides those fields. Otherwise, size defaults to the ≤1k, 1k–15k and >15k bands.
Use `--enforce-source-splits` to require a development or holdout split on every row.
The allocator samples each eligible stratum proportionally using the seeded SHA-256 order, then distributes the
sample among roles. An aggregate `final_test` target produces two balanced halves. Explicit half targets also work.
Give each sealed half room for every eligible stratum. The freeze reserves stratum coverage for
each role before proportional allocation and reports missing strata in `stratum_shortfalls`.
Pass the prior `assignments.json` to `--pinned-roles-json` when re-freezing. This preserves prior
block roles, scored ORCID homes, and exclusions. Keep pinned blocks in the metadata. When supplying
`orcid_mention_counts`, provide a positive count for each listed eligible ORCID.
Review per-role stratum counts and shortfalls in `assignments.json`. The sealed halves draw in pairs
by stratum when the available blocks and targets permit. Free `dev_test` blocks come from development.
Supply every scoring ORCID in the block intake. A scoring ORCID missing from its frozen block raises an error.
Use `orcid_mention_counts` or one list entry per mention for count-based home selection. With unique lists
alone, each known ORCID contributes one mention per block. Evaluation roles take priority.
Pass the prior `assignments.json` as `--pinned-roles-json` to preserve its blocks and ORCID homes.

In `home_pool`, the ORCID lists must cover every valid deposited or verified gold identity that can appear in the
block. A shared ORCID scores only in its manifest home role. Keep other mentions as context in Arrow.
Pass `--block-role-assignments` to `prepare_identity_labels.py` before using joined labels for scoring or training.
The join keeps verified unscored and non-home identities with `scored=False`. The linker task builder requires
that column and uses ineligible identities for seed purity and whole-author exclusion.
Pass the same file to `sweep_eps_on_linking_gold.py --score-mode orcid-b3 --allow-all-reference-identities`
and `apply_reviewed_eps.py`. EPS verifies the manifest digest, the zero scored overlap, and the scored identities
in the saved reference. A nonzero raw overlap is expected when `home_pool` assigns a shared identity.

### Build linker tasks

Run `scripts/production/data/build_linker_task_tables.py --block-role-assignments assignments.json
--block-inputs inputs.json --output-dir linker_tasks --n-jobs N`. Key `inputs.json` by block with entries
`arrow_dir`, `seeds`, `labels`, and `queries`. Give each block a manifest-backed Arrow dataset,
a Parquet seed table with `signature_id` and `component_key` from ORCID-hidden pre-attachment clustering,
verified ORCID label Parquet from the label join, and a JSON list of unique held-out request signature IDs.
Use the real request sample for `gate_thresh`. Keep its unknown gold outcomes in the windows for link counts.

The builder preserves each query's stored first and middle names and uses native automatic query views
(`full` for full names and `initial_only` for real initial-only names). It selects holdout authors
deterministically in `linker_train` and `gate_fit` and excludes all components containing their known seed
signatures before native retrieval. It verifies contiguous top-25 ranks, or the full eligible set for smaller blocks.
Pass the resulting `linker_tasks.json` to `train_linker_and_finalize.py --linker-task-manifest` along with the
same `--block-role-assignments` file. The trainer checks table hashes and computes features on the frozen windows.
Native retrieval can return fewer than 25 candidates. Keep those windows and record their counts.
Use all known seed identities for whole-author exclusion and leave mixed or unresolved targets unscored.
It fits the ranker and gate on their respective tables and leaves the sealed block halves outside those inputs.
For the gate-only arm, pass `--fixed-ranker-bundle` and `--expected-ranker-sha256`. The trainer checks
the supplied hash against the source manifest and preserves the booster bytes. To replay M1's source
calibration rows, pass `--use-legacy-gate-calibration` with the fixed-ranker options and omit the frozen inputs.

## First 1.5 experiment order

1. **Experiment 1:** Keep 1.4's pairwise heads and 53-feature candidate ranker fixed for the gate-only experiment. Refit the 240-feature gate on `gate_fit` and inspect the `gate_thresh` report for link count, gold precision on scored links, and gold recall against known safe seed components before retrieval. Blind-judge the links each weight adds and removes. The report includes the configured weight and keeps it in production unless `classic.promoted_stratified_gate.false_abstain_weight` explicitly sets a new one. Measure cross-subblock recovery on unseen >15k blocks and non-ORCID precision. Across 14 blocks above 15k in Batches 1–6, 1.4 changed pooled F1 by −0.145 points [−0.609, +0.311] versus 1.3.2. The across-sub-block component was −0.456 points.
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

### Non-negotiable packet rules (`scripts/production/data/review_packets.py`)
- **Strict field allowlist:** Paper metadata only for pair packets (never predicted-cluster summaries, scores, ranks, split names, cluster IDs, or ORCIDs).
- **Isolated passes across judge families** with independently randomized A/B or candidate order. Report agreement and Cohen's κ on jointly decided items, the jointly decided share, and missing verdicts.
- **10–15% embedded ORCID gold controls** per packet (drop/re-examine below 90% control accuracy) + **10–20% cross-family model audit** on sealed test packets and classic benchmarks.
- **No "any abstain drops query" rule:** Never discard an entire 25-candidate query when a secondary candidate receives `abstain`.
- **Automatic ORCID selection-bias logging:** Every gate/EPS evaluation logs linked Gold-ORCID share and linked median publication year alongside B3 metrics.

Build controls with `review_packets.py build --block-role-assignments assignments.json --allowed-role gate_thresh`
for a gate threshold packet. The allowed roles apply to regular items and controls. Include the scoring
ORCIDs and any source `scored` flags on the private paper inputs. Controls require home-role evidence.
Set `metadata.stratum` in every item and set control truth with `metadata.gold_label` (0/1) for pairs
or `metadata.gold_candidate_keys` for linker packets. Keep each answer key private.

Run `review_packets.py score --judge sol sol.jsonl sol_key.json --judge claude claude.jsonl claude_key.json`.
Each verdict row contains `item_id` and a pair `verdict` or a linker `candidate_verdicts` mapping from packet
slot to verdict. Use `same`, `different`, or `insufficient_evidence`. The scorer aligns independent packet orders
by private source and candidate IDs. A linker control request succeeds when every candidate verdict is correct.
Report request accuracy, same and different control accuracy, and directional error rates by shard and stratum.
Report paired coverage, agreement, and Cohen’s κ by task and stratum on natural items. Kappa uses jointly
decided verdicts. The agreement measure that includes abstentions counts decided versus abstain as disagreement.



### Score and sample by frozen role

Run `uv run python -m scripts.eps_sweep.orcid_b3` with `--block-role-assignments`, `--role`, `--block`,
`--arrow-dir`, `--snapshot-root`, `--assignments-json` and `--eps`. Pass verified labels with
`--external-labels` when available. The report gives reference coverage and B3 with and without singletons.

Run `uv run python -m scripts.production.data.mine_pairwise_pairs` with `--block-role-assignments`,
`--blocks-json`, `--pairs-per-block` and `--output`. Map block keys to Arrow directories in the input JSON.
The miner samples uniformly within `pairwise_train` blocks and writes signature ID pairs and inclusion probabilities.
