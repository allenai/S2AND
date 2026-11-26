# S2AND 1.5 Training, Split, and Labeling Design

This design addresses the data-split, leakage, distribution, and release-engineering issues surfaced during the S2AND `1.4` build (`2026-11-20` to `2026-11-23`).

## Core rules

1. **One block key = one lifecycle role.** Freeze a role for each canonical block key in the exported snapshot. Score or sample an identity only when every block holding its evidence has the same role. Keep cross-role mentions as context during evaluation clustering.
2. **Separate blocks only where a leakage test showed it matters.** Gold ORCID v2 provides 1.77M development blocks (~2,800 giants >10k) and 888k holdout blocks (1,443 giants). The 2026-11-24/25 leakage tests (pool vs cross-fit, giants, mid and small blocks) found:
   - no measurable effect from fitting the pairwise heads and the ranker on the same blocks;
   - a mean gate difference of −0.03 B3 F1 points [−0.21, +0.15] over five splits when fitting the gate on blocks outside the ranker's training pool;
   - a small, unproven upward drift in EPS chosen on mined blocks.

   The pairwise heads, ranker, and gate share `fit_pool`. Keep `eps_cal` disjoint from `fit_pool` and cluster each fitting block once per pairwise model.
3. **Separate linker-table construction from model release.** Freeze validated linker/gate task tables before starting a release fit. Do not mix data refresh, model fitting, and new giant-block task design into one release run.
4. **Ship what was validated.** Do not pool `train + val` and refit after selection; refit seed noise (±0.4 F1 pts on small benchmarks) shifts downstream distances seen by the linker.
5. **Redact and gate once at intake.** Hash email fields, strip plaintext emails from free text (`affiliations`, `title`, `abstract`), filter non-S2AG papers, and redact unshareable abstracts to `"Has Abstract"` (preserving `has_abstract` and redacting correction history) *before* Arrow conversion or pair sampling. The `write_release_bundles.py` CLI scans final bundles automatically by default (deleting output on failure).
6. **Freeze native top-25 windows before labeling.** Use ORCID-hidden reclustered seeds for fitting and the incremental export's production seeds for gate threshold evaluation.

## Development pools and sealed test halves

All legacy `1.4` giants (`a khan`, `a silva`, `h wang`, `j smith`, `s gupta`, `s lee`, `s park`, `s feldman`) and the 27 Batch 1–6 blocks are permanently assigned to `dev_test` (regression/iteration only). `f matsen` (1 scored identity) is excluded.

| Pool | Sources & allocation | Fits / tunes | Key guarantees |
| --- | --- | --- | --- |
| `fit_pool` | ~110 `orcid-blocks-v2/dev` blocks across size × surname-origin strata, plus the 80% benchmark train split (reclaiming 1.4's ~9% `unused_internal_test`) and fresh augmentation | Main and nameless pairwise heads (benchmarks + mined pairs), the 53-feature ranker, and the gate on the marked subset | Cap mining across strata at the smaller of 2,000 and 5% of verified within-block pairs, with a floor of up to 100 verified pairs. Pairwise validation uses benchmark val and the frozen `pairwise_validation_blocks`. The stratified `gate_training_blocks` subset can overlap that fold. Ranker replays real `full_name` and `initial_only` queries with real `author_middle`, with `no_safe` via whole-author holdout before retrieval. |
| `eps_cal` | ~40 `orcid-blocks-v2/dev` blocks across the same strata | Production-route EPS, close-pair bonus, subblocking and post-linkage corrections | Disjoint from `fit_pool`. |
| `gate_thresh` | 10–15 disjoint `orcid-blocks-v2/dev` blocks at **natural production request frequencies** + a non-ORCID natural sample for blind review | Operating-point report and blind review of added and removed links | Uses natural requests, including retrieval misses, without constructed author holdouts. Keep the configured weight until explicitly changed. Budget about 850 blind-judged items per arm: a 5-point wrong-rate gap needs about 600 consensus verdicts, and about 70% of items reach consensus. |
| `dev_test` | ~40 disjoint `orcid-blocks-v2/dev` blocks + legacy giants + 27 Batch 1–6 blocks + fresh paired Sol disagreements | Model-vs-model iteration & go/no-go | Never fitted on. Include at least 30 giants (pins supply 21); giants carry 88–98% of the paired variance. |
| `final_test_a` and `final_test_b` *(sealed)* | About 100 fresh `orcid-blocks-v2/holdout` blocks per half, including about 30 giants (±0.3 F1 paired intervals; ±0.15 would need 80–110 giants per half) + a sealed blind panel + canonical classic outer test (regression report) | Evaluate the frozen candidate on A and keep B sealed for a subsequent candidate | Score verified `holdout` gold labels through the identity ledger. |

**Tooling status.** Freeze one master `assignments.json` and pass it unchanged to every stage. The miner and ranker use `fit_pool`. The gate uses only marked `fit_pool` blocks, and `train_pairwise.py` uses the frozen block fold to split mined pairs. Reassign blocks before using a manifest with the retired gate role.

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

## Stage 0. Prepare evidence and freeze roles

### 1. Export production blocks

Export fresh full Arrow request bundles for every candidate block. Also export each gate threshold
block in incremental mode with its production `cluster_seeds` table. Keep both exports from the same
snapshot. Save the deployed model and name-counts index digests. Obtain paper allowlists for public
S2AG membership and abstract sharing from the export's permission snapshot.

### 2. Supply the Athena arrival join

For the exported `gate_thresh` block keys, query `content_ext.paper_authors` for
`cluster_block_key`, `corpus_paper_id`, `position`, `inserted`, `updated`, and `clusterer`. Supply one
Parquet table with unique block, paper, and position keys. Interpret timestamps in UTC. Set the
inclusive cutoff to the start of the export window. Keep `clusterer` for diagnostics.

### 3. Gate and index the exports

~~~sh
uv run --offline --no-sync python -m scripts.intake.requests_to_datasets requests/full \
  --output-root snapshot --name-counts-index inference/name_counts_index \
  --allowed-papers s2ag-papers.json --abstract-allowed-papers shareable-abstracts.json
~~~

Supply a directory or explicit list of full bundles. The builder checks their names against
`author_block`, removes papers outside the S2AG allowlist, preserves abstract presence with
`Has Abstract`, and hashes or scrubs emails before writing indexes. It scans retained and new
datasets for plaintext emails. Scrubbing and scanning cover Unicode domains with combining marks.
Re-run with the same index and allowlists that contain the prior
permissions to append new blocks. Each block retains the permission digests it had at intake.
Papers absent from the supplied lists stay restricted. A failed append preserves prior permissions.
Free-text scrubbing covers addresses followed immediately by long runs of letters.

### 4. Build the identity evidence table

~~~sh
uv run --offline --no-sync python -m scripts.intake.build_identity_evidence \
  --snapshot snapshot --gold-parquet gold/labels.parquet --output-dir evidence
~~~

Add `--reviewed-errors-json` with a map from block keys to reviewed signature IDs when available.
The table includes every exported mention, verified gold rows with the original split and scoring
flag, deposited ORCIDs, conflicts, and exclusions. Any rejected gold join vetoes the affected mention
for fitting, mining, and scoring. Position shifts are detected from the same block and paper when the
gold position is missing. A gold row ambiguous within its paper and block vetoes each mention there.
Only verified gold supplies a score. Freeze identity lists and seed purity
use matched gold and deposited identities. Inspect the per-block join audit.
The builder caps join errors at 2% per block with `--max-unmatched-gold` and missing export rows at 10%
with `--max-absent-from-export`. It excludes blocks that exceed either cap and reports exclusion counts
by reason. The audit records missing export papers, shifted positions, other failed joins, and papers
removed by that block's intake permissions.
Gold rows on a paper with two mentions in the block stay vetoed but do not count toward the limit,
so blocks with many same-surname coauthors (common in Vietnamese and South Asian names) are kept.
Only papers actually removed at intake are excluded from the denominator. An incomplete permission
list can leave a small, unrepresentative sample. Review the reason counts and the permission source
before using `--max-unmatched-gold` for a specific replay.
Older snapshots need fresh intake receipts to recompute coverage. To append, use `--append` on the same evidence
directory with the same gold file. Each appended block requires at least one gold match. The manifest
records each new partition and leaves the earlier partitions intact.

### 5. Build the freeze input

~~~sh
uv run --offline --no-sync python -m scripts.intake.build_freeze_input \
  --identity-evidence evidence/manifest.json --gold-blocks gold/blocks.parquet \
  --surname-origin-csv surname_origin.csv --output blocks.parquet
~~~

Strata combine the existing Arrow size bands (at most 1,000, 1,001 to 15,000, and above 15,000)
and the `broad` origin of the block surname. The required CSV lists `surname` and `broad`. Missing
surnames use `other`. The command reports matched and missing block and surname counts.

### 6. Freeze exported block roles

~~~sh
uv run --offline --no-sync python -m scripts.intake.assign_block_roles \
  --blocks blocks.parquet --identity-evidence evidence/manifest.json --targets-json targets.json \
  --extra-pins-json reviewed-blocks.json --giant-caps-json giant-caps.json \
  --pairwise-validation-fraction 0.15 --gate-training-block-count 20 --output-dir freeze
~~~

Use an aggregate `final_test` target or distinct half targets. Giant caps map role names to counts
above 15,000 signatures and include pinned blocks. Pass quoted leakage-test block keys to
`--fit-pool-blocks`. Keep legacy and reviewed `dev_test` keys out of that list. Inspect target and
stratum shortfalls and cross-role identity and mention counts.
The gate subset defaults to 20 blocks, capped by the available `fit_pool` blocks.
For an append, build a new freeze input from the expanded evidence manifest and pass the previous
`assignments.json` with `--pinned-roles-json`. Prior roles and pairwise fold membership are pinned.
Existing gate marks are kept on append unless you pass a new `--gate-training-block-count`.
Use `--fresh --pinned-roles-json freeze/assignments.json` without allocation options to reproduce
the master on the same blocks. After an append, rebuild hash-bound artifacts with the new master.
Judging keys retain their per-block evidence, eligibility, and seed checks.
The freeze records the ORCID exclusion policy digest. Rebuild evidence and freeze again after it changes.
An identity with evidence in another assigned role is ineligible for scoring and sampling. Adding
unassigned blocks preserves existing bindings. Evaluation clustering keeps ineligible mentions as context.

### 7. Check sealed overlap with 1.4 training

~~~sh
uv run --offline --no-sync python -m scripts.intake.build_v14_training_inventory \
  --gold-parquet gold/labels.parquet --linker-source-root v14-linker-source \
  --output inventory/v14_training.parquet
uv run --offline --no-sync python -m scripts.intake.check_v14_training_overlap \
  --identity-evidence evidence/manifest.json --block-role-assignments freeze/assignments.json \
  --blocks-json sealed-blocks.json --training-parquet inventory/v14_training.parquet
~~~

Use the source bundle for the deployed 1.4 linker. The inventory includes corpus-keyed training
and gate calibration queries, their full candidate component memberships, and giant gold ORCIDs.
Add further identified training mentions with `--private-labels`. Map both sealed halves' blocks to their Arrow directories.
The audit checks frozen signature counts and compares corpus paper and author positions and ORCIDs.
It fails when zero corpus keys can be compared. Benchmark overlap is unobservable because benchmark
IDs are dataset-local and the benchmarks lack ORCIDs.

### 8. Build linker inputs and gate requests

Run the 1.5 training passes in this order.

1. Cluster `fit_pool` without its queries using the 1.4 pairwise model.
2. Mine pairs from that partition and fit the 1.5 pairwise heads.
3. Recluster the same `fit_pool` seed population with the calibrated 1.5 pairwise bundle. Write to a fresh directory.
4. Build tasks from that second pass and fit the ranker and gate. The trainer requires the seeds' model digest to match its pairwise bundle.
5. Sweep EPS on `eps_cal` using the finalized 1.5 bundle.

~~~sh
uv run --offline --no-sync python -m scripts.linker.build_linker_inputs \
  --identity-evidence evidence/manifest.json --block-role-assignments freeze/assignments.json \
  --blocks-json linker-blocks.json --model-path inference/production_model_v1.4 \
  --arrivals arrivals.parquet --since 2026-10-20 --requests-per-block 300 \
  --output-dir linker-inputs --n-jobs N
~~~

Map each block to its `arrow_dir`, with an additional `incremental_request` path for gate threshold
blocks. Take the latest eligible mentions by `inserted` in each gate threshold block. Review any
shortfall against K, missing timestamps, and the actual timestamp range. Arrival paper keys must be
nonnegative integers or canonical decimal strings. Invalid arrival keys and empty request sets fail.
Keep production cluster IDs for the remaining seeded mentions. The manifest counts remaining mentions without production seeds.
For `fit_pool`, sample eligible scored mentions using `--queries-per-block`. The
ORCID-hidden `fit_pool` population is clustered without the queries, and the same partition supplies
assignments for mining and seeds for retrieval. Mining skips query endpoints and counts excluded
query signatures per block. A `fit_pool` block with no eligible signatures writes empty assignments,
seeds, and queries with reason `no_eligible_signatures`. The gate reuses the linker inputs from
the marked blocks. Re-run with the same inputs to verify and reuse completed blocks.

### 9. Freeze linker and gate tasks

~~~sh
uv run --offline --no-sync python -m scripts.linker.build_linker_task_tables \
  --identity-evidence evidence/manifest.json --block-role-assignments freeze/assignments.json \
  --block-inputs linker-inputs/inputs.json --output-dir linker-tasks --n-jobs N
~~~

The builder preserves real names and native first tokens for gate features and freezes native top-25 windows. The `fit_pool` tasks hold out whole
authors before retrieval. Ledger-backed fitting tasks require scored queries and seed assignments
for every eligible non-query signature. Gold-labeled roles check every seed member for identity purity.
The gate threshold windows record mixed production candidates in `candidate_gold_mixed`.
Agreed blind verdicts supply the gate threshold labels, including for mixed candidates. Score the fitted bundle
with `python -m scripts.linker.blind_gate_metrics`. Pass its `--model-path`, `--linker-task-manifest`,
`--block-role-assignments`, `--gate-thresh-judgments`, `--name-counts-index-root`, and `--n-jobs`.
The report checks the judgment bindings and gives judged-choice coverage across the requested queries.
Run `scripts.linker.train_linker_and_finalize` with the frozen task manifest, role assignments, linker target,
name-counts index, and pairwise bundle. The frozen fit writes `linker_training_report.json`.
Use `--false-abstain-weight` or `--initial-only-offset` to pass a reviewed gate setting explicitly.
Use `--blocks` and a fresh output directory to inspect a subset. Training requires every frozen
`fit_pool` block. Keep the linker input
receipts available so the trainer can check the input model, name-counts index, and seed digests.
Rebuild older ledger-backed task tables to include the receipt paths.
The gate selects logistic `C` and feature count `k` by four-fold cross-validation grouped by block, minimizing binary safe-link log loss across at least four marked blocks with scored queries.
The gate keeps `query_view_initial_only` and adds `classic.promoted_stratified_gate.initial_only_offset` (default `0.0`) to its safe-class weight after folding the scaler into the weights.
The offset sets the initial-only operating point in subblocked giants, where 99.6% of runtime gate calls are initial-only.
Blind judging passed +2.5 on the 51-block CV gate, a 52% runtime initial-only link rate per call.
The logit value does not transfer across refits: a 16-block gate already links 42% without an offset.
For each refit, pick the offset on `eps_cal` giants that reaches at most that judged link rate.
Confirm it with the sized `gate_thresh` blind review before release.

### 10. Mine pairs and train the pairwise heads

~~~sh
uv run --offline --no-sync python -m scripts.pairwise.mine_pairwise_pairs \
  --identity-evidence evidence/manifest.json --block-role-assignments freeze/assignments.json \
  --blocks-json miner-blocks.json --pairs-per-block 2000 --verified-pair-fraction 0.05 \
  --pairs-per-block-floor 100 --output mined/pairs.csv
uv run --offline --no-sync python -m scripts.pairwise.train_pairwise \
  --model-plan model-plan.json --name-counts-index-root name-counts-index \
  --mined-pairs mined/pairs.csv --identity-evidence evidence/manifest.json \
  --block-role-assignments freeze/assignments.json --matrix-work-dir matrix-work \
  --output-dir pairwise-model --n-jobs N --run-full
~~~

Map every block in the nonempty frozen `fit_pool` to its `arrow_dir` and `assignments` entry from step 8. The mining
manifest records each block's frozen fold, assignment path, and binding, including blocks with zero
sampled pairs. Keep the frozen assignment files and receipts available for training. Assignment
receipts and the trainer's name-counts index must match the intake index. All assignment receipts
must name one prediction model. The trainer verifies the ledger against the master assignment.

The fitted strata are `false_merge`, `missed_link`, and `random`. The trainer requires two ledger
scores for each fitted pair and uses them to compute the label. `merged_unlabeled` pairs require
at least one unscored endpoint and go to judging only. Each block uses its own random sequence
derived from the run seed and block key. Judge-only pairs have a separate `--pairs-per-block` budget.
The trainer replays selection from the ledger, frozen assignments, and recorded policy. It checks
pair IDs, strata, inclusion probabilities, and the report counts against the replay. Training
requires at least one fitted mined pair in the frozen validation fold. Set
`--min-mined-validation-pairs` to require more. The report
counts drops by reason from the eligible population through the frame and sample. Re-mine older
samples before training.

### 11. Sweep and apply production EPS

~~~sh
uv run --offline --no-sync python -m scripts.eps.sweep_eps_on_linking_gold \
  --dataset a_ross --arrow-root snapshot --model-path finalized-1.5 \
  --identity-evidence evidence/manifest.json --block-role-assignments freeze/assignments.json \
  --allow-all-reference-identities --route production --batching-threshold 15000 \
  --eps-start 0.34 --eps-stop 0.42 --eps-step 0.02 --compute-missing-dists --allow-full-run \
  --output-dir eps/a_ross --n-jobs N
uv run --offline --no-sync python -m scripts.eps.apply_reviewed_eps \
  --source-model finalized-1.5 --output-model reviewed-model \
  --sweep-root eps --datasets a_ross --expected-batching-threshold 15000 \
  --identity-evidence evidence/manifest.json --block-role-assignments freeze/assignments.json \
  --eps 0.38 --precision-f1-tolerance 0.001 --review-note "Record the review decision." \
  --output-report eps-review.json
~~~

Sweep all `eps_cal` blocks, with one folder per snapshot dataset. Pass each dataset once to the EPS
review. Production clustering hides ORCID
and keeps the full population. B3 scores verified, eligible mentions only. Apply verifies both
digests and the reference coverage. Reload the revised model and verify the chosen EPS.
## First 1.5 experiment order

1. **Experiment 1:** Keep 1.4's pairwise heads and 53-feature candidate ranker fixed for the gate-only experiment. Replay M1 gate calibration with the 1.4 ranker. For a contract-2 ranker, refit the gate on the marked `fit_pool` blocks and inspect the `gate_thresh` report for link count, blind-judged precision and recall within the frozen windows. Blind-judge the links each weight adds and removes. The report includes the configured weight and keeps it in production unless `classic.promoted_stratified_gate.false_abstain_weight` explicitly sets a new one. Measure cross-subblock recovery on unseen >15k blocks and non-ORCID precision. Across 14 blocks above 15k in Batches 1–6, 1.4 changed pooled F1 by −0.145 points [−0.609, +0.311] versus 1.3.2. The across-sub-block component was −0.456 points.
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

Provide signature references for pair sides, linker queries, and exemplars. The builder reads their
display fields from the ledger's frozen Arrow snapshot, or from `--snapshot` for builds without a ledger.
Supply full `member_signature_ids` for any linker component resolved outside the frozen gate seed assignments.
The builder rejects a linker query that belongs to any candidate component.
- **Strict field allowlist:** Paper metadata only for pair packets (never predicted-cluster summaries, scores, ranks, split names, cluster IDs, or ORCIDs).
- **Isolated passes across judge families** with independently randomized A/B or candidate order. Report agreement and Cohen's κ on jointly decided items, the jointly decided share, and missing verdicts.
- **10–15% embedded ORCID gold controls** per packet (re-examine when the family fails the 90% control bar) + **10–20% cross-family model audit** on sealed test packets and classic benchmarks.
- **Partial gate judgments:** Keep decided candidate outcomes. Score a query in abstention metrics
  only when every candidate has an agreed verdict. An abstention is correct when all verdicts are
  `different`.
- **Automatic ORCID selection-bias logging:** Every gate/EPS evaluation logs linked Gold-ORCID share and linked median publication year alongside B3 metrics.

Build controls with `review_packets.py build --identity-evidence evidence/manifest.json --block-role-assignments assignments.json --allowed-role gate_thresh`
for a gate threshold packet. The allowed roles apply to regular items and controls. Include the scoring
ORCIDs and any source `scored` flags on the private paper inputs. Controls use the identity ledger.
The builder derives pair and linker control truth from the verified ORCIDs and checks any supplied truth.
Natural gate threshold candidates can receive blind labels on mixed production clusters. A member
with an excluded or foreign-role identity vetoes a candidate. Gate bindings cover every identity
named by the gate evidence, including rejected gold and reviewed errors. Pass `--linker-task-manifest tasks/linker_tasks.json`
to use the frozen seed assignments and the bound Arrow papers. Supply query signature IDs and
candidate keys for gate items. The builder looks up all displayed query metadata in the bound Arrow dataset.
The builder selects up to four exemplars per component in frozen seed order. For controls, it
prefers members with matched gold or a deposited ORCID agreeing with gold. Each displayed control
component needs at least one such exemplar and must pass the full-component eligibility check.
The trainer resolves candidate membership from the same seeds
before applying judgments. Rebuild old gate packets and keys to record the frozen seed digest.
It balances same and different pair controls, or linkable and unlinkable linker requests, within available counts.
Pass `--balance-controls-by-score-band` and set private `metadata.score_band_v14` on each eligible control
to draw matched classes within supplied 1.4 score bands. Set `metadata.stratum` on every item.
Keep each answer key private.

Run `review_packets.py score --identity-evidence evidence/manifest.json --block-role-assignments assignments.json --judge sol sol.jsonl sol_key.json --judge claude claude.jsonl claude_key.json`.
The report records ledger bindings for every scored block. Gate labels use their own bindings
for the gate blocks in the judging panel and assigned blocks holding identities named by those
candidates. The scorer checks each linker key independently and skips stale control-only blocks
when its natural blocks and any gate dependencies still match. Changes to those dependencies invalidate the judgments.
Each verdict row contains `item_id`, `display_sha256`, and a pair `verdict` or a linker
`candidate_verdicts` mapping from packet slot to verdict. Compute `display_sha256` when preparing each
judge's input by applying `s2and.incremental_linking.contracts.canonical_json_digest` to the full parsed
blind JSONL record shown to the judge. Rebuild packets and obtain fresh votes for keys and verdicts without
display hashes. Use `same`, `different`, or `insufficient_evidence`. List the two primary families first, followed
by any audit families. The report lists audit coverage and checks audit consensus on the candidates
assigned to an audit family. The scorer aligns independent packet orders
by private source and candidate IDs. A linker control request succeeds when every candidate verdict is correct.
The control bar requires at least 10 control requests, both request classes and at least 90% accuracy
overall and within each request and verdict class. The builder rejects a shortfall in balanced controls
when the requested control fraction is positive. The report includes overall, shard, and stratum accuracy
and directional error rates.
The scorer counts missing primary verdicts against the planned panel. It lists missing request and candidate keys for each
family, reports request coverage and counts zero-candidate requests. The unresolved share includes missing
verdicts and planned requests with zero candidates. Cohen’s κ uses jointly decided verdicts.

### Score by frozen role

Run `uv run --offline --no-sync python -m scripts.eps.orcid_b3` with `--identity-evidence`,
`--block-role-assignments`, `--role`, `--block`, `--arrow-dir`, `--snapshot-root`,
`--assignments-json`, and `--eps`. The scorer requires assignments for every eligible gold
signature and reports B3 with and without singletons.
