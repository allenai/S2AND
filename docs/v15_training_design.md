# S2AND 1.5 training runbook

## Freeze block roles

Freeze one role per canonical block in `assignments.json` and pass it unchanged to every stage.
Score or sample an identity only when all assigned blocks holding its evidence share a role.
Keep ineligible mentions as clustering context. Only verified, score-eligible Gold ORCID labels supply truth.

| Role | Block targets | Use |
| --- | --- | --- |
| `fit_pool` | About 110 | Fit both pairwise heads, the 53-feature ranker, and the gate. Use the frozen validation fold for mined pairs and the marked subset for gate fitting. |
| `eps_cal` | About 40 | Tune the production clustering threshold (EPS) on blocks disjoint from `fit_pool`. Require a repeated scored ORCID per block. |
| `gate_thresh` | 10–15 plus a non-ORCID blind sample | Review the operating point on natural incremental requests, including retrieval misses. Use production seeds and blind judgments. |
| `dev_test` | About 40 fresh blocks plus pins, with at least 30 giants total | Iterate and check regressions. Pin the legacy 1.4 giants and the 27 Batch 1–6 blocks here. |
| `final_test` | About 200 fresh blocks, including about 60 giants | Evaluate the frozen candidate with verified holdout labels, the sealed blind panel, and the classic outer-test regression report. |

Draw development roles from Gold ORCID v2 development blocks across size and surname-origin strata.
Draw `final_test` from the holdout split. Exclude `f matsen`. Pin leakage-test blocks with `--fit-pool-blocks`.
The 28 recreation evaluation blocks have informed development. Reserve fresh blocks for final validation.

Export full Arrow requests for all candidate blocks and incremental requests with production `cluster_seeds`
for `gate_thresh`, from one snapshot. Save model and name-counts index digests. Use its Semantic Scholar
Academic Graph (S2AG) and abstract-sharing allowlists. For `gate_thresh`, export unique `cluster_block_key`, `corpus_paper_id`,
`position` rows with `inserted`, `updated`, and `clusterer` from Athena's `content_ext.paper_authors`.
Interpret timestamps in UTC and use the export window start as the inclusive cutoff.

~~~sh
uv run --offline --no-sync python -m scripts.intake.requests_to_datasets requests/full \
  --output-root snapshot --name-counts-index inference/name_counts_index \
  --allowed-papers s2ag-papers.json --abstract-allowed-papers shareable-abstracts.json
uv run --offline --no-sync python -m scripts.intake.build_identity_evidence \
  --snapshot snapshot --gold-parquet gold/labels.parquet --output-dir evidence
uv run --offline --no-sync python -m scripts.intake.build_freeze_input \
  --identity-evidence evidence/manifest.json --gold-blocks gold/blocks.parquet \
  --surname-origin-csv surname_origin.csv --output blocks.parquet
uv run --offline --no-sync python -m scripts.intake.assign_block_roles \
  --blocks blocks.parquet --identity-evidence evidence/manifest.json --targets-json targets.json \
  --extra-pins-json reviewed-blocks.json --giant-caps-json giant-caps.json \
  --pairwise-validation-fraction 0.15 --output-dir freeze
~~~

Intake filters non-S2AG papers, preserves restricted abstracts as `Has Abstract`, hashes structured
email, and scrubs plaintext before sampling. The release bundle writer scans for plaintext by default.
Supply reviewed signature vetoes with `--reviewed-errors-json`. Inspect per-block join and exclusion counts.
Evidence excludes blocks above 2% join errors or 10% missing export rows by default. Rejected joins veto affected
mentions. Same-paper ambiguities also veto mentions and are counted separately.
The surname CSV requires `surname` and `broad`. Set role targets in `targets.json` and role caps for blocks
above 15,000 signatures in `giant-caps.json`. Check freeze shortfalls and cross-role counts.

To append, reuse the index and allowlists, append evidence with `--append` and the same gold file, and rebuild
the freeze input. Pass the prior assignment with `--pinned-roles-json` to preserve roles and validation folds.
Rebuild hash-bound artifacts after a change to the master. Use `--fresh --pinned-roles-json` on the same
evidence and blocks to reproduce the master.
Rebuild evidence and freeze after an ORCID exclusion policy change.

Check sealed overlap against the deployed 1.4 linker source and its Gold ORCID training labels. Add known
training mentions with `--private-labels`. The audit requires comparable corpus positions and training ORCIDs.
Benchmark overlap is unobservable because benchmark IDs are dataset-local.

~~~sh
uv run --offline --no-sync python -m scripts.intake.build_v14_training_inventory \
  --gold-parquet gold/labels.parquet --linker-source-root v14-linker-source \
  --output inventory/v14_training.parquet
uv run --offline --no-sync python -m scripts.intake.check_v14_training_overlap \
  --identity-evidence evidence/manifest.json --block-role-assignments freeze/assignments.json \
  --training-parquet inventory/v14_training.parquet
~~~

## Run the fitting passes

Use fresh output directories and set `N` to the approved thread allocation. Feature contract 2 treats
1601 and 1800 as missing publication years. Rebuild features, linker windows, and EPS distances when their
package or distance-policy binding changes.

### 1. Seed and fit the pairwise heads

Hold out the queries and hide ORCID while clustering `fit_pool` with the 1.4 pairwise model.
Keep at least half the scored mentions as seeds. The miner writes an optional review report. Training
samples directly from the same frozen partitions, excludes query endpoints, fits only pairs with two
ledger scores, and requires mined validation pairs. Judge-only `merged_unlabeled` pairs have a separate budget.

~~~sh
uv run --offline --no-sync python -m scripts.linker.build_linker_inputs \
  --identity-evidence evidence/manifest.json --block-role-assignments freeze/assignments.json \
  --fit-pool-only --model-path inference/production_model_v1.4 \
  --output-dir mining-inputs --n-jobs N
uv run --offline --no-sync python -m scripts.pairwise.mine_pairwise_pairs \
  --identity-evidence evidence/manifest.json --block-role-assignments freeze/assignments.json \
  --block-inputs mining-inputs/inputs.json --pairs-per-block 2000 --verified-pair-fraction 0.05 \
  --pairs-per-block-floor 100 --output mined/pairs.csv
uv run --offline --no-sync python -m scripts.pairwise.train_pairwise \
  --model-plan model-plan.json --name-counts-index-root inference/name_counts_index \
  --block-inputs mining-inputs/inputs.json --identity-evidence evidence/manifest.json \
  --block-role-assignments freeze/assignments.json --matrix-work-dir matrix-work \
  --pairs-per-block 2000 --verified-pair-fraction 0.05 --pairs-per-block-floor 100 \
  --output-dir pairwise-model --n-jobs N --run-full
uv run --offline --no-sync python -m scripts.pairwise.release_pairwise calibrate-eps \
  --source-bundle pairwise-model --model-plan model-plan.json \
  --output-bundle calibrated-pairwise-1.5 --output-report benchmark-eps.json \
  --name-counts-index-root inference/name_counts_index --n-jobs N
~~~

### 2. Reseed, freeze tasks, and fit the linker

Recluster the same `fit_pool` seed population with the calibrated 1.5 pairwise model. Cluster oversized
blocks within native subblocks before the linker exists. For `gate_thresh`, use the latest eligible
arrivals and keep the production seed IDs for remaining mentions. Inspect request shortfalls and seed coverage.

~~~sh
uv run --offline --no-sync python -m scripts.linker.build_linker_inputs \
  --identity-evidence evidence/manifest.json --block-role-assignments freeze/assignments.json \
  --model-path calibrated-pairwise-1.5 --incremental-requests requests/incremental \
  --arrivals arrivals.parquet --since 2026-10-20 --requests-per-block 300 \
  --output-dir linker-inputs --n-jobs N
uv run --offline --no-sync python -m scripts.linker.build_linker_task_tables \
  --identity-evidence evidence/manifest.json --block-role-assignments freeze/assignments.json \
  --block-inputs linker-inputs/inputs.json --output-dir linker-tasks --n-jobs N
uv run --offline --no-sync python -m scripts.linker.train_linker_and_finalize \
  --linker-task-manifest linker-tasks/linker_tasks.json --block-role-assignments freeze/assignments.json \
  --target-json linker-target.json --pairwise-model-path calibrated-pairwise-1.5 \
  --name-counts-index-root inference/name_counts_index --output-dir finalized-1.5 --gate-cutoffs T_FULL T_INIT --n-jobs N
~~~

Freeze native top-25 windows with real names and native first tokens. Hold out whole authors before
retrieval for constructed fitting abstentions. Check purity across all gold-labeled seed members.
Keep all `fit_pool` tasks and input receipts. The trainer checks the seed model, count index, and seed
digests and writes `linker_training_report.json`. Ship the validated fit without pooling train and validation.

### 3. Review the gate operating point

Use natural `gate_thresh` windows for blind review and keep them outside fitting. Build packets with frozen
signature and candidate keys. Display only paper metadata and keep answer keys private. Embed 10–15% gold
controls. Use a separate output directory and `--seed` per judge family with a shared `--control-seed`.
Audit 10–20% of sealed test and classic benchmark packets across families. Require at least 10 controls,
both request classes, and 90% accuracy overall and within each request and verdict class.

~~~sh
uv run --offline --no-sync python -m scripts.judging.review_packets build \
  --packet-type linker --items-json gate-items.json --controls-json gate-controls.json \
  --identity-evidence evidence/manifest.json --block-role-assignments freeze/assignments.json \
  --allowed-role gate_thresh --linker-task-manifest linker-tasks/linker_tasks.json --output-dir packets
uv run --offline --no-sync python -m scripts.judging.review_packets score \
  --identity-evidence evidence/manifest.json --block-role-assignments freeze/assignments.json \
  --judge sol sol.jsonl sol_key.json --judge claude claude.jsonl claude_key.json --output-json gate-judgments.json
uv run --offline --no-sync python -m scripts.linker.blind_gate_metrics \
  --model-path finalized-1.5 --linker-task-manifest linker-tasks/linker_tasks.json \
  --block-role-assignments freeze/assignments.json --gate-thresh-judgments gate-judgments.json \
  --name-counts-index-root inference/name_counts_index --n-jobs N
~~~

Each verdict carries `item_id` and `display_sha256`, the canonical JSON digest of the full blind record.
For linker items, map candidate slots to `same`, `different`, or `insufficient_evidence` in `candidate_verdicts`.
Keep agreed candidate verdicts. Score an abstention only when every candidate has an agreed verdict.
An abstention is correct when all verdicts are `different`. Review judged-choice coverage, undecided share,
control accuracy, and agreement on jointly decided items.

Pass the reviewed full-name and initial-only ranker probability cutoffs with `--gate-cutoffs T_FULL T_INIT`.

### 4. Sweep production EPS and score the frozen candidate

EPS stays at 0.38, the value 1.4 shipped, until judged evidence moves it. Gold B-cubed (B3) cannot choose EPS:
the giant curve is flat, and it misses wrong merges among unlabeled mentions. The merges that EPS 0.40 adds over
0.38 failed blind judging, with at least 30% wrong.

Sweep all `eps_cal` blocks into one folder per dataset as a diagnostic. Hide ORCID and keep the full
clustering population. Apply 0.38 with the review's no-sweep form, which records a review note instead of a gold
F1 check: the sweep form's giant F1 tolerance can reject 0.38, because gold F1 favors looser cuts. The same call
stamps judged gate cutoffs onto the trained bundle with `--gate-cutoffs T_FULL T_INIT` and keeps the ranker bytes.

To move EPS, blind-judge the merges a candidate value adds or removes relative to 0.38. Pass only if at most
15% are wrong and the 95% upper bound is at most 25%. Reload the reviewed model and verify its EPS before
final testing.

~~~sh
uv run --offline --no-sync python -m scripts.eps.sweep_eps_on_linking_gold \
  --dataset a_ross --arrow-root snapshot --model-path finalized-1.5 \
  --identity-evidence evidence/manifest.json --block-role-assignments freeze/assignments.json \
  --allow-all-reference-identities --route production --batching-threshold 15000 \
  --eps-start 0.34 --eps-stop 0.42 --eps-step 0.02 --compute-missing-dists --allow-full-run \
  --output-dir eps/a_ross --n-jobs N
uv run --offline --no-sync python -m scripts.eps.apply_reviewed_eps \
  --source-model finalized-1.5 --output-model reviewed-model \
  --eps 0.38 --gate-cutoffs T_FULL T_INIT \
  --review-note "Cite the judged merge check and the judged cutoff packages." \
  --output-report eps-review.json
uv run --offline --no-sync python -m scripts.eps.orcid_b3 \
  --identity-evidence evidence/manifest.json --block-role-assignments freeze/assignments.json \
  --role final_test --block "BLOCK KEY" --arrow-dir "BLOCK ARROW DIRECTORY" --snapshot-root snapshot \
  --assignments-json predicted-clusters.json --eps 0.38
~~~

Pass every swept dataset once to the review. For sealed evaluation, score stored predictions over every
eligible signature and report B3 with and without singletons. See [release acceptance](release.md#stage-5-fit-evaluate-and-approve).

History is in the [plan-folder summary](../../s2and_v15_plan/MORNING_20261126.md).
Evidence is in the [1.4 recreation report](../../s2and_v15_plan/RECREATE_14.md).
