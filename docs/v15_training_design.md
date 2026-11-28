# S2AND 1.5 training runbook

## Freeze block roles

Freeze one role per canonical block in `assignments.json` and pass it unchanged to every stage.
Score or sample an identity only when all assigned blocks holding its evidence share a role.
Keep ineligible mentions as clustering context. Only verified, score-eligible Gold ORCID labels supply truth.

| Role | Block targets | Use |
| --- | --- | --- |
| `fit_pool` | About 110 | Fit both pairwise heads and the 53-feature ranker. Use the frozen validation fold for mined pairs. |
| `eps_cal` | About 40 | Check the production clustering threshold (EPS) on blocks disjoint from `fit_pool`. Require a repeated scored ORCID per block. |
| `gate_thresh` | 10–15 plus a non-ORCID blind sample | Review the operating point on natural incremental requests, including retrieval misses. Use production seeds and blind judgments. |
| `dev_test` | About 40 fresh blocks plus pins, with at least 30 giants total | Iterate and check regressions. Pin the legacy 1.4 giants and the 27 Batch 1–6 blocks here. |
| `final_test` | About 200 fresh blocks, including about 60 giants | Evaluate the frozen candidate with verified holdout labels, the sealed blind panel, and the classic outer-test regression report. |

Draw development roles from Gold ORCID v2 development blocks across size and surname-origin strata.
Draw `final_test` from the holdout split. Exclude `f matsen`. Pin leakage-test blocks with `--fit-pool-blocks`.
Reserve fresh blocks for final validation. The 28 recreation blocks are spent for tuning.

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

Recluster the same `fit_pool` seed population with the benchmark-calibrated pairwise model. Cluster oversized
blocks within native subblocks before the linker exists. For `gate_thresh`, use the latest eligible
arrivals and keep the production seed IDs for remaining mentions. Inspect request shortfalls and seed coverage.
Pass provisional values to the trainer's required `--gate-cutoffs`.

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
  --name-counts-index-root inference/name_counts_index --output-dir finalized-1.5 \
  --gate-cutoffs T_FULL T_INIT --n-jobs N
~~~

Freeze native top-25 windows with real names and native first tokens. Hold out whole authors before
retrieval for constructed fitting abstentions. Check purity across all gold-labeled seed members.
Keep all `fit_pool` tasks and input receipts. The trainer checks the seed model, count index, and seed
digests and writes `linker_training_report.json`. Ship the validated fit without pooling train and validation.

### 3. Review production EPS

Start at 0.38, the EPS shipped in 1.4. With new heads, keep 0.38 if the cluster count is within ±2% of
1.4 on the same blocks. Otherwise blind-judge the marginal merges. To move EPS, judge the merges the
new value adds or removes relative to 0.38. Require at most 15% wrong merges and a 95% Wilson upper
bound of at most 25%. Keep the full clustering population with ORCID hidden. Use gold EPS curves as
diagnostics. Merge review and pass bar (internal report `gate_shape_blind/PASS_BAR.md`)

### 4. Judge and set the gate cutoffs

Set `t_full` for full-name queries and `t_init` for initial-only queries after training. Rejudge these
probability cutoffs for each release ranker. With the 1.4 heads and the refit ranker, `t_init` is 0.792
and judged cost is flat from 0.74 to 0.89. `t_full` is 0.868, the isotonic 0.80 crossing on held-out
full-name requests. On 16 fresh blocks it beat shipped 1.4 by 5.15 cost per 100 full-name decisions
[2.26, 6.49]. Refit probabilities are overconfident on fresh full names. Use probability alone.
Full-name holdout (internal report `fullname_holdout/RESULT.md`) · Confirmation (internal report `fullname_confirm/RESULT.md`)
A top-two score margin failed independent confirmation. Cutoff result (internal report `gate_cutoff_blind/RESULT.md`) · Margin result (internal report `margin_confirm/RESULT.md`)

Use the cutoff pass bar (internal report `gate_cutoff_blind/PASS_BAR.md`) as the method. Freeze the
pass bar before building packages. Sample top candidates from final runtime decisions across probability
bands, including ordinary gate abstentions. Record inclusion probabilities and exclusion counts. After
the judge check passes, fit isotonic P(right) with inverse inclusion weights. Choose the lowest score
where the curve reaches 0.80. Give undecided items half credit and report both sensitivities from the pass bar.

1. Run `sweep_eps_on_linking_gold --capture-dir` with the production route and one `--eps-values` value.
   Replay real incremental requests at the reviewed EPS with the same [decision recorder](../scripts/judging/decision_capture.py).
2. Run `judged_bank build-package --captured`. Reserve 10–15% for planted same-person and different-person
   controls, including same-name different-person pairs. The bank matches query and block, then requires
   all previously shown signatures to remain and form a [strict majority of the evidence](judged_bank.md#build-a-package).
3. Give Fable and Neon the blind package in separate folders. Neon replaced Astra after Astra failed
   the planted-control check. Freeze both verdict files before unblinding.
4. Run `merge-verdicts` and then `judge-check`. Stop if a judge fails the fixed bar. Freeze the control
   rules up front and keep every planted control in the check.

The bank supports a first cutoff estimate. Its initial-only bias check crossed at 0.738 versus 0.792
for the weighted cutoff sample. Bank bias check (internal report `judged_bank/README.md`) · [Bank commands](judged_bank.md)

Judge `t_full` on simulated incremental requests. In blocks that neither ranker trained on, hold out
each block's newest whole publication years as queries, and cluster the older signatures at the
reviewed EPS as seeds. Include cases where both arms abstain. Then confirm the chosen cutoff against
the incumbent on fresh blocks, judging only the decisions where they disagree. If the cutoff fails,
test a two-parameter gate on `logit(p)` and target cluster size next, not a many-feature fitted gate.
The ranker trains only on gold-scored candidates, so it is overconfident on small unlabeled clusters at runtime.

For the cutoff candidate, stamp the reviewed EPS and cutoffs without refitting. Use the no-sweep form
because the sweep's giant F1 tolerance can reject 0.38. Substitute the reviewed EPS if it moved.

~~~sh
uv run --offline --no-sync python -m scripts.eps.apply_reviewed_eps \
  --source-model finalized-1.5 --output-model reviewed-model \
  --eps 0.38 --gate-cutoffs T_FULL T_INIT \
  --review-note "Cite the judged merge check and the judged cutoff packages." \
  --output-report eps-review.json
~~~

### 5. Validate on fresh blocks

Reload the reviewed bundle and verify its EPS and cutoffs. Freeze it before comparing it with shipped
1.4 on fresh blocks disjoint from the cutoff review. Score every eligible signature with `orcid_b3`.
Report B-cubed (B3) with and without singletons. Require pooled B3 F1 ≥ −0.30 points, every size band
≥ −1.0 F1 points, B3 precision ≥ −0.10 points, and cluster count within ±2% of 1.4. Deltas are candidate minus shipped.

Apply the paired-cost rule (internal report `step6_compare/PASS_BAR.md`) to final runtime
disagreements in both views. Match linked targets by shared decision-time seed members. Weight cost by
volume and report paired block-bootstrap intervals and the undecided sensitivities. The 28-block replay
and the 20-block gold holdout used the 1.4 heads. Complete the fresh comparison with the release bundle.
Recreation replay (internal report `step6_compare/RESULT.md`) · Gold holdout (internal report `untouched_holdout/RESULT.md`) · [Release acceptance](release.md#stage-5-fit-evaluate-and-approve)
