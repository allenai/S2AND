# Judge author pairs with the bank

Run `python -m scripts.judging.judged_bank` from the repository root through the project environment.

## Capture production decisions

Use an existing EPS replay with one explicit EPS. For example, reuse available distance caches:

```text
uv run --offline --no-sync python -m scripts.eps.sweep_eps_on_linking_gold --dataset DATASET --arrow-root ARROW_ROOT --model-path MODEL --batching-threshold 15000 --distance-cache-dir DISTANCES --output-dir SWEEP_DIR --eps-values 0.38 --capture-dir CAPTURE_DIR
```

The replay writes one Parquet file per block. Each row is the last linker decision for one
`query_signature_id`. Bulk-only blocks have no linker decisions and get an empty table.
Each row includes `query_view`, `eps`, `pass_index`, `final_action`, and `exclusion_reason`.
`candidate_component`, `candidate_signature_ids`, and `chosen_probability` describe the scored
choice, including ordinary gate abstentions. `runner_up_component` and `runner_up_score`
describe the highest-scoring allowed row from another component. Forced links retain their
scores and targets and are excluded from judging.

`target_component` and `target_signature_ids` describe the accepted link, when present.
`target_seed_signature_ids` omits members that attached in earlier passes. Residual clusters
can supply new seeds. Compare linked targets across arms with `targets_match` in
`scripts.judging.decision_capture`. The targets match when their seed lists intersect.

## Build a package

```text
uv run --offline --no-sync python -m scripts.judging.judged_bank build-package --bank BANK --captured decisions.parquet --strata strata.json --snapshot ARROW_ROOT --output-dir BLIND_DIR --key PRIVATE/key.json
```

Pass a replay capture file or its directory to `--captured` directly. Set quotas for the
`full` or `initial_only` views you want judged. The bank samples scored choices for links
and ordinary gate abstentions. It skips rows with an exclusion reason and views without a
quota. With an identity ledger, the bank uses its scored labels for reuse checks.

For experiment-defined comparisons and strata, provide one row per comparison with these
Parquet columns:

| Column | Value |
|---|---|
| `block` | Name block in the frozen Arrow snapshot |
| `query_signature_id` | Signature shown as Record A |
| `target_signature_ids` | All candidate members at decision time |
| `labeled_signature_ids` | Labeled members of that candidate, or an empty list |
| `shown_group_signature_ids` | Optional Group B display. The tool samples this list when absent. |
| `stratum` | A key in the quotas map |
| `target_component`, `model`, `arm` | Optional private strings |
| `score`, `cutoff` | Optional probabilities in [0, 1] |

Keep one row per distinct query and shown Group B. Assign strata in the experiment's capture code.
An example strata file is:

```json
{
  "package": "2026-11-26-example",
  "seed": 42,
  "max_shown": 8,
  "quotas": {"lower": 30, "upper": 50},
  "controls": {"same": 6, "different": 6}
}
```

The tool reuses an earlier pair when its query and block match, every previously shown
signature is in the candidate, and those signatures make up more than half of the union of
the candidate's labeled members and its previously or currently shown members. The private
key records every matching bank pair and the bank version. It also records the number drawn
divided by the eligible unjudged count for each stratum.

Pass `--sample-before-reuse` when the quota counts both new and reused judgments. Each
stratum's eligible count then includes bank matches. The tool draws the quota from that
full stratum before replacing matching draws with reused verdicts. Both kinds of draws
carry the same inclusion probability. The private key reports fresh and reused counts.
Pass `--max-authors N` to show the first N author-list entries per paper and an ellipsis
for omitted entries. The focal author remains visible in its own field.

Supply `--block-role-assignments` and `--identity-evidence` to plant controls. The ledger
identifies pure pairs among the unjudged candidates left after sampling. Set aside a capture
stratum with quota zero if the experiment supplies a distinct control pool. Use an empty `controls`
map to build without controls. Set `--orcid-cache` outside the blind directory for public ORCID lookups.

Set a positive `different_same_name` control quota to plant distinct ledger identities
whose public ORCID person records show the same normalized given name or leading initial.
The ordinary `different` control requires two nonempty, distinct normalized given names. The tool
assigns each pair to one slot when a pair qualifies for both.
Both control types have gold verdict `different` and contribute to the combined judge check.

Give each judge `BLIND_DIR/package.md` in a separate folder. Keep the answer key and bank private. If the draw
produced zero items, use the key's reuse records directly.

## Freeze verdicts and check judges

Freeze both verdict files before merging. Set the thresholds before building the package.

```text
uv run --offline --no-sync python -m scripts.judging.judged_bank merge-verdicts --bank BANK --key PRIVATE/key.json --judge fable judgments_fable.jsonl --judge neon judgments_neon.jsonl
uv run --offline --no-sync python -m scripts.judging.judged_bank judge-check --bank BANK --package 2026-11-26-example --judge-different 0.8 --consensus-different 0.7 --consensus-same 0.85
```

Each judge file needs one JSON object per item with its integer `item`, a `verdict` of `same`,
`different`, or `cannot_tell`, and a `confidence` of `low`, `medium`, or `high`.
Reimporting the same bytes returns the existing version. Import later judges with the same key.
Use `--judge` to choose judges for threshold checks and `--exclude-disputed` to exclude flagged controls.
An unmet threshold exits with status 1. A consensus verdict requires both judges to agree on
`same` or `different`.

Read the highest numbered directory under `BANK/versions`. Its `judgments.parquet` contains one
row per package item and judge. Its `pairs.parquet` indexes occurrences and per-item judge-pair
consensus by a stable hash of the query and sorted shown signature IDs. Find the original
JSONL by its SHA-256 file name under the version where it was first added.

Use this command for package assembly, verdict storage, and judge checks in place of the
per-package implementations. Keep population definitions, threshold fits, and paired costs in
the experiment. `review_packets.py` continues to support training packets and supplies the
blind validation and Markdown renderer. `gate_exemplars.py` supplies frozen paper metadata.
