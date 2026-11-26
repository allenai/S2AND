# Judge author pairs with the bank

Run `python -m scripts.judging.judged_bank` from the repository root through the project environment.

## Build a package

```text
uv run --offline --no-sync python -m scripts.judging.judged_bank build-package --bank BANK --captured decisions.parquet --strata strata.json --snapshot ARROW_ROOT --output-dir BLIND_DIR --key PRIVATE/key.json
```

Capture one row per comparison. The Parquet columns are:

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

Supply `--block-role-assignments` and `--identity-evidence` to plant controls. The ledger
identifies pure pairs among the unjudged candidates left after sampling. Set aside a capture
stratum with quota zero if the experiment supplies a distinct control pool. Different-person
controls require two nonempty, distinct public ORCID given names. Supply `--orcid-cache` outside
the blind directory for these lookups. Use an empty `controls` map to build without controls.

Give judges only `BLIND_DIR/package.md`. Keep the answer key and bank private. If the draw
produced zero items, use the key's reuse records directly.

## Freeze verdicts and check judges

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
