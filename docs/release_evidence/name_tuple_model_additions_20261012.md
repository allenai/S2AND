# Model-checked name aliases

Add 276 manually reasonable first-name pairs to the curated source and canonical artifact.
The canonical artifact contains 5,330 pairs after this addition.
[The evidence table](name_tuple_model_additions_20261012.csv) preserves each manual reason and evaluated alias group.

The proposal retains complete groups of aliases that affect the same evaluated block.
Each tested block therefore preserves its measured baseline or full 758-pair proposal constraint state.
Frozen features, held-out LightGBM heads, and cluster thresholds remain identical across proposals.
The subset results derive from those measured partitions; no new model run occurs for the subset.

| Known-label scope | Gold links recovered | Gold links lost | New false links | False links resolved |
|---|---:|---:|---:|---:|
| Latest metadata | 80,469 | 0 | 0 | 13 |
| Older risk blocks | 374 | 0 | 0 | 356 |

The source comparisons cover 361 latest blocks, 84 older risk blocks, and 17 external blocks.
External tests use seven separate model configurations at their own thresholds.
The retained external blocks have no measured gold or negative pair changes.
Unknown signatures remain in cluster inference but do not contribute to primary benchmark metrics.
The strict filter also excludes groups with new errors under secondary labels or the earlier 501-pair arm.

The older scope does not include every older block with positive-only alias changes.
These results do not guarantee accuracy on unseen data or establish external production-model accuracy.
Do not sum metadata versions, which can represent the same people.

Reproduce the source comparisons with the commands in
`scratch/missing_name_pairs_20261012/model_evaluation/run_record.json`,
`historical_run_record.json`, and `external_run_record.json`.
Apply `conservative_subset.py` to those saved results to reproduce the 276-pair selection.
