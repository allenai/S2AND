# Subblocking for Large Blocks

This doc explains S2AND’s “Subblocking” strategy: splitting a too-large clustering problem
into smaller subproblems so memory stays bounded, while making the semantic tradeoffs explicit.

Related docs:
- `docs/stage_memory_estimates.md` (memory telemetry + predictors used by guards)
- `docs/work_plan.md` (performance next steps + backlog)

## What “Subblocking” means here

S2AND’s clustering cost is dominated by pairwise work. For a set of `N` signatures,
dense clustering has `O(N²)` pairs, which becomes infeasible for large blocks.

**Subblocking** is the operational strategy of:

1. Splitting a large block into smaller **subblocks** (often based on blocking keys / heuristics).
2. Running clustering logic on each subblock (or on subblock-sized chunks of work).
3. Optionally running a global step that lets signatures from different subblocks “see” each other
   again, when that global step fits in memory.

Subblocking is primarily about **bounded memory** and **bounded wall time** on giant blocks.
When global steps are skipped, it is also an **approximation**: results can differ from running the same algorithm monolithically over the whole block.

### How subblocks are chosen in practice

Subblock construction is done by `make_subblocks(...)` in `s2and/subblocking.py`. The algorithm is:

1. Split by normalized first-name prefixes (increasing prefix length) until each subgroup is `< maximum_size`.
2. For groups that still cannot be split by first name, repeat prefix splitting on middle names.
3. For groups that still cannot be split, run SPECTER-based clustering (`cluster_with_specter`) to produce about
   `ceil(group_size / maximum_size)` clusters, then deterministically chop any oversize cluster to respect `maximum_size`.
4. Greedily merge very small neighboring subblocks when their combined size stays under `maximum_size`, prioritizing
   strong name-prefix compatibility (and ORCID-derived co-occurrence priors where available).
5. Enforce ORCID co-location: signatures with the same ORCID are moved into one subblock when possible.

Design intent:

- Keep every subblock near the size budget for predictable memory.
- Preserve likely-merge candidates together (name-prefix compatibility + ORCID handling).
- Keep behavior deterministic (sorted traversals + seeded shuffles in SPECTER oversize fallback).

### Terminology (incremental clustering)

For large blocks, common paths include both global/no-seed inference and seeded incremental inference.
For the seeded incremental path, `predict_incremental` has a natural split:

- **Seed signatures / seed clusters (`S`)**: existing clusters we assign into.
- **Unassigned signatures (`U_total`)**: all new signatures to place in the block.
- **`U_sb`**: unassigned signatures in one subblock.
- **Subblock**: a partition of the unassigned signatures processed separately.
- **`phase_b_mode`**: `"exact"` (global B/C/D) or `"subblock_local"` (fallback B/C/D per subblock).

When `U_total` is small (the normal incremental case), monolithic clustering is fine. When `U_total` is large
(giant blocks), Subblocking is required.

## The incremental pipeline phases (A–D)

Conceptually, `_predict_incremental_helper` can be understood as four phases:

- Notation used below:
  - `U_total`: all unassigned signatures in the block.
  - `U_sb`: unassigned signatures in one subblock.
  - `R_total`: all post-assignment singletons when Phase D runs globally.
  - `R_sb`: post-assignment singletons inside one subblock when fallback is subblock-local.

- **Phase A — seed distances**: distances from each unassigned signature to all seed signatures/clusters.
  - Work: `O(U_total×S)` total, executed across subblocks/chunks.
  - Memory risk: pair buffers / intermediate matrices (bounded via chunking).
- **Phase B — pre-cluster unassigned**: cluster the unassigned signatures among themselves.
  - `phase_b_mode="exact"`: runs globally on `U_total`; work+memory `O(U_total²)` (dense distances dominate).
  - `phase_b_mode="subblock_local"`: runs per subblock; total work `O(Σ U_sb²)`, peak memory `O(max U_sb²)`.
- **Phase C — merge at the cluster level**: “average-of-averages” merge using Phase A aggregates.
  - `phase_b_mode="exact"`: `O(U_total×C)` where `C` is cluster count.
  - `phase_b_mode="subblock_local"`: `O(Σ U_sb×C)` with no cross-subblock unassigned interaction.
- **Phase D — assign + singleton recluster**: assign to closest seed cluster; recluster remaining singletons.
  - Assignment pass is linear in unassigned count.
  - Singleton recluster dominates: global mode `O(R_total²)` vs subblock-local fallback `O(Σ R_sb²)`.

Subblocking is mostly about deciding which of these phases can run “per subblock” and which must run “globally”.

## Why naïve Subblocking drifts

The original fully-subblocked approach ran **all phases A–D inside each subblock**. That keeps memory bounded, but it changes semantics:

- **Phase A is safe to subblock**: each unassigned signature’s distance to seeds depends only on the seed set.
- **Phases B and D are not safe to subblock**: they require unassigned signatures from *different* subblocks to
  interact. If they cannot, merges that would happen monolithically can be missed.

Evidence (10k-signature probe, seed=43, threshold=7500):

- Fully-subblocked A–D caused **4.27% partition drift** (`427/10000`) vs monolithic output.

## Phase-split Subblocking (current design)

The phase-split design keeps the memory benefits of Subblocking while recovering monolithic-equivalent behavior whenever the “global” steps fit in memory:

1. **Always subblock Phase A** (bounded memory, safe semantics).
2. After Phase A completes, run Phases **B/C/D globally only if the estimated global Phase B dense vector fits budget**:
   `recluster_bytes(U_total) <= phase_b_budget_bytes`.
3. If global Phase B is over budget, deliberately fall back to **subblock-local B/C/D** (approximate semantics)
   rather than erroring or risking OOM.

### Phase placement rationale

| Phase | Subblocked? | Why |
|---|---|---|
| A: seed distances | Yes (+ chunked) | Safe: depends only on seeds. Total `O(U_total×S)`; peak memory is governed by chunking. |
| B: pre-cluster unassigned | Global when possible | `exact`: `O(U_total²)` memory/time. `subblock_local`: `O(max U_sb²)` peak memory, `O(Σ U_sb²)` total work. |
| C: average-of-averages | Global when possible | Must align with B scope. `exact`: global over `U_total`; fallback: per-subblock only. |
| D: assign + singleton recluster | Global when possible | `exact`: singleton recluster over `R_total`; fallback: per-subblock over `R_sb`, which can miss cross-subblock merges. |

### Order-independence (Phase A sum+count)

Phase A runs across many subblocks and chunks. To avoid floating-point order effects, Phase A accumulates
`(sum, count)` per `(unassigned_signature, seed_cluster)` and converts to averages once at the end.
This makes Phase A commutative and eliminates a known source of cross-run drift.

## Memory guards and explicit “approximate mode”

### Phase B memory guard

Before running global Phase B (`phase_b_mode="exact"`), the implementation estimates the condensed distance vector size on `U_total`:

```
U_total = len(all_unassigned)
recluster_bytes = U_total * (U_total - 1) // 2 * 8   # float64 condensed vector
```

| U_total | Condensed vector |
|---:|---:|
| 1,000 | ~4 MB |
| 5,000 | ~100 MB |
| 10,000 | ~400 MB |
| 50,000 | ~10 GB |
| 100,000 | ~40 GB |
| 600,000 | ~1.3 TiB |

When `recluster_bytes > phase_b_budget_bytes`, runtime falls back to subblock-local B/C/D.
`phase_b_budget_bytes` is derived from Phase A memory budgets and post-Phase-A live-state estimates
(not simply a fixed 20% constant).

### Phase A accumulator overflow (partial Phase A)

Phase A’s accumulator (`signature_to_cluster_sum_count`) can grow very large on big `U_total×S` workloads. When the
accumulator exceeds its configured entry limit, Phase A stops early and remaining unassigned signatures proceed
with **partial** seed distances.

This is intentionally conservative (avoid OOM), but it changes semantics. The degraded mode is explicitly surfaced:

- Result payload: `phase_a_accumulator_overflow_early_stop: bool`
- Log telemetry: `Telemetry: phase_split_phase_a_overflow overflow_early_stop=<bool> ...`
- Regression test: `tests/test_cluster_incremental.py::test_phase_a_overflow_surfaces_in_result_and_telemetry`

## Telemetry contract (how to tell what happened)

Every incremental result includes a machine-readable contract for the key Subblocking decision points:

- `phase_b_mode`: `"exact"` or `"subblock_local"`
- `phase_b_budget_bytes`: budget used for the decision
- `phase_b_required_bytes`: estimated condensed-vector bytes
- `phase_a_accumulator_overflow_early_stop`: whether Phase A stopped early

Interpretation:

- `phase_b_mode="exact"` means B/C/D ran globally on `U_total` and is intended to match monolithic semantics
  (given the same inputs and no earlier degraded mode).
- `phase_b_mode="subblock_local"` means B/C/D ran per subblock and is intentionally approximate; cross-subblock
  unassigned interactions are missing.

## Operational guidance (when to expect what)

### Normal incremental (typical)

When `U_total << S` (few new signatures, many seeds), global Phase B is small and typically runs in `"exact"` mode.
Subblocking mainly serves as a Phase A memory governor.
When there are no seeds (`S=0`), current `predict_incremental` behavior falls back to the monolithic incremental
helper for parity, so this phase-split seeded analysis does not apply directly.

### Giant blocks (expected approximate behavior)

For very large `U_total` (e.g., hundreds of thousands of unassigned signatures), `phase_b_mode="subblock_local"`
is expected and is what makes the workload feasible. The goal becomes **bounded memory + bounded runtime**, not
monolithic equivalence.

Rule-of-thumb sizing (illustrative):

- 600k signatures split into ~10k subblocks ⇒ ~60 signatures/subblock on average.
- Dense Phase B inside one ~60-signature subblock is tiny:
  `60 * 59 // 2 * 8 = 14,160 bytes` for the condensed float64 vector.

## Evidence (parity gate)

10k variance probe: seed=43, threshold=7500, python backend (2026-02-24).

| Metric | Result |
|---|---|
| `cluster_equivalent` | `True` |
| Signature partition diff | `0 / 10000` |
| Runtime vs monolithic | −106s |
| Peak RSS vs monolithic | −4.4 GB |

Artifact: `scratch/big_block/compare_phase_split_10k_seed43_python_20260224.json`.

## Active controls

| Input | Default | Effect |
|---|---|---|
| `predict_incremental(..., total_ram_bytes=<int>)` | unset | Optional explicit RAM input for Phase A/B budget derivation. |
| `predict_incremental(..., max_chunk_pairs=<int>)` | `1000000` | Optional explicit cap on Phase A `chunk_pairs` (pair buffer size). If None, uses `PHASE_A_MAX_CHUNK_PAIRS_DEFAULT` (1M). Set to 0 to disable cap and rely solely on memory-budget-derived limits. Can be increased (e.g., 100M) for large-RAM machines. |
| RAM autodetect safety factor | `0.8` | Applied to detected cgroup/host RAM before deriving budgets. |

## Residual risks (what can still surprise you)

1. **Approximate mode on large `U_total`**: over-budget Phase B falls back to `subblock_local`, which is intentionally non-equivalent.
2. **Autodetect uncertainty**: when `total_ram_bytes` is not passed explicitly, budgets depend on best-effort RAM detection and the `0.8` safety factor.
3. **Partial Phase A**: accumulator overflow can stop Phase A early; this is now machine-visible via `phase_a_accumulator_overflow_early_stop`.
