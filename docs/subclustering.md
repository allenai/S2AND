# Subclustering (Subblocking) for Large Blocks

This doc explains S2AND’s “subclustering” strategy: splitting a too-large clustering problem
into smaller subproblems so memory stays bounded, while making the semantic tradeoffs explicit.

Status: phase-split incremental subclustering promoted to default (2026-02-24).

Related docs:
- `docs/memory/stage_memory_estimates.md` (memory telemetry + predictors used by guards)
- `docs/rust/roadmap.md` (performance work that interacts with subclustering)

## What “subclustering” means here

S2AND’s clustering cost is dominated by pairwise work. For a set of `N` signatures,
dense clustering has `O(N²)` pairs, which becomes infeasible for large blocks.

**Subclustering** is the operational strategy of:

1. Splitting a large block into smaller **subblocks** (often based on blocking keys / heuristics).
2. Running clustering logic on each subblock (or on subblock-sized chunks of work).
3. Optionally running a global step that lets signatures from different subblocks “see” each other
   again, when that global step fits in memory.

Subclustering is primarily about **bounded memory** and **bounded wall time** on giant blocks.
When global steps are skipped, it is also an **approximation**: results can differ from running the
same algorithm monolithically over the whole block.

### Terminology (incremental clustering)

This repo’s large-block work is mostly in `predict_incremental`, which has a natural split:

- **Seed signatures / seed clusters (`S`)**: existing clusters we assign into.
- **Unassigned signatures (`U`)**: new signatures to place.
- **Subblock**: a partition of the unassigned signatures processed separately.

When `U` is small (the normal incremental case), monolithic clustering is fine. When `U` is large
(giant blocks), subclustering is required.

## The incremental pipeline phases (A–D)

Conceptually, `predict_incremental_helper` can be understood as four phases:

- **Phase A — seed distances**: distances from each unassigned signature to all seed signatures/clusters.
  - Work: `O(U×S)` pairs.
  - Memory risk: pair buffers / intermediate matrices (bounded via chunking).
- **Phase B — pre-cluster unassigned**: cluster the unassigned signatures among themselves.
  - Work + memory: `O(U²)` (dense distances dominate).
- **Phase C — merge at the cluster level**: “average-of-averages” merge using Phase A aggregates.
  - Work: `O(U×C)` where `C` is cluster count; typically cheap.
- **Phase D — assign + singleton recluster**: assign to closest seed cluster; recluster remaining singletons.
  - Work: `O(U + R²)` where `R` is remaining singleton count.

Subclustering is mostly about deciding which of these phases can run “per subblock” and which must run “globally”.

## Why naïve subclustering drifts

The original fully-subblocked approach ran **all phases A–D inside each subblock**. That keeps memory bounded,
but it changes semantics:

- **Phase A is safe to subblock**: each unassigned signature’s distance to seeds depends only on the seed set.
- **Phases B and D are not safe to subblock**: they require unassigned signatures from *different* subblocks to
  interact. If they cannot, merges that would happen monolithically can be missed.

Evidence (10k-signature probe, seed=43, threshold=7500):

- Fully-subblocked A–D caused **4.27% partition drift** (`427/10000`) vs monolithic output.

## Phase-split subclustering (current design)

The phase-split design keeps the memory benefits of subclustering while recovering monolithic-equivalent behavior
whenever the “global” steps fit in memory:

1. **Always subblock Phase A** (bounded memory, safe semantics).
2. After Phase A completes, run Phases **B/C/D globally only if `U` fits in memory**.
3. If global Phase B is over budget, deliberately fall back to **subblock-local B/C/D** (approximate semantics)
   rather than erroring or risking OOM.

### Phase placement rationale

| Phase | Subblocked? | Why |
|---|---|---|
| A: seed distances | Yes (+ chunked) | Safe: depends only on seeds. `O(U×S)`; memory is governed by chunking. |
| B: pre-cluster unassigned | Global when possible | Needed for monolithic semantics. `O(U²)` memory is the limiting factor. |
| C: average-of-averages | Global when possible | Reads Phase B clusters; cheap but must align with B. |
| D: assign + singleton recluster | Global when possible | Needed for monolithic semantics; singleton recluster is the secondary drift source. |

### Order-independence (Phase A sum+count)

Phase A runs across many subblocks and chunks. To avoid floating-point order effects, Phase A accumulates
`(sum, count)` per `(unassigned_signature, seed_cluster)` and converts to averages once at the end.
This makes Phase A commutative and eliminates a known source of cross-run drift.

## Memory guards and explicit “approximate mode”

### Phase B memory guard

Before running global Phase B, the implementation estimates the condensed distance vector size:

```
U = len(all_unassigned)
recluster_bytes = U * (U - 1) // 2 * 8   # float64 condensed vector
```

| U | Condensed vector |
|---:|---:|
| 1,000 | ~4 MB |
| 5,000 | ~100 MB |
| 10,000 | ~400 MB |
| 50,000 | ~10 GB |
| 100,000 | ~40 GB |
| 600,000 | ~1.3 TiB |

When `recluster_bytes` exceeds the Phase B budget (20% of available memory headroom), runtime falls back to
subblock-local B/C/D.

### Phase A accumulator overflow (partial Phase A)

Phase A’s accumulator (`signature_to_cluster_sum_count`) can grow very large on big `U×S` workloads. When the
accumulator exceeds its configured entry limit, Phase A stops early and remaining unassigned signatures proceed
with **partial** seed distances.

This is intentionally conservative (avoid OOM), but it changes semantics. The degraded mode is explicitly surfaced:

- Result payload: `phase_a_accumulator_overflow_early_stop: bool`
- Log telemetry: `Telemetry: phase_split_phase_a_overflow overflow_early_stop=<bool> ...`
- Regression test: `tests/test_cluster_incremental.py::test_phase_a_overflow_surfaces_in_result_and_telemetry`

## Telemetry contract (how to tell what happened)

Every incremental result includes a machine-readable contract for the key subclustering decision points:

- `phase_b_mode`: `"exact"` or `"subblock_local"`
- `phase_b_budget_bytes`: budget used for the decision
- `phase_b_required_bytes`: estimated condensed-vector bytes
- `phase_a_accumulator_overflow_early_stop`: whether Phase A stopped early

Interpretation:

- `phase_b_mode="exact"` means the run is intended to match monolithic semantics (given the same inputs).
- `phase_b_mode="subblock_local"` means the run is intentionally approximate; cross-subblock merges are missing.

## Operational guidance (when to expect what)

### Normal incremental (typical)

When `U << S` (few new signatures, many seeds), global Phase B is small and typically runs in `"exact"` mode.
Subclustering mainly serves as a Phase A memory governor.

### Giant blocks (expected approximate behavior)

For very large blocks (e.g., hundreds of thousands of signatures), `phase_b_mode="subblock_local"` is expected
and is what makes the workload feasible. The goal becomes **bounded memory + bounded runtime**, not monolithic
equivalence.

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
| RAM autodetect safety factor | `0.8` | Applied to detected cgroup/host RAM before deriving budgets. |
| `S2AND_PHASE_A_MAX_CHUNK_PAIRS` | `500000` | Caps Phase A `chunk_pairs` (pair buffer size) to avoid giant Python buffers; set `0` to disable. |

## Residual risks (what can still surprise you)

1. **Approximate mode on large `U`**: over-budget Phase B falls back to `subblock_local`, which is intentionally non-equivalent.
2. **Autodetect uncertainty**: when `total_ram_bytes` is not passed explicitly, budgets depend on best-effort RAM detection and the `0.8` safety factor.
3. **Partial Phase A**: accumulator overflow can stop Phase A early; this is now machine-visible via `phase_a_accumulator_overflow_early_stop`.
