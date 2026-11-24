# S2AND 1.5 Training, Split, and Labeling Design

This design addresses the data-split, leakage, distribution, and release-engineering issues surfaced during the S2AND `1.4` build (`2026-11-20` to `2026-11-23`).

## Core rules

1. **One block key = one lifecycle role.** Assign every canonical block key a single role via seeded SHA-256 hash (`scripts/production/data/assign_block_roles.py`), with multi-block author closure (`strict` union-find or `home_pool` mirroring `cross_holders.parquet` in Gold ORCID v2 to avoid giant-component percolation). A block key keeps the same role across classic benchmarks, Gold ORCID v2, production block exports, and LLM-adjudicated packets.
2. **Partition `orcid-blocks-v2` by block instead of cross-fitting the full pipeline.** Gold ORCID v2 provides 1.77M development blocks (~2,800 giants >10k) and 888k holdout blocks (1,443 giants). Block-disjoint pools keep every downstream stage strictly out-of-sample for upstream models without nested cross-fitting.
3. **Separate linker-table construction from model release.** Freeze validated linker/gate task tables before starting a release fit. Do not mix data refresh, model fitting, and new giant-block task design into one release run.
4. **Ship what was validated.** Do not pool `train + val` and refit after selection; refit seed noise (±0.4 F1 pts on small benchmarks) shifts downstream distances seen by the linker.
5. **Redact and gate once at intake.** Hash email fields, strip plaintext emails from free text (`affiliations`, `title`, `abstract`), filter non-S2AG papers, and redact unshareable abstracts to `"Has Abstract"` (preserving `has_abstract` and redacting correction history) *before* Arrow conversion or pair sampling. The `write_release_bundles.py` CLI scans final bundles automatically by default (deleting output on failure).
6. **Build linker candidates from batch-clustered profiles, after pairwise fit, and freeze native top-25 windows before any labeling.** Never train on historical fragmented production clusters, and never label candidate lists built with a different retrieval planner than the trainer uses.

## Eight-pool split topology (seven active pools + `dev_test`)

All legacy `1.4` giants (`a khan`, `a silva`, `h wang`, `j smith`, `s gupta`, `s lee`, `s park`, `s feldman`) and the 27 Batch 1–6 blocks are permanently assigned to `dev_test` (regression/iteration only). `f matsen` (1 scored identity) is excluded.

| Pool | Sources & allocation | Fits / tunes | Key guarantees |
| --- | --- | --- | --- |
| `pairwise_train` | Single clean 80% split of 7 benchmarks (reclaiming the ~9% `unused_internal_test` discarded by 1.4's double 80/10/10 split) + fresh augmentation (+ optional gated `>45yr` historical or ORCID dev pairs) | Main (33-feat) & Nameless (17-feat) pairwise LightGBM heads | Zero block/author overlap with any downstream or test pool. Freeze heads before generating downstream scores. |
| `pairwise_val` | 10% of 7 benchmarks (macro-7 + INSPIRE-capped) + 20–30 stratified `orcid-blocks-v2/dev` blocks (small/med/large/giant, incl. non-English & missing-SPECTER2) | Pairwise HPO / early stopping / feature & missingness safety checks | Catches giant/non-English regressions at Stage 3; no longer the sole EPS selector. |
| `linker_train` | 40–60 diverse `orcid-blocks-v2/dev` blocks (Western, Iberian, Slavic, CJK, South-Asian) + optional OOF benchmark slice | Retrieval weights (replace `h_wang_any_input_v2`) & 53-feature LightGBM ranker | Replays real `full_name` and real `initial_only` queries (**preserving real `author_middle`**; retire middle-wiped `visibleq:` clones). `no_safe` via deterministic **whole-author holdout before retrieval**. |
| `gate_fit` | 15–25 disjoint `orcid-blocks-v2/dev` blocks (~15k queries) | Gate medians, `StandardScaler`, feature selection (`k`), `C`, and logistic weights (grouped block CV) | 100% out-of-sample pairwise & ranker scores; enriched to **25–40% whole-author-holdout `no_safe`** across all size bands. Never pooled with `gate_thresh`. |
| `gate_thresh` | 10–15 disjoint `orcid-blocks-v2/dev` blocks at **natural production request frequencies** + Sol 6.2 non-ORCID natural sample | Prior/intercept calibration and `w_fa` operating point (optionally per-bucket `full_name` vs `initial_only`) | Calibrates link volume to true production prevalence (natural absentees + retrieval misses), not artificial holdout prevalence. |
| `eps_cal` | ~40 disjoint `orcid-blocks-v2/dev` blocks across size × script/region bands (plus benchmark `pairwise_val` as a secondary diagnostic) | Production-route EPS (step `0.02`), close-pair bonus, subblocking & post-linkage corrections | **100% of ORCID authors retained** (no final-test stripping → all ≥50-paper authors intact; authorized via `--block-role-assignments` in `apply_reviewed_eps.py`). |
| `dev_test` | ~40 disjoint `orcid-blocks-v2/dev` blocks + legacy giants + 27 Batch 1–6 blocks + fresh paired Sol disagreements | Model-vs-model iteration & go/no-go | Never fitted on. |
| `final_test` *(sealed)* | 130–300 fresh `orcid-blocks-v2/holdout` blocks (incl. 40–60 fresh giants) + sealed Sol 6.2 panel + canonical classic outer test (regression report) | **Nothing.** Scored once per frozen release candidate | Requires verified `(corpus_paper_id, position)` label joins (`prepare_identity_labels.py --expected-split holdout`). Both ORCID and Sol primary endpoints must be non-negative to ship. |

## First 1.5 experiment order

1. **Experiment 1 (highest ROI, zero schema risk):** Keep 1.4's pairwise heads and 53-feature candidate ranker fixed; refit the existing 240-feature gate on `gate_fit` (real `initial_only` with `author_middle` + whole-author-holdout `no_safe`) and calibrate `w_fa` on `gate_thresh`. Measure cross-subblock recovery on unseen `>15k` blocks (1.4's `-0.46` pooled gap across Batches 4–6) plus non-ORCID precision.
2. **Experiment 2:** Evaluate compact asymmetric middle & career-year gate features (`query_mid_x_cand_no_mid_x_size`, `query_mid_overlap_frac`, `query_mid_conflict_any`, `year_predates_log`) and retrieval re-tuning only after Experiment 1 establishes the clean baseline.
3. **Experiment 3 (conditional):** Lift `YEAR_DIFF_CAP=50` only if backed by targeted `>45yr` historical labeled pairs (`≥5,000` pairs so splits exceed `min_child_samples`).

## Sol 6.2 blind judging protocol & phased budget

Use ORCID Gold v2 and classic benchmarks for bulk pairwise training, ranker training, and modern STEM recall. Reserve `gpt-6.2-sol` (Sol 6.2) for ORCID's structural blind spots: non-ORCID / pre-2010 records, non-English & regional name regimes, institutional homonyms (e.g., BARC), non-ORCID marginal linker accepts/abstains, constructed-negative contamination, and post-linkage rule precision.

### Core budget (~6,600 packets / ~13,200 dual-pass judgments)
- **Phase 0 — Pilot & human anchor (200 packets + 200–300 blind human items):** Measure token cost, consensus rate, clustered design effect, and Sol directional skew per stratum before scaling.
- **Phase 1A — Gate prevalence, contamination & reliability (1,800 packets):** 600 natural production requests across 3 size bands (`gate_thresh`), 500 constructed-negative contamination checks, 500 non-ORCID reliability checks, 200 seed-purity / one-sided middle-initial checks.
- **Phase 1B — Sealed final release panel (3,000 packets):** 1,800 probability-weighted natural production requests (600 each across `≤1k`, `1k–15k`, `>15k`; ~±1.24 pp CI at 5% error) + 6 challenge cohorts × 200 (pre-2010/no-ORCID, non-English/missing-vector, East-Asian, South-Asian/BARC, marginal `initial_only`/abstains, residual splits/missed merges).
- **Phase 1C — Per-release paired disagreements (1,600 packets):** Fresh `1.5 != 1.4` disagreements (`600` bulk diffs accounting for ~51% "can't tell" on raw paper pairs, `600` linker attach/abstain diffs, `200` correction diffs, `200` controls).
- **Phase 2 (conditional):** `5,000` `>45yr` historical pairwise training pairs (only if lifting `YEAR_DIFF_CAP=50`); `1,200–3,000` gate hard-negative packets (only if Phase 1A reliability flags >5 pp error after reweighting); `700–1,500` one-time cross-family audit of AMiner v6 merges & unknown signatures.

### Non-negotiable packet rules (`scripts/production/data/review_packets.py`)
- **Strict field allowlist:** Paper metadata only for pair packets (never predicted-cluster summaries, scores, ranks, split names, cluster IDs, or ORCIDs).
- **Two isolated passes** with independently randomized A/B or candidate order; consensus only counts; report splits and "insufficient evidence" separately (require Cohen's κ ≥ 0.85 on decided items).
- **10–15% embedded ORCID gold controls** per packet (drop/re-examine below 90% control accuracy) + **10–20% cross-family model audit** on sealed test packets and classic benchmarks.
- **No "any abstain drops query" rule:** Never discard an entire 25-candidate query when a secondary candidate receives `abstain`.
- **Automatic ORCID selection-bias logging:** Every gate/EPS evaluation logs linked Gold-ORCID share and linked median publication year alongside B3 metrics.
