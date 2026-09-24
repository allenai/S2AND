# S2AND Docs

This directory holds user guides, reference specifications, release
operations, and contributor notes. Project overview and install quick start:
[../README.md](../README.md).

## Project status

[release.md](release.md) is the sole v1.3 release policy and runbook, and the
one place that records release status. Current-behavior docs do not imply that
canonical artifacts or a v1.3 model have already been published. Callouts
labeled **Update at v1.3 release** mark workflows that cannot work until the
v1.3 model and public-format-1 data exist;
[release.md](release.md#stage-5-fit-evaluate-and-approve) requires resolving
every one of them (`git grep -n "Update at v1.3 release"`) before the final
package commit.

## User guides

- Install and setup: [install.md](install.md)
- Data download, layout, and `path_config.json`: [data.md](data.md)
- Production inference: [production_inference.md](production_inference.md)
- Training and evaluation: [training.md](training.md)
- Large blocks and subblocking: [subblocking.md](subblocking.md)
- Threading and parallelism: [threading.md](threading.md)
- Environment variables: [environment.md](environment.md)
- Reproducibility and paper-era notes: [root README](../README.md#reproducibility)

## Reference

- Canonical name rules: [canonical_names.md](canonical_names.md)
- Public Arrow/name-count formats and the direct Rust dataset schema:
  [rust/arrow_dataset_spec.md](rust/arrow_dataset_spec.md)
- Release notes: [release_notes.md](release_notes.md)

## Release operations

- v1.3 release policy and runbook: [release.md](release.md)
- Production command reference, in release-stage order:
  [../scripts/production/README.md](../scripts/production/README.md),
  including the
  [promoted incremental performance report](../scripts/production/README.md#promoted-incremental-performance-report)
- Maintained, verification, and archived script catalog:
  [../scripts/README.md](../scripts/README.md)

## Contributors

- Development workflow, local CI, and native build troubleshooting:
  [development.md](development.md), including
  [runtime routing verification](development.md#runtime-routing-verification)
- Test suite guide and evidence standard: [../tests/README.md](../tests/README.md)
- Native `s2and-rust` package overview and local build:
  [../s2and_rust/README.md](../s2and_rust/README.md)

## Evidence and measurements

- Bulk Arrow optimization measurements:
  [initial optimizations](rust/profiling/2026-09-04-bulk-optimizations.md) and
  [specialization follow-up](rust/profiling/2026-09-04-bulk-followup.md)
- Manual review of legacy name-tuple aliases:
  [release_evidence/name_tuple_legacy_adjudication_v1.md](release_evidence/name_tuple_legacy_adjudication_v1.md)

Dated profiling snapshots under `rust/profiling/` are immutable historical
evidence. Their dates and commands are preserved intentionally; they are not
current commands or release gates.

## Glossary

| Term | Meaning |
|---|---|
| Signature | One author position on one paper: the unit S2AND clusters. |
| Block | Signatures sharing a blocking key (`author_info.block`, for example `j smith`). Clustering happens within a block. |
| Subblock | A bounded partition of a large block; see [subblocking.md](subblocking.md). |
| Cluster, profile | A set of signatures predicted or recorded as one author. |
| Seeds | Existing assignments supplied with a request: `cluster_seeds_require` maps signatures to existing profile IDs, and `cluster_seeds_disallow` lists signature pairs that must not be linked. |
| Claimed profile | An existing profile whose membership is supplied as required seeds; prediction preserves it. |
| Altered profile | A claimed profile listed in `altered_cluster_signatures`. Incremental linking scores new papers against its natural components, then restores the original profile; bulk prediction ignores the marker. |
| Initial-only | A signature whose first name is only an initial (for example `J.`). |
| Bulk (full-block) prediction | Clustering every signature in the requested blocks: `predict` / `predict_from_arrow`. |
| Incremental prediction, query | Attaching new signatures (queries: requested signatures absent from `cluster_seeds_require`) to existing seed profiles: `predict_incremental` / `predict_incremental_from_arrow`. |
| Pairwise model | The LightGBM boosters that score signature pairs: `main`, and `nameless`, which excludes name features. |
| EPS | The distance threshold for average-linkage agglomerative clustering, selected on validation data only. |
| Promoted incremental linker | The learned link-or-abstain model (a LightGBM booster over 53 linker features) that attaches queries on the Arrow incremental route and initial-only groups after Arrow subblocking. |
| Raw planner | The Rust candidate planner that reads raw Arrow tables through batch indexes to retrieve candidate seed components for each query. |
| Batch index | An S2AND-generated `*_batch_index.bin` sidecar that maps lookup keys to Arrow record batches; required by `ArrowDataset.open(...)`. |
| Pairwise-only vs complete bundle | A training-stage bundle with only the pairwise boosters (EPS `pending` or `calibrated`) vs the loadable production bundle that adds the linker. |
| Version axes | Package version (`s2and` and `s2and-rust`, `1.0.0`), model/public-data release version (`1.3`), and public Arrow/name-count format (`1`). |
| `canonical_v2` | The name and feature normalization behavior shipped with `1.0.0`; see [canonical_names.md](canonical_names.md). |

## Scope notes

- User guides and reference docs describe current behavior, knobs, and
  verification commands.
- The vendored `s2and_rust/vendor/cld2/README.md` is third-party documentation.
