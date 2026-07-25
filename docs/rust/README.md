# Rust Docs

The `s2and_rust` extension accelerates S2AND's most expensive stages (featurization,
preprocessing, constraint lookups) by replacing Python hot paths with Rust via PyO3.
This folder documents the runtime contract, benchmark evidence, and migration plans
for the Rust integration.

---

## Documents

| File | Purpose |
|---|---|
| [baselines.md](baselines.md) | Rust performance/baseline command authority and artifact conventions. Start here when refreshing Rust evidence; use the v1.3 runbook for release acceptance. |
| [runtime.md](runtime.md) | Runtime contract: explicit Python/Rust routes, exact native-version loading, failure semantics, Arrow validation, and verification commands. |
| [artifact_formats.md](artifact_formats.md) | Current artifact-format choices and rejected alternatives. |
| [arrow_dataset_spec.md](arrow_dataset_spec.md) | Required Arrow dataset layout, schemas, manifests, and validation checks for direct Rust predict and predict_incremental inputs. |
| [profiling/2026-05-28-promoted-incremental-arrow.md](profiling/2026-05-28-promoted-incremental-arrow.md) | Retained release-build comparison for the active memory-mapped name-count index. Historical evidence, not an active gate. |

---

## Key policies

- **`baselines.md` is the Rust command and baseline-evidence authority.** The
  complete v1.3 gate, execution order, thresholds, and approvals live in
  [../1_3_release_todo.md](../1_3_release_todo.md).
- **Point-in-time profiling evidence** belongs in `profiling/YYYY-MM-DD.md`, not inline in design docs.
- **Release sequence** lives in `docs/1_3_release_todo.md`; the remediation
  backlog lives in `docs/work_plan.md`.
- **Development artifacts** may live under `scratch/` (gitignored). Release
  JSON/log evidence must instead be retained under the runbook's durable
  reports root and bound by `quality_report.json` and `release.json`.

---

## Quick links

- Canonical-v2 remediation ledger: [`docs/work_plan.md`](../work_plan.md)
- v1.3 release runbook: [`docs/1_3_release_todo.md`](../1_3_release_todo.md)
- Verification commands: [`runtime.md` -- Verification Commands section](runtime.md)
- Gate commands + artifact conventions: [`baselines.md`](baselines.md)
