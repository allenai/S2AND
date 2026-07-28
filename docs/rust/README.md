# Rust Docs

The `s2and_rust` extension accelerates S2AND's most expensive stages (featurization,
preprocessing, constraint lookups) by replacing Python hot paths with Rust via PyO3.
This folder documents the runtime contract, benchmark evidence, and migration plans
for the Rust integration.

---

## Documents

| File | Purpose |
|---|---|
| [baselines.md](baselines.md) | Current promoted incremental performance-report command; use the v1.3 runbook for release acceptance. |
| [runtime.md](runtime.md) | Runtime contract: explicit Python/Rust routes, exact native-version loading, failure semantics, Arrow validation, and verification commands. |
| [artifact_formats.md](artifact_formats.md) | Current artifact-format choices and rejected alternatives. |
| [arrow_dataset_spec.md](arrow_dataset_spec.md) | Required Arrow dataset layout, schemas, manifests, and validation checks for direct Rust predict and predict_incremental inputs. |
| [profiling/2026-05-28-promoted-incremental-arrow.md](profiling/2026-05-28-promoted-incremental-arrow.md) | Retained release-build comparison for the active memory-mapped name-count index. Historical evidence, not an active gate. |

---

## Key policies

- **`baselines.md` owns only the performance-report command.** The complete
  v1.3 gate, execution order, thresholds, and approvals live in
  [../release.md](../release.md).
- **Point-in-time profiling evidence** belongs in `profiling/YYYY-MM-DD.md`, not inline in design docs.
- **Release sequence and acceptance requirements** live in
  `docs/release.md`.
- **Development artifacts** may live under `scratch/` (gitignored). Release job
  logs are durable operational records. The trusted-owner runbook keeps only
  the records needed to review results and diagnose failures.

---

## Quick links

- v1.3 release runbook: [`docs/release.md`](../release.md)
- Verification commands: [`runtime.md` -- Verification Commands section](runtime.md)
- Performance-report command: [`baselines.md`](baselines.md)
