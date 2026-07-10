# Rust Docs

The `s2and_rust` extension accelerates S2AND's most expensive stages (featurization,
preprocessing, constraint lookups) by replacing Python hot paths with Rust via PyO3.
This folder documents the runtime contract, benchmark evidence, and migration plans
for the Rust integration.

---

## Documents

| File | Purpose |
|---|---|
| [baselines.md](baselines.md) | Gate operator guide: canonical gate commands, artifact conventions, and promotion policy. Start here when verifying Rust behavior or promoting a new baseline. |
| [runtime.md](runtime.md) | Runtime contract: explicit Python/Rust routes, exact native-version loading, failure semantics, Arrow validation, and verification commands. |
| [inference_architecture.md](inference_architecture.md) | Current Rust inference boundary, before/after table, name-count decision, and remaining Python-heavy paths. |
| [public_surface_inventory.md](public_surface_inventory.md) | Python-visible Rust/PyO3 surface inventory used to plan API cleanup and module splitting. |
| [artifact_formats.md](artifact_formats.md) | Current artifact-format choices and rejected alternatives. |
| [arrow_dataset_spec.md](arrow_dataset_spec.md) | Required Arrow dataset layout, schemas, manifests, and validation checks for direct Rust predict and predict_incremental inputs. |
| [profiling/](profiling/) | Point-in-time profiling snapshots, named by date (indexed in `baselines.md`). Historical evidence, not active gates. |

---

## Key policies

- **`baselines.md` is the gate authority.** Any promotion decision must cite an artifact from there.
- **Point-in-time profiling evidence** belongs in `profiling/YYYY-MM-DD.md`, not inline in design docs.
- **Next steps + backlog** live in `docs/work_plan.md`.
- **Artifacts** (benchmark JSONs, logs) live under `scratch/` (gitignored).

---

## Quick links

- Rust/Arrow execution backlog: [`docs/work_plan.md`](../work_plan.md)
- Verification commands: [`runtime.md` -- Verification Commands section](runtime.md)
- Gate commands + artifact conventions: [`baselines.md`](baselines.md)
