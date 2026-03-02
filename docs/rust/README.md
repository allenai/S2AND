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
| [runtime.md](runtime.md) | Runtime contract: backend resolution (`python`/`rust`/`auto`), stage defaults, failure semantics, verification gates, risk register, and key capability gates (incl. training-mode deferred paper preprocessing). |
| [artifact_divergence.md](artifact_divergence.md) | Artifact format divergence map between Python and Rust paths. Format migration plan (MessagePack, Safetensors) and deferred unification backlog. |
| [profiling/](profiling/) | Point-in-time profiling snapshots, named by date. Historical evidence, not active gates. |

---

## Key policies

- **`baselines.md` is the gate authority.** Any promotion decision must cite an artifact from there.
- **Point-in-time profiling evidence** belongs in `profiling/YYYY-MM-DD.md`, not inline in design docs.
- **Next steps + backlog** live in `docs/work_plan.md` (see `Backlog` section for long-term ideas/refactors).
- **Artifacts** (benchmark JSONs, logs) live under `scratch/` (gitignored).

---

## Quick links

- Next steps: [`docs/work_plan.md`](../work_plan.md)
- Backlog: [`docs/work_plan.md#backlog`](../work_plan.md#backlog)
- Verification commands: [`runtime.md` -- Verification Commands section](runtime.md)
- Gate commands + artifact conventions: [`baselines.md`](baselines.md)
