# Rust docs

Entry points:

- Current work plan (bundles): `docs/work_plan.md`
- Runtime contract + controls: `docs/rust/runtime.md`
- Baselines + promotion rules: `docs/rust/baselines.md`
- Optimization frontier / worklist: `docs/rust/roadmap.md`
- Artifact divergence + format migration plan: `docs/rust/artifact_divergence.md`
- Profiling snapshots (dated): `docs/rust/profiling/`

Policy:

- Treat `docs/rust/baselines.md` as the source of truth for “what is gated”.
- Put point-in-time profiling evidence in `docs/rust/profiling/YYYY-MM-DD.md`.
- Keep the roadmap doc as the living worklist; avoid mixing it with baseline evidence.

Latest local verification snapshot (2026-02-28):
- Bundle 1–4 execution summary + artifacts: `docs/work_plan.md`
- Runtime policy + latest verification-grade metrics: `docs/rust/runtime.md`
