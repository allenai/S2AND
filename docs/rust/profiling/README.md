# Profiling Snapshots

This folder holds **dated** profiling snapshots. Each snapshot captures one refresh
of the benchmark suite at a point in time. They are historical evidence, not active
gates.

---

## Structure

- Files are named `YYYY-MM-DD.md` by the date the refresh was run.
- Each file should include: environment, commands run, artifacts produced, and a
  short interpretation of results.
- Prefer referencing `scratch/*.json` paths rather than pasting large log output inline.
- Only snapshots explicitly promoted in `../baselines.md` become active gates.

## Snapshots

| Date | Highlights |
|---|---|
| [2026-03-02](2026-03-02.md) | Gate rerun refresh snapshot (commands + artifact paths). |


## Active baselines

The active gate baselines are in `../baselines.md`, not here. Profiling snapshots
provide context and trended evidence; `baselines.md` is the source of truth for
promotion decisions.
