# Profiling Snapshots

Dated profiling snapshots live in this folder (`YYYY-MM-DD.md`). The index + rules are maintained in
`../baselines.md` under the "Profiling snapshots" section.

This folder can also hold small, deterministic repro notes for known parity gaps. Those snapshots are
not promotion gates. They exist to preserve one-command repros with the exact failure shape and code
pointers needed to debug or regression-test the issue later.

Current snapshots:

- [docs/rust/profiling/2026-03-02.md](docs/rust/profiling/2026-03-02.md): gate rerun refresh snapshot.
