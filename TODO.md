# TODO

## PR 80 audit fixes

- [x] Preserve explicit seed disallows during Arrow subblocking and directional supervision during planning.
- [x] Isolate retained Windows reader cursors without copying Arrow data.
- [x] Remove target-label leakage from linker features; preserve actual input seed constraints.
- [x] Persist and verify cluster holdout identities across JSON and Arrow ordering.
- [x] Verify prediction parity, bounded memory, and full local CI.

Full local CI passed on Windows/Python 3.11 with the rebuilt native extension:
1429 Python tests, 120 Rust tests, 88.56% coverage; lint, formatting, Clippy,
type checks, and native smoke passed. Command:
`uv run --no-project python -u scripts/run_ci_locally.py`.
Run record and logs: `scratch/pr80_fixes/ci-1/`.

All 53 linker features match runtime across the tested clean/conflicting/seeded
cases and are invariant to target-label changes. Independent seed-restoration
comparison matched the previous algorithm on 2500 randomized cases.
On the bounded 100000-signature fixture, seed-restoration peak traced Python
allocation fell from 8196498 to 5767832 bytes; pair-supervision allocation was
unchanged. The new holdout fingerprint used 88152 temporary Python bytes.
Profile evidence: `scratch/pr80_fixes/*profile*`.
Production-scale accuracy and memory remain release gates requiring the missing
canonical data exports and v1.3 artifacts; label-assisted metrics must be regenerated.

## Shared incremental completion

- [x] Verify completion invariants, integration, and full local CI.

Full local CI passed: 1134 Python tests, 119 Rust tests, 87.40% coverage.
Direct component tests cover grouping, IDs, telemetry, input isolation, and
failure propagation; existing Python/Arrow integration tests pass through the
shared component. Log: `scratch/incremental-completion-local-ci.log`.
