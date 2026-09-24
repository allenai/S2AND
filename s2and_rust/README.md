# s2and-rust

Rust extension module for `s2and`.

## Published installs

Use the PyPI install path only for versions that have actually been published:

```bash
uv pip install s2and
```

`s2and-rust` is now a required dependency of `s2and`; the historical
`s2and[rust]` extra no longer exists. The unreleased `1.0.0` worktree
currently pins `s2and-rust==1.0.0`; use a local same-checkout build until the
coordinated Python/Rust release is published. At runtime the two installed
package versions must match exactly.

When working from a checkout, use a local build so `s2and` and `s2and-rust`
come from the same tree.

## Local dev build

```bash
uv sync --active --extra dev
```

`uv sync` builds and installs `s2and-rust` from this directory, and `uv run`
uses timestamps to detect ordinary build-input edits. The test guard checks
source contents, including additions and deletions that uv's timestamp keys
can miss. See [build recovery](../docs/install.md#build-the-rust-extension-from-source)
for forced reinstall and Cargo cache-cleaning commands.
