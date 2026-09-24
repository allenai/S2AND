# s2and-rust

Rust extension module for `s2and`.

## Published installs

Install `s2and`; it depends on the exactly matching `s2and-rust` version:

```bash
uv pip install s2and
```

`s2and-rust` is a required dependency of `s2and`; the historical
`s2and[rust]` extra no longer exists. At runtime the two installed package
versions must match exactly.

> **Update at v1.3 release:** this README is the `s2and-rust` PyPI description.
> Until the coordinated `1.0.0` packages are published, the worktree pins
> `s2and-rust==1.0.0` and PyPI serves only earlier versions, so use a local
> same-checkout build instead.

When working from a checkout, use a local build so `s2and` and `s2and-rust`
come from the same tree.

## Local dev build

From the repository root (not this directory):

```bash
uv sync --extra dev
```

`uv sync` builds and installs `s2and-rust` from `s2and_rust/`, and `uv run`
uses timestamps to detect ordinary build-input edits. The test guard checks
the Rust build inputs by content, including additions and deletions that uv's
timestamp keys can miss. See
[build recovery](https://github.com/allenai/S2AND/blob/main/docs/install.md#build-the-rust-extension-from-source)
for forced reinstall and Cargo cache-cleaning commands.
