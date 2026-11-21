# s2and-rust

Rust extension module for `s2and`.

## Distribution

The root `s2and` wheel includes `s2and._s2and_rust`.
This directory holds the Rust crate, not a separate Python distribution.
Build and install from the repository root with `uv sync --extra dev`.
Build release wheels and the source archive with `uv build` from the root.

## Local dev build

From the repository root (not this directory):

```bash
uv sync --extra dev
```

`uv sync` builds the unified package with the crate in `s2and_rust/`, and `uv run`
uses timestamps to detect ordinary build-input edits. The test guard checks
the Rust build inputs by content, including additions and deletions that uv's
timestamp keys can miss. See
[build recovery](https://github.com/allenai/S2AND/blob/main/docs/install.md#build-the-rust-extension-from-source)
for forced reinstall and Cargo cache-cleaning commands.
