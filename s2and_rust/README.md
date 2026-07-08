# s2and-rust

Rust extension module for `s2and`.

## Published installs

Use the PyPI install path only for versions that have actually been published:

```bash
uv pip install "s2and[rust]"
```

As of 2026-05-23, PyPI latest for both `s2and` and `s2and-rust` is `0.49.0`.
This checkout is `0.60.0`, so use a local build when working from this tree
until the matching packages are published.

## Local dev build

```bash
uvx --from maturin maturin develop -m s2and_rust/Cargo.toml
```
