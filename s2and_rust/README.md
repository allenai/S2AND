# s2and-rust

Rust extension module for `s2and`.

## Published installs

Use the PyPI install path only for versions that have actually been published:

```bash
uv pip install s2and
```

`s2and-rust` is now a required dependency of `s2and`; the historical
`s2and[rust]` extra is only a compatibility alias.

When working from a checkout, use a local build so `s2and` and `s2and-rust`
come from the same tree.

## Local dev build

```bash
uvx --from maturin maturin develop -m s2and_rust/Cargo.toml
```
