# Installation

This document covers the fuller install and setup options for S2AND.

## Requirements

- Python `3.11.x`
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Rust, if you want to build the native extension from source: [`rustup`](https://www.rust-lang.org/tools/install)

If you are building the Rust extension from source, install OS prerequisites first.

Ubuntu / Debian / WSL2:

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libgomp1
```

Windows:

- Install Visual Studio Build Tools with the `Desktop development with C++` workload.

Toolchain sanity check:

```bash
uv --version
rustc --version
cargo --version
```

## Package install

Python-only runtime:

```bash
uv pip install s2and
```

Rust-enabled runtime when wheels are available:

```bash
uv pip install "s2and[rust]"
```

## Repo checkout

Create and activate a Python 3.11 environment:

```bash
uv venv --python 3.11.13
```

Activation examples:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
. .venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat
```

Install repo dependencies:

```bash
uv sync --active --extra dev
```

If you do not want to activate the environment first, `uv sync --extra dev` also works and will use the project environment.

## Build the Rust extension from source

Install the extension into the active environment:

```bash
uv run --active --no-project maturin develop -m s2and_rust/Cargo.toml
```

Notes:

- This installs the compiled module into `site-packages`.
- If you just installed Rust with `rustup` in the current shell, load its environment first if needed.
- If you prefer a non-editable repo install, you can `uv pip install .` and then run the `maturin develop` step.

## Running repo scripts

When running scripts from a repo checkout, prefer:

```bash
uv run --no-project python path/to/script.py
```

This keeps imports pointed at the installed package and compiled extension in `site-packages`. Avoid setting `PYTHONPATH` to the repo root, which can shadow the compiled module.

## WSL notes

- Some Ubuntu images do not provide a `python` alias by default. Use `python3` for system-Python commands when needed.
- On repo paths mounted from Windows, `uv` may warn about failed hardlinks. To suppress that and avoid repeated warnings, set `UV_LINK_MODE=copy` before `uv sync` or `uv pip install`.
