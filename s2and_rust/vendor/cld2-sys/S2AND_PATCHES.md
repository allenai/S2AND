# Local CLD2 patch

Start from the crates.io cld2-sys 1.0.2 source.
Make `prior_lang`, `prior_unreliable`, and `gDisplayPiece` thread-local.
Preserve debug-function references and per-call resets.
Use compiler thread-local storage to preserve the generated tables' C++03 conversion behavior.

Use the static scoring tables and detection flags of zero through the Rust wrapper.
These settings exclude dynamic-table mutations and debug output buffers.
Keep the Rust version-string lock because the C++ version function writes a shared buffer.

Preserve the upstream source licenses in `cld2/LICENSE` and the crate metadata.
Test serial-versus-parallel language results and complete request partitions before retaining this experiment.
