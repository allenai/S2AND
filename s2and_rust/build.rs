use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

/// Native build inputs covered by the embedded source digest.
///
/// `tests/rust_build_freshness.py` mirrors these lists and the hashing below.
const SOURCE_DIGEST_FILES: [&str; 4] = [
    "Cargo.toml",
    "Cargo.lock",
    "build.rs",
    "vendor/cld2/Cargo.toml",
];
const SOURCE_DIGEST_DIRS: [&str; 2] = ["src", "vendor/cld2/src"];

fn main() {
    emit_source_digest();
    #[cfg(windows)]
    stage_python_runtime_dlls();
}

/// Embed a digest of every native build input so tests can reject stale builds.
///
/// Emitting any `rerun-if-changed` limits Cargo's change detection to the
/// listed paths, so the manifests and both Rust source trees are listed on
/// every platform. Directory watches include new modules; the digest includes
/// only `.rs` files, not editor files or a vendored crate's build outputs.
/// Cargo's timestamp checks are a heuristic, not content-based invalidation.
fn emit_source_digest() {
    // A shared target directory can reuse a compiled build script across
    // source copies. Read Cargo's current package root when the script runs.
    let root = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets the package manifest directory"),
    );
    let mut files: Vec<(String, PathBuf)> = Vec::new();
    for name in SOURCE_DIGEST_FILES {
        println!("cargo:rerun-if-changed={name}");
        files.push((name.to_string(), root.join(name)));
    }
    for name in SOURCE_DIGEST_DIRS {
        println!("cargo:rerun-if-changed={name}");
        collect_files(&root, &root.join(name), &mut files);
    }
    files.sort_by(|left, right| left.0.cmp(&right.0));
    let mut hasher = Sha256::new();
    for (relative, path) in &files {
        let content = fs::read(path)
            .unwrap_or_else(|err| panic!("cannot read build input {}: {err}", path.display()));
        hasher.update(relative.as_bytes());
        hasher.update([0u8]);
        hasher.update((content.len() as u64).to_le_bytes());
        hasher.update(&content);
    }
    println!(
        "cargo:rustc-env=S2AND_RUST_SOURCE_DIGEST=v1:{:x}",
        hasher.finalize()
    );
}

fn collect_files(root: &Path, dir: &Path, files: &mut Vec<(String, PathBuf)>) {
    let metadata = fs::symlink_metadata(dir).unwrap_or_else(|err| {
        panic!(
            "cannot inspect build input directory {}: {err}",
            dir.display()
        )
    });
    assert!(
        !is_linked_directory(&metadata),
        "linked native source directories are unsupported: {}",
        dir.display()
    );
    let entries = fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("cannot list build inputs in {}: {err}", dir.display()));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|err| panic!("cannot list build inputs in {}: {err}", dir.display()))
            .path();
        if path.is_dir() {
            collect_files(root, &path, files);
        } else if path.is_file() && path.extension().is_some_and(|extension| extension == "rs") {
            let relative = path
                .strip_prefix(root)
                .expect("build input is under the manifest directory")
                .components()
                .map(|component| {
                    component.as_os_str().to_str().unwrap_or_else(|| {
                        panic!("native source paths must be valid UTF-8: {path:?}")
                    })
                })
                .collect::<Vec<_>>()
                .join("/");
            files.push((relative, path));
        }
    }
}

fn is_linked_directory(metadata: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        // Include junctions, which Python 3.11 also identifies via this flag.
        metadata.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    {
        metadata.file_type().is_symlink()
    }
}

#[cfg(windows)]
fn stage_python_runtime_dlls() {
    use std::env;

    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");
    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");

    let Some(python_exe) = python_executable() else {
        println!(
            "cargo:warning=Unable to locate Python executable for Windows cargo-test DLL staging"
        );
        return;
    };
    rerun_if_changed(&python_exe);
    if let Ok(Some(python_home)) = python_home(&python_exe) {
        println!(
            "cargo:rustc-env=S2AND_RUST_PYTHONHOME={}",
            python_home.display()
        );
    }
    let Ok(runtime_dlls) = python_runtime_dlls(&python_exe) else {
        println!(
            "cargo:warning=Unable to query Python runtime DLLs from {}",
            python_exe.display()
        );
        return;
    };
    if runtime_dlls.is_empty() {
        println!(
            "cargo:warning=Python runtime DLL query returned no files for {}",
            python_exe.display()
        );
        return;
    }

    let Ok(out_dir) = env::var("OUT_DIR") else {
        println!("cargo:warning=OUT_DIR is unavailable; cannot stage Python runtime DLLs");
        return;
    };
    let Some(profile_dir) = PathBuf::from(out_dir)
        .ancestors()
        .nth(3)
        .map(Path::to_path_buf)
    else {
        println!("cargo:warning=Unable to resolve Cargo profile directory from OUT_DIR");
        return;
    };
    let deps_dir = profile_dir.join("deps");
    if let Err(err) = fs::create_dir_all(&deps_dir) {
        println!(
            "cargo:warning=Unable to create Cargo deps directory {}: {err}",
            deps_dir.display()
        );
        return;
    }

    for source in runtime_dlls {
        rerun_if_changed(&source);
        let Some(file_name) = source.file_name() else {
            continue;
        };
        let destination = deps_dir.join(file_name);
        if let Err(err) = fs::copy(&source, &destination) {
            println!(
                "cargo:warning=Unable to stage Python runtime DLL {} to {}: {err}",
                source.display(),
                destination.display()
            );
        }
    }
}

#[cfg(windows)]
fn rerun_if_changed(path: &std::path::Path) {
    let normalized = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    println!("cargo:rerun-if-changed={}", normalized.display());
}

#[cfg(windows)]
fn python_executable() -> Option<std::path::PathBuf> {
    use std::env;
    use std::path::PathBuf;

    for key in ["PYO3_PYTHON", "PYTHON_SYS_EXECUTABLE"] {
        if let Ok(value) = env::var(key) {
            let candidate = PathBuf::from(value);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    if let Ok(virtual_env) = env::var("VIRTUAL_ENV") {
        let candidate = PathBuf::from(virtual_env)
            .join("Scripts")
            .join("python.exe");
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    let repo_venv_python = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|repo_root| repo_root.join(".venv").join("Scripts").join("python.exe"));
    if let Some(candidate) = repo_venv_python {
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    None
}

#[cfg(windows)]
fn python_home(python_exe: &std::path::Path) -> std::io::Result<Option<std::path::PathBuf>> {
    use std::path::PathBuf;
    use std::process::Command;

    let output = Command::new(python_exe)
        .arg("-c")
        .arg("import sys; print(sys.base_prefix)")
        .output()?;
    if !output.status.success() {
        return Ok(None);
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if value.is_empty() {
        return Ok(None);
    }
    let path = PathBuf::from(value);
    Ok(path.is_dir().then_some(path))
}

#[cfg(windows)]
fn python_runtime_dlls(python_exe: &std::path::Path) -> std::io::Result<Vec<std::path::PathBuf>> {
    use std::path::PathBuf;
    use std::process::Command;

    let script = r#"
import pathlib
import sys

root = pathlib.Path(sys.base_prefix)
names = [
    f"python{sys.version_info.major}{sys.version_info.minor}.dll",
    "python3.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
]
print("\n".join(str(root / name) for name in names if (root / name).exists()))
"#;
    let output = Command::new(python_exe).arg("-c").arg(script).output()?;
    if !output.status.success() {
        return Ok(Vec::new());
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .collect())
}
