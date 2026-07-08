from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (
    REPO_ROOT / "docs",
    REPO_ROOT / "s2and",
    REPO_ROOT / "s2and_rust" / "src",
    REPO_ROOT / "scripts",
    REPO_ROOT / "tests",
)
SKIP_DIR_NAMES = {".git", ".venv", "data", "data-backup", "dist", "scratch", "target", "__pycache__"}
TEXT_SUFFIXES = {".py", ".rs", ".md"}


def _iter_surface_files() -> list[Path]:
    files: list[Path] = []
    this_file = Path(__file__).resolve()
    for root in SCAN_ROOTS:
        for path in root.rglob("*"):
            if any(part in SKIP_DIR_NAMES for part in path.parts):
                continue
            if path.resolve() == this_file:
                continue
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                files.append(path)
    return files


def test_removed_rust_escape_hatches_do_not_reappear() -> None:
    forbidden_literals = (
        "full_scan" + "_without_index",
        "get_constraint" + "_rust",
    )
    forbidden_patterns = (
        re.compile(r"\bfn\s+get_constraint\s*\("),
        re.compile(r"\bdef\s+" + "get_constraint" + r"_rust\s*\("),
    )
    offenders: list[str] = []
    for path in _iter_surface_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        relative = path.relative_to(REPO_ROOT)
        for literal in forbidden_literals:
            if literal in text:
                offenders.append(f"{relative}: contains {literal}")
        for pattern in forbidden_patterns:
            if pattern.search(text):
                offenders.append(f"{relative}: matches {pattern.pattern}")

    assert offenders == []


def test_raw_arrow_runtime_does_not_accept_query_compatibility_args() -> None:
    from s2and.incremental_linking import runtime as runtime_module

    parameters = inspect.signature(
        runtime_module.predict_incremental_link_or_abstain_from_raw_arrow_paths
    ).parameters

    assert "query_signature_ids" not in parameters
    assert "query_view" not in parameters


def test_raw_planner_direct_constructor_has_no_call_sites() -> None:
    call_pattern = re.compile(r"\bRawBlockQueryCandidatePlanner\s*\(")
    offenders: list[str] = []
    for path in _iter_surface_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if call_pattern.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []


def test_raw_planner_has_no_python_direct_constructor() -> None:
    s2and_rust = pytest.importorskip("s2and_rust", reason="s2and_rust is unavailable")
    raw_planner_cls = getattr(s2and_rust, "RawBlock" + "QueryCandidatePlanner")

    with pytest.raises(TypeError, match="No constructor defined"):
        raw_planner_cls({}, [], top_k=1)
