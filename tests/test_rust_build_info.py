from __future__ import annotations

import pytest


def test_rust_get_build_info_contract():
    s2and_rust = pytest.importorskip("s2and_rust")

    get_build_info = getattr(s2and_rust, "get_build_info", None)
    if not callable(get_build_info):
        return

    info = get_build_info()
    assert isinstance(info, dict)
    for key in ("crate_version", "profile", "debug_assertions", "opt_level", "target"):
        assert key in info
