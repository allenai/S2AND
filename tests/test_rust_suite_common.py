from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts._rust_suite import common


def test_get_result_markers_profile():
    start, end = common.get_result_markers("profile")
    assert start == "===S2AND_PROFILE_RESULT_START==="
    assert end == "===S2AND_PROFILE_RESULT_END==="


def test_get_result_markers_unknown_raises():
    with pytest.raises(KeyError):
        common.get_result_markers("unknown")


def test_timed_method_tracks_calls_and_restores_instance_method():
    class _Dummy:
        def add_one(self, value: int) -> int:
            return value + 1

    dummy = _Dummy()
    assert "add_one" not in dummy.__dict__

    with common.timed_method(dummy, "add_one") as stats:
        assert "add_one" in dummy.__dict__
        assert dummy.add_one(1) == 2
        assert dummy.add_one(5) == 6

    assert stats.calls == 2
    assert stats.seconds >= 0.0
    assert "add_one" not in dummy.__dict__
    assert dummy.add_one(2) == 3


def test_timed_method_restores_direct_attribute():
    def _double(value: int) -> int:
        return value * 2

    target = SimpleNamespace(run=_double)
    with common.timed_method(target, "run") as stats:
        assert target.run(3) == 6
    assert stats.calls == 1
    assert target.run is _double
