from __future__ import annotations

from types import SimpleNamespace

from s2and import model


def test_path_cache_fingerprint_omits_ctime(monkeypatch) -> None:
    monkeypatch.setattr(
        model.Path,
        "stat",
        lambda _path: SimpleNamespace(st_size=7, st_mtime_ns=11, st_ctime_ns=13),
    )
    monkeypatch.setattr(model, "_path_full_digest", lambda _path: "digest")

    path = model.Path("artifact.arrow")
    assert model._path_cache_fingerprint(path) == (str(path.resolve()), 7, 11, "digest")
