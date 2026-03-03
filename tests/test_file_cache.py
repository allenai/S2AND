import json
import logging
from pathlib import Path

import s2and.file_cache as file_cache


class _HeadResponse:
    def __init__(self, status_code: int = 200, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self.headers = headers or {}


def test_get_from_cache_download_does_not_write_stdout(monkeypatch, tmp_path, caplog, capsys):
    url = "https://example.org/model.bin"
    etag = "etag-123"
    expected_bytes = b"cached model payload"

    def _fake_head(request_url: str, allow_redirects: bool = True):
        assert request_url == url
        assert allow_redirects is True
        return _HeadResponse(status_code=200, headers={"ETag": etag})

    def _fake_http_get(request_url: str, temp_file):
        assert request_url == url
        temp_file.write(expected_bytes)

    monkeypatch.setattr(file_cache.requests, "head", _fake_head)
    monkeypatch.setattr(file_cache, "http_get", _fake_http_get)

    with caplog.at_level(logging.INFO, logger="s2and"):
        cache_path = file_cache.get_from_cache(url, str(tmp_path))

    captured = capsys.readouterr()
    assert captured.out == ""

    assert Path(cache_path).read_bytes() == expected_bytes
    metadata = json.loads(Path(cache_path + ".json").read_text(encoding="utf-8"))
    assert metadata == {"url": url, "etag": etag}

    logs = "\n".join(caplog.messages)
    assert "not found in cache; downloading" in logs
    assert "Finished download; copying" in logs
