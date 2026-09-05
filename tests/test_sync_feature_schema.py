"""Verify the checked-in native schema cannot silently drift from its source."""

import pytest

from scripts import sync_feature_schema


def test_checked_in_schema_is_current():
    sync_feature_schema.main(["--check"])


def test_schema_check_rejects_stale_output_without_rewriting(tmp_path, monkeypatch):
    output = tmp_path / "feature_schema.rs"
    monkeypatch.setattr(sync_feature_schema, "RUST_PATH", output)
    sync_feature_schema.main([])
    generated = output.read_text()
    output.write_text(generated.replace("FIRST_NAMES_EQUAL: usize = 0", "FIRST_NAMES_EQUAL: usize = 1"))
    stale = output.read_text()
    with pytest.raises(SystemExit, match="schema is stale"):
        sync_feature_schema.main(["--check"])
    assert output.read_text() == stale
    sync_feature_schema.main([])
    sync_feature_schema.main(["--check"])
    assert output.read_text() == generated
