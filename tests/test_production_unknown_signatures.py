"""Protect the production boundary between fallback labels and inference inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.production.model.release_pairwise import _aggregate_b3, _b3_report
from scripts.production.model.unknown_signatures import (
    augmented_unknown_ids,
    exclude_unknown_pairs,
    load_unknown_signatures,
    resolve_unknown_signature_sidecar,
)


def test_unknown_sidecar_uses_label_directory_and_exact_bytes(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    sidecar = labels / "restored_unknown_signature_ids_20261009.json"
    sidecar.write_text('["u2", "u1"]\n', encoding="utf-8")
    source = load_unknown_signatures({"clusters": labels / "clusters.json", "signatures": tmp_path / "names.json"})
    assert source.ids == frozenset({"u1", "u2"})
    assert source.binding() == {
        "path": str(sidecar.resolve()),
        "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "signature_count": 2,
    }


@pytest.mark.parametrize("values", [["u", "u"], [1], [""], {"unknown": ["u"]}])
def test_unknown_sidecar_rejects_invalid_ids(tmp_path: Path, values: object) -> None:
    (tmp_path / "restored_unknown_signature_ids_20261009.json").write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar"):
        load_unknown_signatures({"clusters": tmp_path / "clusters.json"})


def test_legacy_and_ambiguous_unknown_sources(tmp_path: Path) -> None:
    files = {"signatures": tmp_path / "signatures.json"}
    assert load_unknown_signatures(files).ids == frozenset()
    for date in ("20261009", "20261010"):
        (tmp_path / f"restored_unknown_signature_ids_{date}.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Multiple"):
        load_unknown_signatures(files)


def test_explicit_sidecar_name_requires_the_discovered_file(tmp_path: Path) -> None:
    filename = "restored_unknown_signature_ids_reviewed.json"
    assert resolve_unknown_signature_sidecar(tmp_path) is None
    with pytest.raises(FileNotFoundError):
        resolve_unknown_signature_sidecar(tmp_path, filename=filename)
    other = tmp_path / "restored_unknown_signature_ids_other.json"
    other.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected unknown-signature sidecar"):
        resolve_unknown_signature_sidecar(tmp_path, filename=filename)
    assert resolve_unknown_signature_sidecar(tmp_path, filename=other.name) == other.resolve()


@pytest.mark.parametrize(
    "filename", ["restored_unknown_signature_ids_.json", "restored_unknown_signature_ids_dir.json"]
)
def test_unknown_sidecar_rejects_invalid_names_and_directories(tmp_path: Path, filename: str) -> None:
    (tmp_path / filename).mkdir()
    with pytest.raises(ValueError, match="Unknown-signature"):
        load_unknown_signatures({"clusters": tmp_path / "clusters.json"})


def test_supervised_pairs_exclude_unknown_augmented_copies_and_keep_order() -> None:
    unknown = augmented_unknown_ids({"qian": frozenset({"u"}), "augmented": frozenset({"own-u"})})
    pairs = [("qian___u", "qian___k", 1), ("b", "a", 0), ("a", "own-u", 0), ("c", "d", 1)]
    assert exclude_unknown_pairs(pairs, unknown) == [("b", "a", 0), ("c", "d", 1)]
    assert exclude_unknown_pairs(pairs, frozenset()) == pairs
    assert exclude_unknown_pairs(pairs, frozenset({left for left, _, _ in pairs})) == []


def test_b3_ignores_fallback_memberships_and_uses_known_weights() -> None:
    predicted = {"p1": ["k1", "u"], "p2": ["k2"]}
    true = {"t1": ["k1"], "t2": ["k2", "u"]}
    original = {key: list(members) for key, members in predicted.items()}
    report = _b3_report(true, predicted, frozenset({"u"}))
    assert report == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "signature_count": 2}
    assert _b3_report({"t1": ["k1", "u"], "t2": ["k2"]}, predicted, frozenset({"u"})) == report
    assert predicted == original
    aggregate = _aggregate_b3(
        {
            "first": report,
            "second": {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "signature_count": 1,
            },
        }
    )
    assert aggregate["signature_count"] == 3
    assert aggregate["signature_weighted"]["f1"] == pytest.approx(2 / 3)


def test_b3_requires_complete_inference_and_handles_all_unknown_scores() -> None:
    with pytest.raises(ValueError, match="cover all"):
        _b3_report({"t": ["known", "unknown"]}, {"p": ["known"]}, frozenset({"unknown"}))
    with pytest.raises(ValueError, match="at least one known"):
        _b3_report({"t": ["u"]}, {"p": ["u"]}, frozenset({"u"}))
    with pytest.raises(ValueError, match="positive evaluated"):
        _aggregate_b3({"all_unknown": {"signature_count": 0}})
