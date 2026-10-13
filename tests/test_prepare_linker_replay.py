"""Current replay inputs retain IDs and replace stale metadata and vectors."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from scripts.production.data.prepare_linker_replay import prepare_linker_replay


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _pickle(path: Path, keys: list[str], values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors = np.asarray([np.full(768, value) for value in values])
    with path.open("wb") as stream:
        pickle.dump((vectors, np.asarray(keys)), stream)


def _inputs(tmp_path: Path) -> dict[str, Path]:
    roots = {name: tmp_path / name for name in ("raw", "embeddings", "benchmark", "output")}
    old = {
        "2": {
            "signature_id": "2",
            "paper_id": "p2",
            "author_info": {"first": "Jae", "middle": "Hoon", "last": "Kim", "email": None},
        },
        "1": {
            "signature_id": "1",
            "paper_id": "p1",
            "author_info": {"first": "Old", "middle": "", "last": "Name", "email": None},
        },
    }
    current = {
        "1": {
            "signature_id": "1",
            "paper_id": "p1",
            "author_info": {"first": "Current", "middle": "", "last": "Name", "emails": ["a", "b"]},
        },
        "2": {
            "signature_id": "2",
            "paper_id": "p2",
            "author_info": {"first": "Jae-Hoon", "middle": "", "last": "Kim", "emails": ["a"]},
        },
    }
    old_papers = {"p1": {"title": "old title", "abstract": "same"}, "p2": {"title": "same title", "abstract": "same"}}
    new_papers = {**old_papers, "p1": {"title": "current title", "abstract": "same"}}
    _json(roots["raw"] / "arnetminer/signatures.json", old)
    _json(roots["raw"] / "arnetminer/papers.json", old_papers)
    _pickle(roots["embeddings"] / "arnetminer/specter2.pkl", ["p2", "p1"], [2, 1])
    _json(roots["benchmark"] / "arnetminer/arnetminer_signatures.json", current)
    _json(roots["benchmark"] / "arnetminer/arnetminer_papers.json", new_papers)
    _pickle(roots["benchmark"] / "arnetminer/arnetminer_specter2.pkl", ["p1", "p2"], [9, 8])
    _json(roots["raw"] / "a_khan/signatures.json", old)
    _json(roots["raw"] / "a_khan/papers.json", old_papers)
    _pickle(roots["embeddings"] / "a_khan/specter2.pkl", ["p1"], [1])
    return roots


def _run(roots: dict[str, Path], **options: int) -> dict:
    return prepare_linker_replay(
        raw_root=roots["raw"],
        embeddings_root=roots["embeddings"],
        benchmark_root=roots["benchmark"],
        output_root=roots["output"],
        datasets=["arnetminer", "a_khan"],
        **options,
    )


def test_current_raw_names_ids_and_vectors_replace_stale_sources(tmp_path: Path) -> None:
    roots = _inputs(tmp_path)
    sources = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for root in roots.values()
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    }
    result = _run(roots)
    output = roots["output"]
    signatures = json.loads((output / "raw/arnetminer/signatures.json").read_bytes())
    assert list(signatures) == ["2", "1"]
    assert signatures["2"]["author_info"]["first"] == "Jae-Hoon"
    assert signatures["2"]["author_info"]["middle"] == ""
    assert signatures["2"]["author_info"]["email"] == "a"
    assert signatures["1"]["author_info"]["email"] is None
    assert signatures["1"]["author_info"]["emails"] == ["a", "b"]
    with (output / "embeddings/arnetminer/specter2.pkl").open("rb") as stream:
        vectors, keys = pickle.load(stream)
    assert keys.tolist() == ["p2", "p1"]
    np.testing.assert_array_equal(vectors[0], np.full(768, 8))
    np.testing.assert_array_equal(vectors[1], np.full(768, 9))
    record = result["datasets"][0]
    assert record["signature_ids_and_order_equal"] and record["raw_name_parts_exact"]
    assert record["changed_title_or_abstract_records"] == 1
    assert record["numerically_changed_old_vectors"] == 2
    assert record["missing_embedding_count"] == 0
    for relative in ("raw/a_khan/signatures.json", "raw/a_khan/papers.json", "embeddings/a_khan/specter2.pkl"):
        source = roots["raw"] / relative[4:] if relative.startswith("raw/") else roots["embeddings"] / relative[11:]
        assert (output / relative).read_bytes() == source.read_bytes()
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in sources.items())


def test_missing_replay_signature_id_fails(tmp_path: Path) -> None:
    roots = _inputs(tmp_path)
    path = roots["benchmark"] / "arnetminer/arnetminer_signatures.json"
    values = json.loads(path.read_bytes())
    del values["2"]
    _json(path, values)
    with pytest.raises(ValueError, match="fixed replay signature IDs"):
        _run(roots)


def test_changed_encoder_input_cannot_reuse_stale_vector(tmp_path: Path) -> None:
    roots = _inputs(tmp_path)
    _pickle(roots["benchmark"] / "arnetminer/arnetminer_specter2.pkl", ["p2"], [8])
    with pytest.raises(ValueError, match="Changed arnetminer encoder input lacks"):
        _run(roots)


def test_unchanged_encoder_input_can_retain_old_vector(tmp_path: Path) -> None:
    roots = _inputs(tmp_path)
    _pickle(roots["benchmark"] / "arnetminer/arnetminer_specter2.pkl", ["p1"], [9])
    record = _run(roots)["datasets"][0]
    assert record["fresh_vectors"] == 1
    assert record["retained_unchanged_input_vectors"] == 1


def test_source_bound_fails_before_output_creation(tmp_path: Path) -> None:
    roots = _inputs(tmp_path)
    with pytest.raises(ValueError, match="exceed max_source_bytes"):
        _run(roots, max_source_bytes=1)
    assert not roots["output"].exists()
