from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from s2and.incremental_linking_training import data_loading


def test_load_giant_block_dataset_restores_runtime_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    signatures = {
        "s1": {
            "signature_id": "s1",
            "paper_id": "p1",
            "author_info": {
                "block": "a smith",
                "first": "Alice",
                "middle": "",
                "last": "Smith",
                "suffix": "",
                "position": 0,
                "affiliations": [],
            },
        }
    }
    papers = {
        "p1": {
            "paper_id": "p1",
            "title": "One",
            "abstract": "",
            "venue": "",
            "journal_name": "",
            "year": 2020,
            "authors": [{"position": 0, "author_name": "Alice Smith"}],
        }
    }
    (tmp_path / "signatures.json").write_text(json.dumps(signatures), encoding="utf-8")
    (tmp_path / "papers.json").write_text(json.dumps(papers), encoding="utf-8")
    (tmp_path / "cluster_seeds.json").write_text("{}", encoding="utf-8")
    (tmp_path / "altered_cluster_signatures.txt").write_text("", encoding="utf-8")
    with (tmp_path / "specter.pickle").open("wb") as outfile:
        pickle.dump({"p1": np.asarray([1.0, 0.0], dtype=np.float32)}, outfile)

    captured: dict[str, Any] = {}

    def fake_anddata(**kwargs: Any) -> SimpleNamespace:
        captured["backend_during_constructor"] = os.environ["S2AND_BACKEND"]
        captured["omp_threads_during_constructor"] = os.environ["OMP_NUM_THREADS"]
        captured["rayon_threads_during_constructor"] = os.environ["RAYON_NUM_THREADS"]
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setenv("S2AND_BACKEND", "python")
    monkeypatch.setenv("OMP_NUM_THREADS", "9")
    monkeypatch.setenv("RAYON_NUM_THREADS", "11")
    monkeypatch.setattr(data_loading, "ANDData", fake_anddata)

    dataset, load_info = data_loading.load_giant_block_dataset(tmp_path, block_key=None, n_jobs=1)

    assert os.environ["S2AND_BACKEND"] == "python"
    assert os.environ["OMP_NUM_THREADS"] == "9"
    assert os.environ["RAYON_NUM_THREADS"] == "11"
    assert captured["backend_during_constructor"] == "rust"
    assert captured["omp_threads_during_constructor"] == "1"
    assert captured["rayon_threads_during_constructor"] == "1"
    assert dataset is not None
    assert captured["specter_embeddings"]["p1"].shape == (2,)
    assert load_info["selected_signature_ids"] == ["s1"]
