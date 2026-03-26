from __future__ import annotations

import json
from pathlib import Path

import pytest

import s2and.data as data_module
from s2and import feature_port
from s2and.data import ANDData
from s2and.feature_port import get_constraint_rust

if not feature_port.rust_featurizer_available():
    raise pytest.skip.Exception("s2and_rust extension is unavailable", allow_module_level=True)

_RUST_FEATURIZER = getattr(feature_port, "s2and_rust", None)
if _RUST_FEATURIZER is None or not hasattr(_RUST_FEATURIZER.RustFeaturizer, "from_json_paths"):
    raise pytest.skip.Exception("s2and_rust RustFeaturizer.from_json_paths is unavailable", allow_module_level=True)
assert _RUST_FEATURIZER is not None


@pytest.fixture(autouse=True)
def _force_rust_backend(monkeypatch):
    monkeypatch.setenv("S2AND_BACKEND", "rust")
    monkeypatch.setenv("S2AND_SKIP_FASTTEXT", "1")
    feature_port.clear_rust_featurizer_cache()
    yield
    feature_port.clear_rust_featurizer_cache()


def _write_minimal_fixture(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    signatures = {
        "s1": {
            "signature_id": "s1",
            "paper_id": "p1",
            "author_info": {
                "first": "Alex",
                "middle": "",
                "last": "Wang",
                "suffix": "",
                "affiliations": [],
                "email": None,
                "position": 0,
                "block": "a wang",
                "source_ids": [],
            },
        },
        "s2": {
            "signature_id": "s2",
            "paper_id": "p2",
            "author_info": {
                "first": "Bo",
                "middle": "",
                "last": "Wang",
                "suffix": "",
                "affiliations": [],
                "email": None,
                "position": 0,
                "block": "b wang",
                "source_ids": [],
            },
        },
        "s3": {
            "signature_id": "s3",
            "paper_id": "p3",
            "author_info": {
                "first": "Casey",
                "middle": "",
                "last": "Li",
                "suffix": "",
                "affiliations": [],
                "email": None,
                "position": 0,
                "block": "c li",
                "source_ids": [],
            },
        },
    }
    papers = {
        "p1": {
            "paper_id": "p1",
            "title": "Paper One",
            "abstract": "",
            "authors": [{"author_name": "Alex Wang", "position": 0}],
            "venue": "",
            "journal_name": "",
            "year": 2020,
            "references": [],
        },
        "p2": {
            "paper_id": "p2",
            "title": "Paper Two",
            "abstract": "",
            "authors": [{"author_name": "Bo Wang", "position": 0}],
            "venue": "",
            "journal_name": "",
            "year": 2021,
            "references": [],
        },
        "p3": {
            "paper_id": "p3",
            "title": "Paper Three",
            "abstract": "",
            "authors": [{"author_name": "Casey Li", "position": 0}],
            "venue": "",
            "journal_name": "",
            "year": 2022,
            "references": [],
        },
        "p_extra": {
            "paper_id": "p_extra",
            "title": "Unused Paper",
            "abstract": "",
            "authors": [{"author_name": "Unused Author", "position": 0}],
            "venue": "",
            "journal_name": "",
            "year": 2023,
            "references": [],
        },
    }

    signatures_path = tmp_path / "signatures.json"
    papers_path = tmp_path / "papers.json"
    signatures_path.write_text(json.dumps(signatures), encoding="utf-8")
    papers_path.write_text(json.dumps(papers), encoding="utf-8")
    return signatures_path, papers_path


def _build_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_sinonym_overwrite: bool,
) -> ANDData:
    signatures_path, papers_path = _write_minimal_fixture(tmp_path)
    if use_sinonym_overwrite:
        fake_results = {
            "p2": {
                0: {
                    "given_tokens": ["Alex"],
                    "surname_tokens": ["Wang"],
                    "original_compound_surname": None,
                    "middle_tokens": [],
                }
            }
        }

        def _fake_sinonym_preprocess(_papers_dict, _n_jobs):
            return fake_results

        monkeypatch.setattr(data_module, "sinonym_preprocess_papers_parallel", _fake_sinonym_preprocess)

    return ANDData(
        signatures=str(signatures_path),
        papers=str(papers_path),
        name=f"rust_sinonym_json_ingest_{tmp_path.name}_{int(use_sinonym_overwrite)}",
        mode="inference",
        specter_embeddings=None,
        clusters=None,
        cluster_seeds=None,
        block_type="s2",
        train_pairs=None,
        val_pairs=None,
        test_pairs=None,
        train_pairs_size=10,
        val_pairs_size=10,
        test_pairs_size=10,
        n_jobs=1,
        load_name_counts=False,
        preprocess=True,
        random_seed=42,
        name_tuples=set(),
        use_orcid_id=True,
        use_sinonym_overwrite=use_sinonym_overwrite,
        sinonym_overwrite_min_ratio=None,
        compute_reference_features=False,
    )


def test_inference_sinonym_overwrite_materializes_post_sinonym_json_payload(tmp_path, monkeypatch):
    dataset = _build_dataset(tmp_path, monkeypatch, use_sinonym_overwrite=True)

    assert dataset.signatures_path is not None
    assert dataset.papers_path is not None
    assert Path(dataset.signatures_path).name == "signatures_filtered.json"
    assert Path(dataset.papers_path).name == "papers_filtered.json"

    written_signatures = json.loads(Path(dataset.signatures_path).read_text(encoding="utf-8"))
    written_papers = json.loads(Path(dataset.papers_path).read_text(encoding="utf-8"))

    assert written_signatures["s2"]["author_info"]["first"] == "Alex"
    assert written_signatures["s2"]["author_info"]["last"] == "Wang"
    assert written_signatures["s2"]["author_info"]["block"] == "a wang"
    assert written_papers["p2"]["authors"][0]["author_name"] == "Alex Wang"
    assert "p_extra" not in written_papers


def test_inference_sinonym_overwrite_constraint_parity_python_vs_rust(tmp_path, monkeypatch):
    baseline_dataset = _build_dataset(tmp_path / "baseline", monkeypatch, use_sinonym_overwrite=False)
    baseline_constraint = baseline_dataset.get_constraint("s1", "s2")
    assert baseline_constraint == 10000.0

    dataset = _build_dataset(tmp_path / "sinonym", monkeypatch, use_sinonym_overwrite=True)
    flip_pair = ("s1", "s2")
    stable_pairs = [("s1", "s3"), ("s2", "s3")]

    python_flip = dataset.get_constraint(*flip_pair)
    rust_flip = get_constraint_rust(dataset, *flip_pair)
    assert python_flip is None
    assert rust_flip is None
    assert rust_flip != 10000.0

    for left, right in stable_pairs:
        python_constraint = dataset.get_constraint(left, right)
        rust_constraint = get_constraint_rust(dataset, left, right)
        assert python_constraint == rust_constraint
