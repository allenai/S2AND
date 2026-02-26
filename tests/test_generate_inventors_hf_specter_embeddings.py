import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "generate_inventors_hf_specter_embeddings.py"
    spec = importlib.util.spec_from_file_location("generate_inventors_hf_specter_embeddings", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as outfile:
        json.dump(payload, outfile)


def test_load_signature_paper_ids_and_filter_records(tmp_path):
    module = _load_module()
    signatures_path = tmp_path / "signatures.json"
    papers_path = tmp_path / "papers.json"

    _write_json(
        signatures_path,
        {
            "s1": {"paper_id": "10"},
            "s2": {"paper_id": 20},
            "s3": {"paper_id": "10"},
        },
    )
    _write_json(
        papers_path,
        {
            "10": {"title": "t1", "abstract": "a1"},
            "20": {"title": "t2", "abstract": "a2"},
            "30": {"title": "t3", "abstract": "a3"},
        },
    )

    required = module.load_signature_paper_ids(signatures_path)
    records, missing = module.load_paper_records(papers_path, limit=None, required_paper_ids=required)

    assert required == {"10", "20"}
    assert [row.paper_id for row in records] == ["10", "20"]
    assert missing == set()


def test_load_paper_records_reports_missing_required_ids(tmp_path):
    module = _load_module()
    papers_path = tmp_path / "papers.json"

    _write_json(
        papers_path,
        {
            "10": {"title": "t1", "abstract": "a1"},
        },
    )

    records, missing = module.load_paper_records(
        papers_path,
        limit=None,
        required_paper_ids={"10", "999"},
    )

    assert [row.paper_id for row in records] == ["10"]
    assert missing == {"999"}


def test_parse_args_defaults_use_prefixed_filenames(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(sys, "argv", ["generate_inventors_hf_specter_embeddings.py"])
    args = module.parse_args()

    assert args.signatures_path == Path("data/inventors_s2and/inventors_s2and_signatures.json")
    assert args.papers_path == Path("data/inventors_s2and/inventors_s2and_papers.json")
    assert args.output_specter_path == Path("data/inventors_s2and/inventors_s2and_specter.pickle")
    assert args.output_specter2_path == Path("data/inventors_s2and/inventors_s2and_specter2.pkl")
