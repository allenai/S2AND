import importlib.util
import json
import sys
import types
from pathlib import Path


def _load_subset_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "make_inventors_s2and_subset.py"
    spec = importlib.util.spec_from_file_location("make_inventors_s2and_subset", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_ijson(monkeypatch) -> None:
    def _load_payload(file_obj) -> dict:
        raw = file_obj.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    fake_ijson = types.SimpleNamespace()

    def parse(file_obj):
        payload = _load_payload(file_obj)
        for signature in payload.values():
            block = signature.get("author_info", {}).get("block")
            if isinstance(block, str):
                yield ("item.author_info.block", "string", block)

    def kvitems(file_obj, _prefix):
        payload = _load_payload(file_obj)
        yield from payload.items()

    fake_ijson.parse = parse
    fake_ijson.kvitems = kvitems
    monkeypatch.setitem(sys.modules, "ijson", fake_ijson)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as outfile:
        json.dump(payload, outfile)


def _build_tiny_inventors_fixture(input_dir: Path) -> None:
    signatures = {}
    clusters = {}
    papers = {}
    blocks = ["alpha", "beta", "gamma", "delta"]
    signature_index = 0

    for block in blocks:
        signature_ids = []
        for _ in range(2):
            signature_index += 1
            signature_id = f"s{signature_index}"
            paper_id = f"p{signature_index}"
            signatures[signature_id] = {
                "paper_id": paper_id,
                "author_info": {
                    "block": block,
                    "given_block": block,
                },
            }
            papers[paper_id] = {"title": f"title-{paper_id}", "abstract": f"abstract-{paper_id}"}
            signature_ids.append(signature_id)
        clusters[f"c_{block}"] = {"signature_ids": signature_ids}

    _write_json(input_dir / "signatures.json", signatures)
    _write_json(input_dir / "clusters.json", clusters)
    _write_json(input_dir / "papers.json", papers)


def test_subset_script_runs_without_specter_files(tmp_path, monkeypatch):
    input_dir = tmp_path / "inventors"
    output_dir = tmp_path / "inventors_s2and"
    input_dir.mkdir(parents=True, exist_ok=True)
    _build_tiny_inventors_fixture(input_dir)

    _install_fake_ijson(monkeypatch)
    module = _load_subset_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_inventors_s2and_subset.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--n-blocks",
            "2",
            "--val-ratio",
            "0.5",
            "--seed",
            "7",
            "--full-run",
        ],
    )
    module.main()

    output_prefix = output_dir.name
    expected_outputs = [
        f"{output_prefix}_clusters.json",
        f"{output_prefix}_signatures.json",
        f"{output_prefix}_papers.json",
        f"{output_prefix}_train_keys.json",
        f"{output_prefix}_val_keys.json",
        f"{output_prefix}_subset_summary.json",
    ]
    for filename in expected_outputs:
        assert (output_dir / filename).exists()
    assert not (output_dir / "specter.pickle").exists()

    with (output_dir / f"{output_prefix}_subset_summary.json").open("r", encoding="utf-8") as infile:
        summary = json.load(infile)
    assert "specter" not in summary["outputs"]
    assert "specter_embeddings_written" not in summary["counts"]
    assert summary["config"]["output_prefix"] == output_prefix
    assert summary["counts"]["kept_papers"] > 0
