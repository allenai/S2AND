from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from s2and._sha256 import sha256_file
from s2and.arrow_inputs import ArrowDataset
from scripts.linker import build_linker_inputs
from scripts.pairwise.mine_pairwise_pairs import mine_pairs, read_fit_pool_inputs
from tests.stage0_world import build_stage0_world

pytestmark = pytest.mark.tooling


def test_fit_pool_predicts_only_non_queries_and_rejects_stale_assignments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = build_stage0_world(tmp_path)
    block = "a ross"
    output = tmp_path / "output"
    calls: dict[str, list[str]] = {}

    class FixtureModel:
        def predict_from_arrow(self, blocks: dict[str, list[str]], hidden: ArrowDataset, *, batching_threshold: int):
            assert batching_threshold == 15000
            calls.update(blocks)
            with hidden.use() as lease, lease.open_file("signatures") as source:
                table = pa.ipc.open_file(pa.PythonFile(source)).read_all()
                if "author_orcid" in table.column_names:
                    assert table["author_orcid"].null_count == len(table)
            return {"fixture": next(iter(blocks.values()))}, None

    monkeypatch.setattr(build_linker_inputs, "load_production_model", lambda *a, **kw: FixtureModel())
    args = (world.ledger, world.inputs.keys(), output, tmp_path / "model")
    counts = build_linker_inputs.build_linker_inputs(*args, queries_per_block=1)
    inputs = output / "inputs.json"
    entry = json.loads(inputs.read_text(encoding="utf-8"))[block]
    queries = set(json.loads(Path(entry["queries"]).read_text(encoding="utf-8")))
    assert len(queries) == 1
    assert calls == {
        key: sorted(
            set(world.ledger.for_block(key).rows)
            - set(json.loads((output / key.replace(" ", "_") / "queries.json").read_text()))
        )
        for key in world.inputs
    }
    partition = Path(entry["seeds"])
    rows = pd.read_parquet(partition)
    assert list(partition.parent.glob("*.parquet")) == [partition]
    calls.clear()
    assert build_linker_inputs.build_linker_inputs(*args, queries_per_block=1) == counts
    assert not calls

    output_pairs = tmp_path / "pairs.csv"
    mine_pairs(
        world.roles_path,
        read_fit_pool_inputs(world.ledger, inputs),
        output_pairs,
        identity_evidence=world.evidence_path,
    )
    pairs = pd.read_csv(output_pairs)
    assert not pairs.empty
    assert queries.isdisjoint(set(pairs["signature_id_1"]) | set(pairs["signature_id_2"]))

    pd.concat([rows, pd.DataFrame({"signature_id": sorted(queries), "component_key": "fixture"})]).to_parquet(partition)
    receipt_path = partition.with_name("manifest.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["files"][partition.name] = sha256_file(partition)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="Query signatures in assignments.*Rebuild linker inputs"):
        build_linker_inputs.build_linker_inputs(*args, queries_per_block=1)
