from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from s2and.arrow_tables import write_cluster_seeds_arrow
from scripts.intake.assign_block_roles import BlockSpec, assign_block_roles
from tests.test_build_linker_task_tables import _ORCIDS, _OUTSIDE_HOME, _block_inputs

pytestmark = pytest.mark.tooling


def test_real_gate_queries_are_excluded_from_orcid_hidden_seed_prediction_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.linker import build_linker_inputs
    from scripts.linker.build_linker_task_tables import _read_signatures

    block = "a ross"
    spec = _block_inputs(tmp_path / "block", block, _ORCIDS[0])
    arrow_dir = Path(spec["arrow_dir"])
    roles = tmp_path / "roles.json"
    roles.write_text(
        json.dumps(
            assign_block_roles([BlockSpec(block, 32, frozenset({_ORCIDS[0], _OUTSIDE_HOME}))], {"gate_thresh": 1})
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "a_ross.arrow.tar.gz"
    request_seed_path = tmp_path / "cluster_seeds.arrow"
    queries = {"q_outside_home", "s04"}
    remaining = {"q_full", "q_initial", *(f"s{i:02}" for i in range(29))} - queries
    write_cluster_seeds_arrow(request_seed_path, dict.fromkeys(sorted(remaining), "seed"))
    with tarfile.open(bundle, "w:gz") as archive:
        archive.add(arrow_dir / "signatures.arrow", arcname="signatures.arrow")
        archive.add(request_seed_path, arcname="cluster_seeds.arrow")
    paths = tmp_path / "blocks.json"
    paths.write_text(json.dumps({block: {**spec, "incremental_request": bundle.name}}), encoding="utf-8")
    model_path, output = tmp_path / "model", tmp_path / "output"
    model_path.write_text("model", encoding="utf-8")

    class FixtureModel:
        def predict_from_arrow(self, blocks, hidden, *, batching_threshold):
            assert batching_threshold == 15000
            assert blocks == {block: sorted(remaining)}
            assert all(orcid is None for _, orcid in _read_signatures(hidden, block).values())
            return {"fixture": blocks[block]}, None

    monkeypatch.setattr(build_linker_inputs, "load_production_model", lambda *a, **kw: FixtureModel())
    counts = build_linker_inputs.build_linker_inputs(roles, paths, output, model_path, queries_per_block=1)
    assert counts[block] == {"queries": 2, "assigned_signatures": 30, "components": 1}
    entry = json.loads((output / "inputs.json").read_text(encoding="utf-8"))[block]
    assert set(json.loads(Path(entry["queries"]).read_text(encoding="utf-8"))) == queries
    assert set(pd.read_parquet(entry["seeds"])["signature_id"]) == remaining
    monkeypatch.setattr(
        build_linker_inputs, "load_production_model", lambda *a, **kw: pytest.fail("Rebuilt a completed block")
    )
    assert build_linker_inputs.build_linker_inputs(roles, paths, output, model_path, queries_per_block=1) == counts
