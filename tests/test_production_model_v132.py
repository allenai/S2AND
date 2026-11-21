"""Compare release 1.3.2 predictions with the parent revision on a real block."""

import hashlib
import json
from pathlib import Path

import pytest

from s2and.arrow_inputs import ArrowDataset
from s2and.consts import CONFIG
from s2and.data import ANDData
from s2and.incremental_linking.feature_block import write_raw_arrow_batch_lookup_indexes
from s2and.production_model import load_production_model
from scripts.arrow_conversion_helpers import write_raw_planner_arrow_from_anddata

_FIXTURE = Path(__file__).parent / "qian"
_EXPECTED_SHA256 = "2836e6cf60eb4e4d467d0e81d7139347d9a8db01a97c5dc87f14c60005cfc3b6"


def _release_predictions(inference_root: Path, work_dir: Path) -> bytes:
    """Serialize predictions on the first real block in the Qian fixture."""

    model = load_production_model(inference_root / "production_model_v1.3.2", n_jobs=1)
    index = inference_root / "name_counts_index"
    signatures = json.loads((_FIXTURE / "signatures.json").read_bytes())
    signature_ids = [str(i) for i in range(12)]
    block = {"a sattar": signature_ids}
    dataset = ANDData(
        signatures={sid: signatures[sid] for sid in signature_ids},
        papers=str(_FIXTURE / "papers.json"),
        specter_embeddings=str(_FIXTURE / "specter.pickle"),
        name_counts_index=index,
        name="v132-release-parity",
        mode="inference",
        n_jobs=1,
    )
    classic, _ = model.predict(block, dataset)
    paths = write_raw_planner_arrow_from_anddata(dataset, work_dir, signature_ids=signature_ids)
    paths, _ = write_raw_arrow_batch_lookup_indexes(paths, work_dir)
    paths["name_counts_index"] = str(index)
    with ArrowDataset.open_trusted(paths, require_specter=True, require_name_counts_index=True) as arrow:
        bulk, _ = model.predict_from_arrow(block, arrow, batching_threshold=8)
    assert all(
        sorted(sid for members in clusters.values() for sid in members) == sorted(signature_ids)
        for clusters in (classic, bulk)
    )
    return json.dumps({"classic": classic, "arrow_subblocked": bulk}, separators=(",", ":")).encode()


def test_release_132_matches_parent_clusters(tmp_path: Path) -> None:
    inference_root = Path(CONFIG["main_data_dir"])
    if not (inference_root / "production_model_v1.3.2").is_dir():
        pytest.skip("Requires the 1.3.2 inference bundle in the configured data directory")

    assert hashlib.sha256(_release_predictions(inference_root, tmp_path)).hexdigest() == _EXPECTED_SHA256
