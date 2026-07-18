from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts._pair_ablation.evaluation_artifact import (
    build_evaluation_manifest,
    evaluation_pair_digest,
    load_evaluation_artifact,
    write_evaluation_artifact,
)
from scripts._pair_ablation.pair_sources import PAIR_COLUMNS

INPUT_DIGEST = "a" * 64
ORACLES = {"aminer": "gold_cluster_pairs", "medline": "fixed_pair_labels"}
CONFIG = {
    "evaluation_seed": 1111,
    "eval_pairs_per_domain": 100_000,
    "big_proxy_eval_pairs_per_class": 10_000,
}


def _row(domain: str, family: str, pair1: str, pair2: str, label: int) -> dict[str, object]:
    return {
        "source_domain": domain,
        "source_family": family,
        "pair1": pair1,
        "pair2": pair2,
        "label": label,
        "label_rule": "gold",
        "origin": "fixture",
        "group_id": f"{domain}:{pair1}",
    }


def _pairs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("aminer", "evaluation_gold", "a1", "a2", 1),
            _row("aminer", "evaluation_gold", "a1", "a3", 0),
            _row("medline", "evaluation_fixed", "m1", "m2", 1),
            _row("medline", "evaluation_fixed", "m3", "m4", 0),
        ],
        columns=list(PAIR_COLUMNS),
    )


def test_evaluation_artifact_round_trip_and_domain_digests(tmp_path: Path) -> None:
    artifact = write_evaluation_artifact(
        tmp_path / "evaluation",
        _pairs(),
        oracle_by_domain=ORACLES,
        evaluation_config=CONFIG,
        input_digest=INPUT_DIGEST,
    )

    loaded = load_evaluation_artifact(
        tmp_path / "evaluation",
        oracle_by_domain=ORACLES,
        evaluation_config=CONFIG,
        input_digest=INPUT_DIGEST,
    )

    assert artifact.manifest == loaded.manifest
    assert set(loaded.pair_digest_by_domain) == set(ORACLES)
    assert loaded.oracle_by_domain == ORACLES
    assert loaded.manifest["rows"] == 4


def test_evaluation_pair_digest_is_order_invariant_but_oracle_sensitive() -> None:
    frame = _pairs().loc[lambda value: value["source_domain"].eq("aminer")]

    first = evaluation_pair_digest(frame, oracle_kind="gold_cluster_pairs")
    shuffled = evaluation_pair_digest(frame.sample(frac=1, random_state=7), oracle_kind="gold_cluster_pairs")
    changed_oracle = evaluation_pair_digest(frame, oracle_kind="proxy")

    assert first == shuffled
    assert first != changed_oracle


def test_evaluation_manifest_rejects_missing_domain() -> None:
    frame = _pairs().loc[lambda value: value["source_domain"].eq("aminer")]

    with pytest.raises(ValueError, match="evaluation domains mismatch"):
        build_evaluation_manifest(
            frame,
            oracle_by_domain=ORACLES,
            evaluation_config=CONFIG,
            input_digest=INPUT_DIGEST,
            pairs_sha256="b" * 64,
        )


def test_evaluation_manifest_rejects_single_class_domain() -> None:
    frame = _pairs()
    frame.loc[frame["source_domain"].eq("aminer"), "label"] = 1

    with pytest.raises(ValueError, match="must contain both classes"):
        build_evaluation_manifest(
            frame,
            oracle_by_domain=ORACLES,
            evaluation_config=CONFIG,
            input_digest=INPUT_DIGEST,
            pairs_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    ("config", "input_digest", "match"),
    [
        ({**CONFIG, "evaluation_seed": 2222}, INPUT_DIGEST, "configuration mismatch"),
        (CONFIG, "f" * 64, "input identity mismatch"),
    ],
)
def test_evaluation_artifact_rejects_identity_mismatch(
    tmp_path: Path,
    config: dict[str, int],
    input_digest: str,
    match: str,
) -> None:
    root = tmp_path / "evaluation"
    write_evaluation_artifact(
        root,
        _pairs(),
        oracle_by_domain=ORACLES,
        evaluation_config=CONFIG,
        input_digest=INPUT_DIGEST,
    )

    with pytest.raises(ValueError, match=match):
        load_evaluation_artifact(
            root,
            oracle_by_domain=ORACLES,
            evaluation_config=config,
            input_digest=input_digest,
        )


def test_evaluation_artifact_rejects_manifest_content_drift(tmp_path: Path) -> None:
    root = tmp_path / "evaluation"
    write_evaluation_artifact(
        root,
        _pairs(),
        oracle_by_domain=ORACLES,
        evaluation_config=CONFIG,
        input_digest=INPUT_DIGEST,
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["domains"][0]["positives"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match evaluation pair content"):
        load_evaluation_artifact(
            root,
            oracle_by_domain=ORACLES,
            evaluation_config=CONFIG,
            input_digest=INPUT_DIGEST,
        )


def test_evaluation_artifact_rejects_parquet_corruption(tmp_path: Path) -> None:
    root = tmp_path / "evaluation"
    write_evaluation_artifact(
        root,
        _pairs(),
        oracle_by_domain=ORACLES,
        evaluation_config=CONFIG,
        input_digest=INPUT_DIGEST,
    )
    with (root / "pairs.parquet").open("ab") as handle:
        handle.write(b"corruption")

    with pytest.raises(ValueError, match="Parquet hash mismatch"):
        load_evaluation_artifact(
            root,
            oracle_by_domain=ORACLES,
            evaluation_config=CONFIG,
            input_digest=INPUT_DIGEST,
        )
