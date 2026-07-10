"""Concurrency checks for production-bundle publication."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import s2and.production_bundle as production_bundle


def _write_complete_stage(path: Path, writer: str) -> None:
    (path / "incremental_linker").mkdir(parents=True)
    (path / "incremental_linker" / "artifact.bin").write_bytes(b"same-linker")
    reproducibility = path / "reproducibility"
    reproducibility.mkdir()
    (reproducibility / "incremental_linker_training_target.json").write_text(
        json.dumps({"writer": writer}),
        encoding="utf-8",
    )
    (path / "manifest.json").write_text(
        json.dumps({"bundle_status": "complete", "writer": writer}),
        encoding="utf-8",
    )


def test_same_path_finalizers_publish_one_coherent_bundle(tmp_path: Path) -> None:
    output = tmp_path / "production_model_v9.9"
    output.mkdir()
    (output / "manifest.json").write_text(
        json.dumps({"bundle_status": "pairwise_only"}),
        encoding="utf-8",
    )
    stages = [tmp_path / "stage-a", tmp_path / "stage-b"]
    _write_complete_stage(stages[0], "a")
    _write_complete_stage(stages[1], "b")
    barrier = threading.Barrier(2)

    def publish(stage: Path) -> None:
        barrier.wait(timeout=5)
        production_bundle._publish_staged_bundle(  # noqa: SLF001
            stage,
            output,
            allow_pairwise_replacement=True,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish, stage) for stage in stages]
    failures = [future.exception() for future in futures if future.exception() is not None]

    assert len(failures) == 1
    assert isinstance(failures[0], FileExistsError)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    target = json.loads(
        (output / "reproducibility" / "incremental_linker_training_target.json").read_text(encoding="utf-8")
    )
    assert manifest["writer"] == target["writer"]
    assert (output / "incremental_linker" / "artifact.bin").read_bytes() == b"same-linker"
