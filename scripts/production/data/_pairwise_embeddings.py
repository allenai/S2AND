"""Encode changed augmented text with explicitly selected local SPECTER2 files."""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

CHECKPOINT_ROWS = 1024


def checkpoint_due(stop: int, saved: int, rows: int) -> bool:
    """Save progress after each 1,024 new rows for any batch size, and at completion."""
    return stop - saved >= CHECKPOINT_ROWS or stop == rows


def _save(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def encode_refresh(
    *,
    root: Path,
    models_root: Path,
    batch_size: int = 16,
    device: str = "cuda",
    n_jobs: int = 1,
) -> None:
    """Resume local inference only for the same input and pinned model metadata."""
    if batch_size < 1 or n_jobs < 1:
        raise ValueError("Batch size and CPU thread count must be positive")
    input_path = root / "changed_papers.arrow"
    with input_path.open("rb") as stream:
        input_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    with pa.memory_map(str(input_path), "r") as stream:
        rows = pa.ipc.open_file(stream).read_all().to_pylist()
    if not rows:
        return
    if any(not any((row[field] or "").strip() for field in ("title", "abstract")) for row in rows):
        raise ValueError("Blank text must retain absent vectors")
    import torch
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer

    torch.set_num_threads(n_jobs)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable")
    models = json.loads((models_root / "model.json").read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(models_root / "models/base", local_files_only=True)
    model = AutoAdapterModel.from_pretrained(models_root / "models/base", local_files_only=True)
    model.load_adapter(str(models_root / "models/adapter"), load_as="proximity", set_active=True)
    model.set_active_adapters("proximity")
    model.eval().to(device)
    if str(model.active_adapters) != "Stack[proximity]":
        raise ValueError("Wrong active SPECTER2 adapter")
    started = time.perf_counter()
    progress_path, vector_path = root / "progress.json", root / "vectors.npy"
    if progress_path.exists():
        progress = json.loads(progress_path.read_bytes())
        if progress["input_sha256"] != input_sha or progress["rows"] != len(rows) or progress["models"] != models:
            raise ValueError("Checkpoint input or model metadata differs")
        completed = progress["completed_rows"]
        vectors = np.load(vector_path, mmap_mode="r+")
        if vectors.shape != (len(rows), 768) or not 0 <= completed <= len(rows):
            raise ValueError("Checkpoint vector shape or progress differs")
    else:
        completed = 0
        vectors = np.lib.format.open_memmap(vector_path, mode="w+", dtype=np.float32, shape=(len(rows), 768))
        progress = {
            "input_sha256": input_sha,
            "rows": len(rows),
            "completed_rows": 0,
            "started_utc": datetime.now(UTC).isoformat(),
            "models": models,
        }
        _save(progress_path, progress)
    saved = completed
    for start in range(completed, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [(row["title"] or "") + tokenizer.sep_token + (row["abstract"] or "") for row in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=512,
        ).to(device)
        precision = torch.autocast("cuda", dtype=torch.float16) if device.startswith("cuda") else nullcontext()
        with torch.inference_mode(), precision:
            output = model(**encoded).last_hidden_state[:, 0, :].float().cpu().numpy()
        if (
            output.shape != (len(batch), 768)
            or not np.isfinite(output).all()
            or not (np.linalg.norm(output, axis=1) > 0).all()
        ):
            raise ValueError("Invalid SPECTER2 output vectors")
        vectors[start : start + len(batch)] = output
        stop = start + len(batch)
        if checkpoint_due(stop, saved, len(rows)):
            saved = stop
            vectors.flush()
            progress.update(completed_rows=stop, elapsed_seconds=time.perf_counter() - started)
            _save(progress_path, progress)
            print(json.dumps({"completed_rows": stop, "rows": len(rows)}), flush=True)
    progress.update(status="complete", finished_utc=datetime.now(UTC).isoformat())
    _save(progress_path, progress)
