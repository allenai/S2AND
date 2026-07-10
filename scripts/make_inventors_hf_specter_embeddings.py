"""Generate one SPECTER embedding set for the selected inventors papers.

Each process loads exactly one model selected by ``--model``:

- ``specter`` loads ``allenai/specter``.
- ``specter2`` loads ``allenai/specter2_base`` with the ``allenai/specter2``
  proximity adapter activated.

Pickle format matches S2AND expectations: `(X, keys)`, where
`X` is `float32` NumPy array with shape `(N, 768)` and `keys` is a list of paper ids.

Examples:
uv run --with torch --with transformers --with adapters \
  python scripts/make_inventors_hf_specter_embeddings.py \
  --model specter \
  --signatures-path scratch/inventors_s2and/inventors_s2and_signatures.json \
  --papers-path scratch/inventors_s2and/inventors_s2and_papers.json \
  --output-path scratch/inventors_s2and/inventors_s2and_specter.pickle \
  --limit 20105 --batch-size 16

uv run --with torch --with transformers --with adapters \
  python scripts/make_inventors_hf_specter_embeddings.py \
  --model specter2 \
  --signatures-path scratch/inventors_s2and/inventors_s2and_signatures.json \
  --papers-path scratch/inventors_s2and/inventors_s2and_papers.json \
  --output-path scratch/inventors_s2and/inventors_s2and_specter2.pkl \
  --limit 20105 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("specter_embeddings")


@dataclass(frozen=True)
class PaperRecord:
    paper_id: str
    title: str
    abstract: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["specter", "specter2"],
        required=True,
        help="The single embedding model to load in this process.",
    )
    parser.add_argument(
        "--signatures-path",
        type=Path,
        default=_PROJECT_ROOT / "scratch" / "inventors_s2and" / "inventors_s2and_signatures.json",
        help="Path to signatures.json dictionary keyed by signature id; paper ids are collected from this file.",
    )
    parser.add_argument(
        "--papers-path",
        type=Path,
        default=_PROJECT_ROOT / "scratch" / "inventors_s2and" / "inventors_s2and_papers.json",
        help="Path to papers.json dictionary keyed by paper id.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output pickle path. Defaults to a model-specific file under scratch/inventors_s2and.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Tokenizer max_length.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of papers to process (for tiny/smoke runs).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Torch device selection.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    parser.add_argument(
        "--disable-autocast",
        action="store_true",
        help="Disable CUDA autocast mixed precision. Enabled by default when --device resolves to cuda.",
    )
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        return str(value)
    return value.strip()


def load_signature_paper_ids(signatures_path: Path) -> set[str]:
    with signatures_path.open("r", encoding="utf-8") as infile:
        payload = json.load(infile)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict at {signatures_path}, found {type(payload)}")

    paper_ids: set[str] = set()
    for signature_id, signature in payload.items():
        if not isinstance(signature, dict):
            raise ValueError(f"Expected signature payload to be dict for signature_id={signature_id}")
        if "paper_id" not in signature:
            raise ValueError(f"Missing `paper_id` for signature_id={signature_id}")
        paper_ids.add(str(signature["paper_id"]))
    if len(paper_ids) == 0:
        raise ValueError(f"No paper ids found in {signatures_path}")
    return paper_ids


def load_paper_records(
    papers_path: Path,
    limit: int | None,
    required_paper_ids: set[str] | None = None,
) -> tuple[list[PaperRecord], set[str]]:
    with papers_path.open("r", encoding="utf-8") as infile:
        payload = json.load(infile)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict at {papers_path}, found {type(payload)}")

    records: list[PaperRecord] = []
    selected_paper_ids: set[str] = set()
    use_required_filter = required_paper_ids is not None
    required_ids = required_paper_ids or set()
    for paper_id, paper in payload.items():
        paper_id_str = str(paper_id)
        if use_required_filter and paper_id_str not in required_ids:
            continue
        if limit is not None and len(records) >= limit:
            break
        if not isinstance(paper, dict):
            raise ValueError(f"Expected paper payload to be dict for paper_id={paper_id}")
        records.append(
            PaperRecord(
                paper_id=paper_id_str,
                title=clean_text(paper.get("title")),
                abstract=clean_text(paper.get("abstract")),
            )
        )
        selected_paper_ids.add(paper_id_str)
    if len(records) == 0:
        raise ValueError(f"No papers found in {papers_path}")
    missing_required = set()
    if use_required_filter:
        missing_required = required_ids - selected_paper_ids
    return records, missing_required


def iter_batches(records: list[PaperRecord], batch_size: int) -> Iterator[list[PaperRecord]]:
    for i in range(0, len(records), batch_size):
        yield records[i : i + batch_size]


def build_text(title: str, abstract: str, sep_token: str) -> str:
    if title and abstract:
        return f"{title}{sep_token}{abstract}"
    if title:
        return title
    if abstract:
        return abstract
    return ""


def resolve_device(requested: str) -> str:
    import torch  # type: ignore

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available")
    return requested


def load_specter_model(device: str):
    from transformers import AutoModel, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
    model = AutoModel.from_pretrained("allenai/specter")
    model.to(device)
    model.eval()
    return tokenizer, model


def load_specter2_proximity_model(device: str):
    from adapters import AutoAdapterModel  # type: ignore
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
    if hasattr(model, "set_active_adapters"):
        model.set_active_adapters("proximity")
    model.to(device)
    model.eval()
    return tokenizer, model


def embed_records(
    records: list[PaperRecord],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    max_length: int,
    device: str,
    progress_desc: str,
    use_autocast: bool,
) -> tuple[np.ndarray, list[str]]:
    import torch  # type: ignore

    keys: list[str] = []
    embeddings_chunks: list[np.ndarray] = []
    sep_token = tokenizer.sep_token or " [SEP] "
    autocast_enabled = use_autocast and device == "cuda"

    total_batches = (len(records) + batch_size - 1) // batch_size
    with torch.inference_mode():
        for batch in tqdm(
            iter_batches(records, batch_size),
            total=total_batches,
            desc=progress_desc,
        ):
            batch_keys = [row.paper_id for row in batch]
            batch_texts = [build_text(row.title, row.abstract, sep_token) for row in batch]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16) if autocast_enabled else nullcontext():
                output = model(**encoded)
            hidden = output.last_hidden_state
            if hidden is None:
                raise RuntimeError("Model output has no last_hidden_state")
            cls_embeddings = hidden[:, 0, :].detach().cpu().numpy().astype(np.float32, copy=False)
            keys.extend(batch_keys)
            embeddings_chunks.append(cls_embeddings)

    embeddings = np.concatenate(embeddings_chunks, axis=0)
    if embeddings.shape[0] != len(keys):
        raise RuntimeError(f"Embedding row count mismatch: rows={embeddings.shape[0]} vs keys={len(keys)}")
    return embeddings, keys


def dump_pickle(path: Path, embeddings: np.ndarray, keys: list[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Re-run with --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as outfile:
        pickle.dump((embeddings, keys), outfile, protocol=4)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.max_length <= 0:
        raise ValueError("--max-length must be > 0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be > 0 when provided")

    required_paper_ids = load_signature_paper_ids(args.signatures_path)
    LOGGER.info("Loaded %d unique paper ids from %s", len(required_paper_ids), args.signatures_path)

    records, missing_required = load_paper_records(
        papers_path=args.papers_path,
        limit=args.limit,
        required_paper_ids=required_paper_ids,
    )
    if args.limit is None and missing_required:
        example_missing = sorted(missing_required)[:5]
        raise ValueError(
            f"{len(missing_required)} signature paper ids were not found in papers.json. "
            f"Examples: {example_missing}"
        )
    if args.limit is not None and missing_required:
        LOGGER.info(
            "Skipped %d signature paper ids because --limit=%d truncated the run.",
            len(missing_required),
            args.limit,
        )
    LOGGER.info("Loaded %d papers from %s after signature filtering", len(records), args.papers_path)

    device = resolve_device(args.device)
    LOGGER.info("Using device: %s", device)
    use_autocast = (not args.disable_autocast) and device == "cuda"
    LOGGER.info("CUDA autocast enabled: %s", use_autocast)

    if args.model == "specter":
        output_path = args.output_path or (
            _PROJECT_ROOT / "scratch" / "inventors_s2and" / "inventors_s2and_specter.pickle"
        )
        LOGGER.info("Loading `allenai/specter`...")
        tokenizer, model = load_specter_model(device)
        progress_desc = "Embedding allenai/specter"
    else:
        output_path = args.output_path or (
            _PROJECT_ROOT / "scratch" / "inventors_s2and" / "inventors_s2and_specter2.pkl"
        )
        LOGGER.info("Loading `allenai/specter2_base` + `allenai/specter2` proximity adapter...")
        tokenizer, model = load_specter2_proximity_model(device)
        progress_desc = "Embedding specter2 proximity"

    embeddings, keys = embed_records(
        records=records,
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        progress_desc=progress_desc,
        use_autocast=use_autocast,
    )
    LOGGER.info("%s output shape: %s", args.model, tuple(embeddings.shape))
    dump_pickle(output_path, embeddings, keys, overwrite=args.overwrite)
    LOGGER.info("Wrote %s", output_path)
    LOGGER.info("Done. model=%s papers=%d dimensions=%d", args.model, len(records), int(embeddings.shape[1]))


if __name__ == "__main__":
    main()
