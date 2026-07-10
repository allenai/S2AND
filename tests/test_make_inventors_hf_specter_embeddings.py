from __future__ import annotations

import pytest

from scripts.make_inventors_hf_specter_embeddings import parse_args


@pytest.mark.parametrize("model", ["specter", "specter2"])
def test_parse_args_requires_one_supported_model(model: str) -> None:
    args = parse_args(["--model", model])

    assert args.model == model
    assert args.output_path is None


@pytest.mark.parametrize("argv", [[], ["--model", "both"]])
def test_parse_args_rejects_missing_or_multiple_model_mode(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(argv)
