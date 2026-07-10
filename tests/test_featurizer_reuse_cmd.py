from __future__ import annotations

import pytest

from scripts._rust_suite import featurizer_reuse_cmd


def test_run_reuse_profile_rejects_unknown_input_format() -> None:
    with pytest.raises(ValueError, match="Unsupported input_format"):
        featurizer_reuse_cmd.run_reuse_profile(
            dataset_name="kisti",
            n_jobs=1,
            repeats=1,
            model_path="model",
            input_format="parquet",
        )
