from __future__ import annotations

import pytest

from scripts._rust_suite import featurizer_reuse_cmd


def test_featurizer_reuse_rejects_json_before_loading_rust_or_data() -> None:
    with pytest.raises(ValueError, match="requires --input-format arrow"):
        featurizer_reuse_cmd.run_reuse_profile(
            dataset_name="qian",
            n_jobs=1,
            repeats=1,
            model_path="missing-model",
            input_format="json",
        )
