from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import s2and.runtime as runtime
import scripts._rust_suite.compare_cmd as compare_cmd


def test_collect_rust_package_info_loads_lazy_extension(monkeypatch) -> None:
    from s2and import feature_port

    current_version = runtime.REQUIRED_RUST_EXTENSION_VERSION
    fake_module = SimpleNamespace(__version__=current_version, __name__="s2and_rust", __file__="native.pyd")
    monkeypatch.setattr(feature_port, "s2and_rust", None)
    monkeypatch.setattr(feature_port, "_ensure_s2and_rust_loaded", lambda: fake_module)
    monkeypatch.setattr(
        compare_cmd,
        "collect_rust_extension_identity",
        lambda **_kwargs: {"module_path": "native.pyd"},
    )

    info = compare_cmd._collect_rust_package_info(False, False)  # noqa: SLF001

    assert info["version"] == current_version
    assert info["module_name"] == "s2and_rust"


def test_run_single_loads_name_counts_for_name_count_features(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(compare_cmd, "_load_dataset_inputs", lambda *_args, **_kwargs: ({}, {}, None))
    monkeypatch.setattr(
        compare_cmd,
        "ProcessTreeRSSMonitor",
        type(
            "FakeMonitor",
            (),
            {
                "__init__": lambda self, interval_seconds=0.05: setattr(self, "peak_gb", 0.25),
                "__enter__": lambda self: self,
                "__exit__": lambda self, exc_type, exc, tb: False,
            },
        ),
    )

    captured_anddata_kwargs = {}

    class FakeANDData:
        def __init__(self, **kwargs):
            captured_anddata_kwargs.update(kwargs)
            self.signatures = {"s1": {}, "s2": {}}

    class FakeFeaturizationInfo:
        def __init__(self, *, features_to_use):
            self.features_to_use = list(features_to_use)

        def get_feature_names(self):
            return list(self.features_to_use)

    def fake_many_pairs_featurize(pairs, dataset, featurizer_info, *_args, **_kwargs):
        assert "name_counts" in featurizer_info.features_to_use
        return np.zeros((len(pairs), len(featurizer_info.features_to_use))), None, None

    monkeypatch.setattr("s2and.data.ANDData", FakeANDData)
    monkeypatch.setattr("s2and.featurizer.FeaturizationInfo", FakeFeaturizationInfo)
    monkeypatch.setattr("s2and.featurizer.many_pairs_featurize", fake_many_pairs_featurize)

    result = compare_cmd._run_single(
        Namespace(
            backend="python",
            dataset="dummy",
            limit=2,
            pair_count=1,
            n_jobs=1,
            chunk_size=10,
            seed=7,
            require_non_dev_rust=0,
            require_rust_release=0,
            output_features_path=str(tmp_path / "features.npy"),
        )
    )

    assert str(captured_anddata_kwargs["name_counts_index"]).endswith("name_counts_index")
    assert "name_counts" in result["feature_names"]
