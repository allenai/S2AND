from __future__ import annotations

import subprocess
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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
    monkeypatch.setattr(compare_cmd, "_set_backend_env", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compare_cmd, "_load_dataset_inputs", lambda *_args, **_kwargs: ({}, {}, None))
    monkeypatch.setattr(
        compare_cmd,
        "_write_bounded_name_counts_index",
        lambda *_args, **_kwargs: (str(tmp_path / "name_counts_index"), "counts"),
    )
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
            data_root=str(tmp_path),
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


def test_run_single_rust_uses_arrow_native_featurizer(monkeypatch, tmp_path: Path) -> None:
    from s2and import arrow_inputs, consts, feature_port, featurizer
    from s2and.incremental_linking import feature_block_arrow
    from scripts import arrow_conversion_helpers

    monkeypatch.setattr(compare_cmd, "_set_backend_env", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(compare_cmd, "_load_dataset_inputs", lambda *_args, **_kwargs: ({}, {}, None))
    monkeypatch.setattr(
        compare_cmd,
        "_write_bounded_name_counts_index",
        lambda *_args, **_kwargs: (str(tmp_path / "name_counts_index"), "counts"),
    )
    monkeypatch.setattr(compare_cmd, "_collect_rust_package_info", lambda *_args: {"version": "test"})
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
    monkeypatch.setattr(consts, "NAME_COUNTS_INDEX_PATH", tmp_path / "name_counts_index")
    monkeypatch.setattr(featurizer, "DEFAULT_FEATURE_GROUPS", ("name_counts",))

    class FakeANDData:
        def __init__(self, **_kwargs):
            self.signatures = {"s1": {}, "s2": {}}
            self.name_tuples = None

    class FakeFeaturizationInfo:
        def __init__(self, *, features_to_use):
            self.features_to_use = list(features_to_use)
            self.feature_group_to_index = {"name_counts": [0]}

        def get_feature_names(self):
            return ["first_name_count"]

    monkeypatch.setattr("s2and.data.ANDData", FakeANDData)
    monkeypatch.setattr(featurizer, "FeaturizationInfo", FakeFeaturizationInfo)

    def fail_python_featurization(*_args, **_kwargs):
        raise AssertionError("Rust comparison must not call Python many_pairs_featurize")

    monkeypatch.setattr(featurizer, "many_pairs_featurize", fail_python_featurization)

    arrow_write_calls = []

    def fake_write_arrow(dataset, output_dir, **kwargs):
        arrow_write_calls.append((dataset, output_dir, kwargs))
        return {
            "signatures": str(tmp_path / "signatures.arrow"),
            "papers": str(tmp_path / "papers.arrow"),
            "paper_authors": str(tmp_path / "paper_authors.arrow"),
        }

    monkeypatch.setattr(arrow_conversion_helpers, "write_feature_block_arrow_from_anddata", fake_write_arrow)
    monkeypatch.setattr(
        feature_block_arrow,
        "write_raw_arrow_batch_lookup_indexes",
        lambda paths, _output_dir: (paths, {}),
    )
    monkeypatch.setattr(
        arrow_inputs,
        "build_arrow_artifact_manifest",
        lambda _paths, _output_dir: {"manifest": "built"},
    )
    monkeypatch.setattr(
        arrow_inputs,
        "write_arrow_artifact_manifest",
        lambda _paths, output_dir: Path(output_dir) / "manifest.json",
    )
    validated_paths = object()
    monkeypatch.setattr(
        arrow_inputs,
        "validate_arrow_prediction_artifacts",
        lambda *_args, **_kwargs: validated_paths,
    )

    native_calls = []

    class FakeRustFeaturizer:
        def signature_ids(self):
            return ["s1", "s2"]

        def featurize_pairs_matrix_indexed(self, indexed_pairs, selected_indices, n_jobs, nan_value):
            native_calls.append((indexed_pairs, selected_indices, n_jobs, nan_value))
            return np.asarray([[7.0]], dtype=np.float64)

    builder_calls = []

    def fake_build_rust_featurizer(paths, **kwargs):
        builder_calls.append((paths, kwargs))
        return FakeRustFeaturizer()

    monkeypatch.setattr(feature_port, "build_rust_featurizer_from_arrow_paths", fake_build_rust_featurizer)

    output_path = tmp_path / "rust_features.npy"
    result = compare_cmd._run_single(
        Namespace(
            backend="rust",
            dataset="dummy",
            data_root=str(tmp_path),
            limit=2,
            pair_count=1,
            n_jobs=1,
            chunk_size=10,
            seed=7,
            require_non_dev_rust=0,
            require_rust_release=0,
            output_features_path=str(output_path),
        )
    )

    assert len(arrow_write_calls) == 1
    assert arrow_write_calls[0][2]["signature_ids"] == ["s1", "s2"]
    assert builder_calls == [
        (
            validated_paths,
            {
                "expected_normalization_version": consts.NORMALIZATION_VERSION,
                "signature_ids": ["s1", "s2"],
                "name_tuples": None,
                "load_name_counts": True,
                "preprocess": True,
                "num_threads": 1,
            },
        )
    ]
    assert len(native_calls) == 1
    assert native_calls[0][0] == [(1, 0)]
    assert native_calls[0][1:3] == ([0], 1)
    assert np.isnan(native_calls[0][3])
    assert result["execution_route"] == compare_cmd.RUST_EXECUTION_ROUTE
    assert np.array_equal(np.load(output_path), np.asarray([[7.0]], dtype=np.float64))


def test_run_compare_rejects_different_sampled_pairs(monkeypatch) -> None:
    subprocess_results = iter(
        [
            {
                "records_sha256": "same-records",
                "name_counts_sha256": "same-counts",
                "signature_ids_sha256": "same-signatures",
                "pairs_sha256": "python-pairs",
            },
            {
                "records_sha256": "same-records",
                "name_counts_sha256": "same-counts",
                "signature_ids_sha256": "same-signatures",
                "pairs_sha256": "rust-pairs",
            },
        ]
    )
    monkeypatch.setattr(compare_cmd, "_run_subprocess_once", lambda **_kwargs: next(subprocess_results))

    with pytest.raises(RuntimeError, match="same sampled pairs"):
        compare_cmd._run_compare(
            Namespace(
                dataset="dummy",
                limit=2,
                pair_count=1,
                n_jobs=1,
                seed=7,
            )
        )


def test_run_subprocess_surfaces_child_stderr(monkeypatch, tmp_path: Path) -> None:
    def fail(cmd, **_kwargs):
        raise subprocess.CalledProcessError(3, cmd, output="child stdout", stderr="child failure details")

    monkeypatch.setattr(compare_cmd.subprocess, "run", fail)

    with pytest.raises(RuntimeError, match="child failure details"):
        compare_cmd._run_subprocess_once(
            script_path=tmp_path / "compare.py",
            backend="rust",
            features_npy_path=tmp_path / "features.npy",
            args=Namespace(
                dataset="dummy",
                data_root=str(tmp_path),
                limit=2,
                pair_count=1,
                n_jobs=1,
                chunk_size=10,
                seed=7,
                require_non_dev_rust=0,
                require_rust_release=0,
            ),
        )


@pytest.mark.parametrize(
    ("limit", "pair_count", "message"),
    [
        (0, 1, "--limit must be a positive integer"),
        (-1, 1, "--limit must be a positive integer"),
        (2, -1, "--pair-count must be non-negative"),
    ],
)
def test_validate_bounded_args_rejects_unsafe_limits(limit: int, pair_count: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compare_cmd._validate_bounded_args(Namespace(limit=limit, pair_count=pair_count))


def test_bounded_name_count_mappings_use_canonical_keys() -> None:
    mappings = compare_cmd._bounded_name_count_mappings(
        {
            "s1": {"author_info": {"first": "Abdul", "middle": None, "last": "Sattar"}},
            "s2": {"author_info": {"first": "ABDUL", "middle": "X", "last": "Sattar"}},
        }
    )

    assert mappings == (
        {"abdul": 2},
        {"sattar": 2},
        {"abdul sattar": 2},
        {"sattar a": 2},
    )
