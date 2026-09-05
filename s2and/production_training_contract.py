"""Shared authorities for production training inputs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from s2and._sha256 import is_lowercase_sha256, sha256_file
from s2and.consts import _PACKAGE_DATA_DIR
from s2and.name_counts_index import NameCountsIndex
from s2and.name_tuple_artifact import NameTupleArtifact, load_packaged_name_tuple_artifact
from s2and.orcid_prefix_counts import LoadedOrcidPrefixCounts, load_canonical_orcid_prefix_counts

REQUIRED_LINKER_TABLE_KEYS = (
    "train_path",
    "classic_gate_source_path",
    "s2and_eval_path",
    "hwang_eval_path",
)
INTEGER_OFFICIAL_METRIC_KEYS = frozenset(
    {
        "training_rows",
        "training_positive_rows",
        "stratified_test_queries",
        "stratified_test_errors",
        "stratified_test_false_abstain",
        "stratified_test_false_link",
        "stratified_test_wrong_candidate_link",
    }
)
FLOAT_OFFICIAL_METRIC_KEYS = frozenset(
    {
        "stratified_test_accuracy",
        "stratified_test_balanced_accuracy",
        "stratified_test_error_rate",
        "false_abstain_error_rate",
        "false_link_error_rate",
        "wrong_link_error_rate",
        "weighted_average_error",
    }
)
SUPPORTED_OFFICIAL_METRIC_KEYS = (
    INTEGER_OFFICIAL_METRIC_KEYS | FLOAT_OFFICIAL_METRIC_KEYS | {"weighted_average_error_weights"}
)
_COMMON_MODEL_ROLES = frozenset({"papers", "signatures", "specter_embeddings"})
_RANDOM_BLOCK_ROLES = _COMMON_MODEL_ROLES | {"clusters"}
_FIXED_PAIR_ROLES = _COMMON_MODEL_ROLES | {"train_pairs", "val_pairs"}


def block_membership_sha256(blocks: Mapping[str, Sequence[str]]) -> str:
    """Hash block membership independently of row order with bounded scratch space.

    Only the block keys and one block's sorted members are materialized. Length
    prefixes distinguish IDs containing arbitrary separators without building a
    serialized copy of the complete population.
    """
    digest = hashlib.sha256()
    for block_id in sorted(blocks):
        members = blocks[block_id]
        encoded = block_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(members).to_bytes(8, "big"))
        for signature_id in sorted(members):
            encoded = signature_id.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def frozen_test_blocks(blocks: Mapping[str, list[str]], record: Mapping[str, Any]) -> dict[str, list[str]]:
    """Select recorded held-out blocks after verifying the complete population."""
    if not isinstance(record, Mapping) or set(record) != {"block_membership_sha256", "test_block_ids"}:
        raise ValueError("Invalid recorded cluster test split")
    expected_digest = record["block_membership_sha256"]
    if not is_lowercase_sha256(expected_digest):
        raise ValueError("Recorded cluster test split requires a lowercase block_membership_sha256")
    block_ids = record["test_block_ids"]
    if (
        not isinstance(block_ids, list)
        or not block_ids
        or any(not isinstance(block_id, str) for block_id in block_ids)
        or len(set(block_ids)) != len(block_ids)
    ):
        raise ValueError("Recorded cluster test split requires unique test_block_ids")
    if block_membership_sha256(blocks) != expected_digest:
        raise ValueError("Evaluation block membership differs from the recorded training population")
    if any(block_id not in blocks for block_id in block_ids):
        raise ValueError("Recorded test block is absent from the evaluation population")
    return {block_id: blocks[block_id] for block_id in block_ids}


@dataclass(frozen=True, slots=True)
class EpsPolicy:
    """EPS candidates and minimum accepted calibration scores."""

    grid: tuple[float, ...]
    minimum_dataset_f1: float
    minimum_signature_weighted_f1: float


@dataclass(frozen=True, slots=True)
class ModelDataset:
    """Verified files for one model-development dataset."""

    files: Mapping[str, Path]

    @property
    def split_mode(self) -> str:
        """Infer how pairs are selected from the declared file roles."""

        roles = frozenset(self.files)
        if roles == _RANDOM_BLOCK_ROLES:
            return "random_blocks"
        if roles == _FIXED_PAIR_ROLES:
            return "fixed_pairs"
        raise ValueError("invalid model dataset roles")


@dataclass(frozen=True, slots=True)
class ModelPlan:
    """One verified model-development plan and its byte identity."""

    release_version: str
    datasets: Mapping[str, ModelDataset]
    eps: EpsPolicy
    sha256: str


def _unit_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("expected a number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("expected a number in [0, 1]")
    return result


def load_model_plan(path: Path) -> ModelPlan:
    """Load and verify one unversioned model-development plan."""

    contents = Path(path).read_bytes()
    payload = json.loads(contents)
    if not isinstance(payload, dict) or set(payload) != {"release_version", "datasets", "eps"}:
        raise ValueError("invalid model plan")
    release_version = payload["release_version"]
    if not isinstance(release_version, str) or not release_version or release_version.strip() != release_version:
        raise ValueError("invalid release version")

    raw_eps = payload["eps"]
    if not isinstance(raw_eps, dict) or set(raw_eps) != {
        "grid",
        "minimum_dataset_f1",
        "minimum_signature_weighted_f1",
    }:
        raise ValueError("invalid EPS policy")
    raw_grid = raw_eps["grid"]
    if not isinstance(raw_grid, list) or not raw_grid:
        raise ValueError("invalid EPS grid")
    grid = tuple(sorted(_unit_float(value) for value in raw_grid))
    if len(grid) != len(set(grid)):
        raise ValueError("duplicate EPS")
    eps = EpsPolicy(
        grid=grid,
        minimum_dataset_f1=_unit_float(raw_eps["minimum_dataset_f1"]),
        minimum_signature_weighted_f1=_unit_float(raw_eps["minimum_signature_weighted_f1"]),
    )

    raw_datasets = payload["datasets"]
    if not isinstance(raw_datasets, dict) or not raw_datasets:
        raise ValueError("invalid model datasets")
    observed_digests: dict[Path, str] = {}
    datasets: dict[str, ModelDataset] = {}
    for name, raw_files in raw_datasets.items():
        if not isinstance(name, str) or not name or not isinstance(raw_files, dict):
            raise ValueError("invalid model dataset")
        if frozenset(raw_files) not in {_RANDOM_BLOCK_ROLES, _FIXED_PAIR_ROLES}:
            raise ValueError("invalid model dataset roles")
        files: dict[str, Path] = {}
        for role, raw_spec in raw_files.items():
            if not isinstance(raw_spec, dict) or set(raw_spec) != {"path", "sha256"}:
                raise ValueError("invalid model file")
            raw_path = raw_spec["path"]
            expected_digest = raw_spec["sha256"]
            if (
                not isinstance(raw_path, str)
                or not raw_path
                or not Path(raw_path).is_absolute()
                or not is_lowercase_sha256(expected_digest)
            ):
                raise ValueError("invalid model file")
            file_path = Path(raw_path).resolve()
            if file_path not in observed_digests:
                observed_digests[file_path] = sha256_file(file_path)
            if observed_digests[file_path] != expected_digest:
                raise ValueError("model file digest mismatch")
            files[role] = file_path
        datasets[name] = ModelDataset(files=files)
    return ModelPlan(
        release_version=release_version,
        datasets=datasets,
        eps=eps,
        sha256=hashlib.sha256(contents).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class ProductionArtifactAuthority:
    """The external name counts and packaged runtime artifacts used for training."""

    name_counts_index: NameCountsIndex
    name_tuples: NameTupleArtifact
    orcid_prefix_counts: LoadedOrcidPrefixCounts

    @property
    def hashes(self) -> dict[str, str]:
        """Return immutable artifact identities for model and report bindings."""

        return {
            "name_counts_manifest_sha256": self.name_counts_index.manifest_sha256,
            "name_tuples_data_sha256": self.name_tuples.data_sha256,
            "orcid_prefix_counts_data_sha256": self.orcid_prefix_counts.data_sha256,
        }


def load_packaged_artifact_authority(
    *,
    name_counts_index_root: Path,
) -> ProductionArtifactAuthority:
    """Load the external name counts and the package's runtime artifacts."""

    authority = ProductionArtifactAuthority(
        name_counts_index=NameCountsIndex.open(name_counts_index_root),
        name_tuples=load_packaged_name_tuple_artifact(),
        orcid_prefix_counts=load_canonical_orcid_prefix_counts(Path(_PACKAGE_DATA_DIR)),
    )
    if authority.orcid_prefix_counts.name_tuples_sha256 != authority.name_tuples.data_sha256:
        raise ValueError("ORCID prefix counts were generated from a different name-tuple artifact")
    return authority
