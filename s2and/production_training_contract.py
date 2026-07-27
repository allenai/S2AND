"""Shared authorities for production training inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from s2and.consts import _PACKAGE_DATA_DIR
from s2and.name_counts_index import NameCountsIndex
from s2and.name_tuple_artifact import NameTupleArtifact, load_packaged_name_tuple_artifact
from s2and.orcid_prefix_counts import LoadedOrcidPrefixCounts, load_canonical_orcid_prefix_counts

PAIRWISE_TRAINING_PLAN_SCHEMA_VERSION = "s2and_pairwise_training_plan_v1"
LINKER_TARGET_SCHEMA = "incremental_linker_training_target_v1"
LINKER_EVALUATION_REPORT_SCHEMA = "s2and_linker_evaluation_report_v1"
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
            "orcid_prefix_counts_manifest_sha256": self.orcid_prefix_counts.manifest_sha256,
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
