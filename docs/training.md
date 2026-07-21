# Training and Evaluation

This document expands the root README's Arrow-native training example with the
main steps for training, evaluating, and publishing a model.

## Build a Rust-backed training dataset

The maintained training constructor consumes a manifest-backed Arrow
generation. Ground-truth clusters remain JSON by design, but signatures,
papers, paper authors, embeddings, batch indexes, and name counts come from the
same immutable Arrow bundle.

The example below expects a canonical training generation produced by the
current `scripts/convert_to_arrow.py` contract. This migration branch does not
bundle one. Older Arrow directories without `normalization_version` and the
content-addressed `artifact_generation` inventory are intentionally rejected.

```python
import json
from pathlib import Path

from s2and.arrow_training import build_training_anddata_from_arrow
from s2and.consts import NORMALIZATION_VERSION

bundle_dir = Path("/path/to/canonical_arrow_training_bundle/pubmed")
manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
manifest_paths = manifest["paths"]
embedding_key = "specter"
embedding_index_key = "specter_batch_index"
training_keys = [
    "signatures",
    "signatures_batch_index",
    "papers",
    "papers_batch_index",
    "paper_authors",
    "paper_authors_batch_index",
    "name_counts_index",
    embedding_key,
    embedding_index_key,
]
arrow_paths = {
    key: str((bundle_dir / manifest_paths[key]).resolve())
    for key in training_keys
}

dataset = build_training_anddata_from_arrow(
    arrow_paths,
    "pubmed",
    expected_normalization_version=NORMALIZATION_VERSION,
    clusters=str((bundle_dir / manifest_paths["clusters"]).resolve()),
    block_type="s2",
    train_pairs_size=1000,
    val_pairs_size=200,
    test_pairs_size=200,
    n_jobs=4,
)
```

Set `bundle_dir` to a canonical training generation with this manifest
contract. The constructor validates required tables,
raw-planner batch indexes, checksums, normalization provenance, and the
name-count index before it samples any pairs. It always selects the Rust
training runtime and pins canonical preprocessing, regardless of
`S2AND_BACKEND`. The requested pair counts are upper bounds when a split
contains fewer eligible within-block pairs.
The returned dataset's Python-visible signatures and papers are reconstructed
from the validated Arrow bundle; the constructor never injects pre-conversion
source objects. Keep source and reconstructed datasets separate in parity
checks.

## Featurize pairs and train the pairwise model

```python
from s2and.featurizer import FeaturizationInfo, featurize
from s2and.model import PairwiseModeler

featurization_info = FeaturizationInfo()
train, val, test = featurize(dataset, featurization_info, n_jobs=4)
X_train, y_train, _ = train
X_val, y_val, _ = val
X_test, y_test, _ = test

pairwise_model = PairwiseModeler(
    n_iter=25,
    monotone_constraints=featurization_info.lightgbm_monotone_constraints,
)
pairwise_model.fit(X_train, y_train, X_val, y_val)
```

For repeated training experiments on unchanged inputs, wrap featurization with
the snapshot cache instead, exposed only by `train_pairwise.py` as
`--feature-cache-dir`. It stores each split's output
matrices as one content-addressed uncompressed NPZ file and recomputes on any
input change. See [caching.md](caching.md) for the exact semantics.

## Evaluate the pairwise classifier

```python
from s2and.eval import pairwise_eval

pairwise_metrics = pairwise_eval(
    X_test,
    y_test,
    pairwise_model.classifier,
    figs_path="figs/",
    title="example",
    shap_feature_names=featurization_info.get_feature_names(),
)
print(pairwise_metrics)
```

This writes useful diagnostic plots such as ROC, PR, and SHAP outputs under `figs/`.

## Fit the clusterer

```python
from hyperopt import hp

from s2and.model import Clusterer, FastCluster

clusterer = Clusterer(
    featurization_info,
    pairwise_model,
    cluster_model=FastCluster(linkage="average"),
    search_space={"eps": hp.uniform("eps", 0, 1)},
    n_iter=25,
    n_jobs=8,
)
clusterer.fit(dataset)
```

S2AND uses agglomerative clustering with average linkage on top of the pairwise model.

## Evaluate clustering

```python
from s2and.eval import cluster_eval

metrics, metrics_per_signature = cluster_eval(dataset, clusterer)
print(metrics)
```

`metrics_per_signature` is useful when you want to slice performance by signature properties.

## Production pair selection

The existing `train_pairwise_bundle` implementation samples internally from
legacy JSON/pickle `ANDData` inputs. It does not consume the explicit pair
catalogs produced by the pair-source experiments and therefore does not declare
or claim a prospective experiment recipe.

Before promoting an experiment result, route its exact selected rows through
the maintained Arrow featurizer and record the realized source counts,
selection audit, and pair-identity digest in the production training summary.
A tiny end-to-end test must prove that the resolved rows are the rows actually
used for training before running the full production job.

## Publish and reload a trained model

Do not publish a pickle. The public loader accepts only a complete canonical
native bundle containing the pairwise boosters, clusterer configuration,
promoted linker, reproducibility target, and checksummed manifest. The
production training scripts write a pairwise-only staging bundle and then
atomically finalize a complete bundle; see
[production_inference.md](production_inference.md#atomic-publication) for the
exact commands and release gates.

Pairwise production training also verifies that every dataset uses the packaged
canonical name tuples and records two data hashes in `feature_contract`:
`name_tuples_data_sha256` and `orcid_prefix_counts_data_sha256`. Bundle export
does not synthesize missing hashes. Export and load both compare the recorded
values with the canonical artifacts installed in the package.

The promoted incremental-linker artifact uses the strict
`incremental_linking_artifact_v4` contract. It records the canonical digest of
the complete training target JSON. Final bundle assembly and production loading
both reject a different target, including a target modified after manifest
checksums are refreshed.

After a bundle passes those gates, reload it explicitly:

```python
from s2and.production_model import load_production_model

clusterer = load_production_model("/path/to/production_model_vX.Y")
pred_clusters, pred_distance_matrices = clusterer.predict_from_arrow_paths(
    blocks,
    arrow_paths,
    total_ram_bytes=32 * 1024**3,
)
```

`pred_distance_matrices` may be `None` when the fused clustering path is active.

## Reference scripts

- `scripts/production/model/train_pairwise.py`: pairwise production-bundle stage
- `scripts/production/model/train_linker_and_finalize.py`: complete native-bundle finalization
- `scripts/tutorial_for_predicting_with_the_prod_model.py`: released-model inference example
- `scripts/README.md`: script catalog
