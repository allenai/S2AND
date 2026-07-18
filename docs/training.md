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
embedding_key = next(key for key in ("specter", "specter2") if key in manifest_paths)
embedding_index_key = next(
    key for key in ("specter_batch_index", "specter2_batch_index") if key in manifest_paths
)
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
training runtime, regardless of `S2AND_BACKEND`. The requested pair counts are
upper bounds when a split contains fewer eligible within-block pairs.

## Featurize pairs and train the pairwise model

```python
from s2and.featurizer import FeaturizationInfo, featurize
from s2and.model import PairwiseModeler

featurization_info = FeaturizationInfo()
train, val, test = featurize(dataset, featurization_info, n_jobs=4, use_cache=False)
X_train, y_train = train
X_val, y_val = val
X_test, y_test = test

pairwise_model = PairwiseModeler(
    n_iter=25,
    monotone_constraints=featurization_info.lightgbm_monotone_constraints,
)
pairwise_model.fit(X_train, y_train, X_val, y_val)
```

Set `use_cache=True` when repeated experiments intentionally reuse the same
pair rows. The persistent cache and the in-process Rust featurizer cache are
separate:

- the persistent pair-feature cache can reuse previously computed pair rows
- the Rust featurizer can be reused in-process even when `use_cache=False`

See [caching.md](caching.md) for the exact cache semantics.

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

## Canonical next-release pair recipe

The selected pairwise recipe is declared as
`PRODUCTION_PAIRWISE_RECIPE` in
`scripts/production/model/train_pairwise.py`. Its resolved name is
`big7_1250_v1`:

- sample 100,000 uniform within-block pairs from each of `aminer`,
  `arnetminer`, `inspire`, `kisti`, `pubmed`, `qian`, and `zbmath`;
- preserve those 700,000 base rows unchanged;
- add 1,250 linker-derived pairs from each of `a_khan`, `a_silva`,
  `h_wang`, `j_smith`, `s_gupta`, `s_lee`, and `s_park`;
- select at most 625 positive and 625 negative linker pairs per domain,
  using deterministic hash-ranked prefixes with no majority-class backfill;
- remove linker rows overlapping the base before selection.

With complete sources, the nominal training set contains 708,750 rows.
`s2and.pairwise_training.resolve_pairwise_training_recipe` fails unless every
base and linker quota is satisfied and returns the selected rows together with
their source counts, selection audit, and identity digest.

The existing `train_pairwise_bundle` implementation still samples internally
from legacy JSON/pickle `ANDData` inputs and therefore does **not** apply or
claim this recipe. The next release must route the exact resolved rows through
the maintained Arrow featurizer once the updated canonical datasets and
name-count generation are available. Do not label a legacy trainer output as
`big7_1250_v1`.

## Publish and reload a trained model

Do not publish a pickle. The public loader accepts only a complete canonical
native bundle containing the pairwise boosters, clusterer configuration,
promoted linker, reproducibility target, and checksummed manifest. The
production training scripts write a pairwise-only staging bundle and then
atomically finalize a complete bundle; see
[production_inference.md](production_inference.md#atomic-publication) for the
exact commands and release gates.

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
