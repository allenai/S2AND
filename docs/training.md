# Training and Evaluation

This document expands the root README's Arrow-native training example with the
main steps for training, evaluating, and publishing a model.

The examples are research/API examples and intentionally make a test split
available for immediate inspection. They are not the v1.3 release protocol.
Release training must keep pairwise, clustering, and linker test identities
sealed until the one-shot Stage 5 evaluation. Follow
[release.md](release.md), not the example order below, for
production work.

## Build a Rust-backed training dataset

The maintained training constructor consumes an open, manifest-backed
`ArrowDataset`. Ground-truth clusters remain JSON by design, but signatures,
papers, paper authors, embeddings, batch indexes, and name counts come from the
same immutable Arrow bundle.

The example below expects a canonical training root produced by
`scripts/convert_to_arrow.py`. This migration branch does not bundle one.

```python
import json
from pathlib import Path

from s2and.arrow_inputs import ArrowDataset
from s2and.arrow_training import build_training_anddata_from_arrow

bundle_dir = Path("/path/to/canonical_arrow_training_bundle/pubmed")
manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
manifest_paths = manifest["paths"]
arrow_dataset = ArrowDataset.open(
    bundle_dir,
    require_name_counts_index=True,
)

dataset = build_training_anddata_from_arrow(
    arrow_dataset,
    "pubmed",
    clusters=str((bundle_dir / manifest_paths["clusters"]).resolve()),
    train_pairs_size=1000,
    val_pairs_size=200,
    test_pairs_size=200,
    n_jobs=4,
)
```

Set `bundle_dir` to a canonical training root. Opening the handle validates
required tables, raw-planner batch indexes, checksums, public format `1`, and
the name-count index before any pairs are sampled. Keep `arrow_dataset`
open while the returned dataset is used, then close it. The constructor always
selects the Rust training runtime and pins canonical preprocessing, regardless
of `S2AND_BACKEND`. The requested pair counts are upper bounds when a split
contains fewer eligible within-block pairs.
The returned dataset's Python-visible signatures and papers are reconstructed
from the validated Arrow bundle; the constructor never injects pre-conversion
source objects. Keep source and reconstructed datasets separate in parity
checks.

Pass `use_orcid_id=False` when a benchmark must remove ORCID evidence. The
policy applies to both reconstructed signatures and the native Rust
featurizer without rewriting the immutable Arrow bundle.

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

The production `train_pairwise.py` command always featurizes from its frozen
`model_plan.json` and has no cache or smoke mode. Programmatic research callers
use `featurize(...)` directly; there is no second persistent feature-snapshot
format to coordinate with the runtime.

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
SHAP diagnostics support directly fitted tree and LightGBM classifiers. For
calibrated, voting, stacking, or non-tree classifiers, pass `skip_shap=True` to
`pairwise_eval`.

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

Omitting `search_space` leaves `clusterer.search_space` as `None` during
construction and inference. `fit()` creates the same uniform EPS space shown
above when calibration starts. Custom calibration code can explicitly call
`s2and.calibration.default_cluster_search_space()`; supplied search spaces are
retained unchanged. Loading a fitted production bundle does not initialize
Hyperopt.

FastCluster's condensed distance matrices use `float64` in both Python and
Rust, including matrices retained for EPS calibration. This matches streaming
inference and prevents cache rounding from changing merges near the threshold.
Storage is eight bytes per pair, or `8 * n * (n - 1) / 2` bytes for a block of
`n` signatures. Python's stored matrices previously used two bytes per pair;
the allocation guard now budgets the full eight bytes.

When supplying `val_dists_precomputed` or prediction `dists`, generate the
distances at `float64` precision. Recompute any previously rounded `float16`
cache; casting it to `float64` cannot recover the original scores.

## Evaluate clustering

```python
from s2and.eval import cluster_eval

metrics, metrics_per_signature = cluster_eval(dataset, clusterer)
print(metrics)
```

`metrics_per_signature` is useful when you want to slice performance by signature properties.

## Publish and reload a trained model

Do not publish a pickle. The public loader accepts one complete native bundle
containing pairwise boosters, clusterer configuration, promoted linker,
embedded replay target, and checksummed manifest. Its root manifest records
`kind: "s2and_model"`, the plan-derived `release_version`, the exact
`generated_by_runtime`, EPS calibration state, and the file checksum
inventory. Pairwise training writes EPS
`0.5` with calibration state `pending`; validation-only calibration writes a
fresh `calibrated` pairwise sibling. After EPS is frozen,
`train_linker_and_finalize.py` fits the linker once, atomically writes
the complete bundle, reloads those exact bytes, and evaluates them. See the
[v1.3 release runbook](release.md).

Pairwise production training verifies the packaged canonical name tuples and
records `name_tuples_data_sha256`, `name_counts_manifest_sha256`,
and `orcid_prefix_counts_data_sha256` in `feature_contract`. Bundle export does
not synthesize missing behavior hashes. Export and load compare tuple/ORCID
data hashes with the canonical package artifacts; the exact name-count
manifest and complete ordered feature contract are also bound into model and
linker provenance.

The promoted linker's fixed-role metadata records
`kind: "s2and_incremental_linker"`, the exact generating runtime, its booster
checksum, and digests binding the pairwise bundle and complete training target
JSON. The complete bundle keeps that target at
`reproducibility/incremental_linker_training_target.json`; finalization and
loading reject a mismatch. Evaluation starts only after the serialized bundle
has been reloaded.

After a bundle passes those gates, reload it explicitly:

```python
from s2and.arrow_inputs import ArrowDataset
from s2and.production_model import load_production_model

clusterer = load_production_model("/path/to/production_model_vX.Y")
with ArrowDataset.open("/path/to/arrow_dataset") as arrow_dataset:
    pred_clusters, pred_distance_matrices = clusterer.predict_from_arrow(
        blocks,
        arrow_dataset,
        total_ram_bytes=32 * 1024**3,
    )
```

`pred_distance_matrices` may be `None` when the fused clustering path is active.

## Reference scripts

- `scripts/production/model/train_pairwise.py`: pairwise production-bundle stage
- `scripts/production/model/release_pairwise.py`: EPS calibration, measurement
  components, and `evaluate-release` aggregation into one report
- `scripts/production/model/train_linker_and_finalize.py`: one-fit
  complete-bundle finalization, reload, and evaluation
- `scripts/tutorial_for_predicting_with_the_prod_model.py`: released-model inference example
- `scripts/README.md`: script catalog
