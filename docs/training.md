# Training and Evaluation

This guide describes research training and evaluation through the API.
It uses the published benchmark data and the name-count index from the inference bundle. See [where to download both bundles](data.md#download).
The examples expose test metrics for direct research inspection.
Production release selection uses development data and keeps candidate test scores separate until the declared release evaluation.
See [the production commands](../scripts/release/README.md#pairwise-model) for the maintained production trainer.
See [the reproduction guide](reproduce.md) for the production training steps on published data.
See [the release process](release.md) for model acceptance and publication.

## Build a Rust-backed training dataset

The maintained training constructor consumes an open, manifest-backed
`ArrowDataset`. Ground-truth clusters remain JSON by design, but signatures,
papers, paper authors, embeddings, batch indexes, and name counts come from the
same corrected snapshot and common index.

Set `bundle_dir` to a benchmark Arrow directory, such as `s2and/data/benchmarks/pubmed`.

```python
import json
from pathlib import Path

from s2and.arrow_inputs import ArrowDataset
from s2and.arrow_training import build_training_anddata_from_arrow

bundle_dir = Path("s2and/data/benchmarks/pubmed")
manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
manifest_paths = manifest["paths"]
arrow_dataset = ArrowDataset.open(
    bundle_dir,
    require_specter=True,
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

Opening the handle validates required tables, raw-planner batch indexes,
checksums, public format `1`, and the name-count index before any pairs are sampled.
Keep `arrow_dataset` open while using the returned dataset, then close it.
The constructor selects the Rust training runtime and pins canonical preprocessing.
The requested pair counts are upper bounds when a split contains fewer eligible within-block pairs.
The returned dataset's Python-visible signatures and papers are reconstructed
from the validated Arrow bundle; the constructor never injects pre-conversion
source objects. Keep source and reconstructed datasets separate in parity
checks.

Pass `use_orcid_id=False` when a benchmark must remove ORCID evidence. The
policy applies to both reconstructed signatures and the native Rust
featurizer without rewriting the immutable Arrow bundle.

## Featurize pairs and train the pairwise model

`s2and/feature_schema.py` defines the ordered feature names, groups, and
monotonic constraints. Python derives selection metadata from that specification;
Rust writes calculated values to its generated named column constants. The
pairwise model uses 33 feature columns; the separate 53-feature target in
`tests/fixtures/incremental_linker_training_target.json` belongs to the
incremental linker.

After a deliberate feature-contract change, regenerate native constants with
`uv run --no-project python scripts/dev/sync_feature_schema.py`. Shared local and
hosted CI run the script with `--check` to reject stale generated code. Feature
calculations remain independent across languages and are checked by parity tests.

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

The production `train_pairwise.py` command featurizes from the Arrow data that its
`model_plan.json` names; programmatic research callers use `featurize(...)` directly.

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
Linkage treats each pair at exactly `LARGE_DISTANCE` as a hard negative: the pair
never joins one cluster below EPS in any block that fits in memory. These pairs include
explicit disallows, caller hard disallows, pairs in different seed groups, and
name-rule conflicts. Other distances keep their averaged meaning. Seed
restoration after linkage enforces only explicit disallows.

Omitting `search_space` leaves `clusterer.search_space` as `None` during
construction and inference. `fit()` creates the same uniform EPS space shown
above when calibration starts. Custom calibration code can explicitly call
`s2and.calibration.default_cluster_search_space()`; supplied search spaces are
used as given. Loading a fitted production bundle does not initialize
Hyperopt.

FastCluster's condensed distance matrices use `float64` in both Python and
Rust, including matrices retained for EPS calibration. This matches streaming
inference and prevents cache rounding from changing merges near the threshold.
Storage is eight bytes per pair, or `8 * n * (n - 1) / 2` bytes for a block of
`n` signatures, and the allocation guard budgets the full eight bytes.

When supplying `val_dists_precomputed` or prediction `dists`, generate the
distances at `float64` precision; casting lower-precision values to `float64`
cannot recover the original scores.

Pairwise training, `Clusterer.fit()`, and clustering evaluation resolve the
same signature splits: explicit block lists take precedence over explicit
signature lists, followed by the configured block, signature, or time split.
Recompute validation distance matrices whenever the selected signatures or
their order change: the precomputed matrix mapping does not record signature
identities and cannot detect an obsolete matrix from its dimensions.

## Evaluate clustering

```python
from s2and.eval import cluster_eval

metrics, metrics_per_signature = cluster_eval(dataset, clusterer)
print(metrics)
```

`metrics_per_signature` is useful when you want to slice performance by signature properties.

`cluster_eval()` predicts and scores the selected split. In
`incremental_cluster_eval()`, validation prediction also includes observed
training signatures, and test prediction includes observed training and
validation signatures in the tested blocks. Only signatures belonging to the
requested evaluation split contribute per-signature scores. Records outside
explicit split lists are excluded from prediction context. B-cubed (B3) uses
the prediction context, and pairwise metrics use evaluation-only memberships.

## Publish and reload a trained model

The public loader requires a complete native bundle; it rejects pickles and
pairwise-only training stages. The research example above trains the pairwise
model and clusterer, but a publishable bundle also needs the promoted linker
and validated release artifacts. See the
[bundle contract](production_inference.md#complete-model-bundles) for required
files, feature provenance, checksums, and staged publication. Follow the
[release process](release.md) for pairwise training, validation-only EPS
selection, linker finalization, and evaluation through the reloaded bundle.

See [production linker training and evaluation](../scripts/release/README.md#complete-model-and-linker-evaluation) for the training splits and reports.
After a complete bundle passes the acceptance checks, [load the model and run inference](production_inference.md#load-a-production-model).

## Reference scripts

- `scripts/pairwise/train_pairwise.py`: pairwise production-bundle stage
- `scripts/pairwise/release_pairwise.py`: EPS calibration and outer-test
  evaluation
- `scripts/linker/train_linker_and_finalize.py`: one-fit
  complete-bundle finalization, reload, and evaluation
- `scripts/tutorial_for_predicting_with_the_prod_model.py`: complete-model
  prediction example on one Arrow or JSON dataset
- `scripts/README.md`: script catalog
