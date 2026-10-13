# Version 1.3 pairwise fit and EPS selection

The run root is `D:/local-unsynced/s2and_v1.3_train_20261012`.
The frozen training source is `922d56195e0c7cd74684cad950b861b21e62e225`.
The model-plan SHA-256 is `529c9dcc3eb657f21bc7396e11efb21e6831d5662ea1ba7e428485a54485e070`.

| Item | Result |
| --- | --- |
| Resources | 20 cores; 128 GiB RAM budget; no paid APIs |
| Search | 50 trials for each pair model; validation AUROC objective |
| Fit pairs | 804,716 train; 60,064 validation |
| Main model | Validation AUROC 0.9936000501; 33 features; 1,600 saved trees |
| Nameless model | Validation AUROC 0.9746198751; 17 features; 238 saved trees |
| Winning tree limits | Main 2,289; nameless 2,286 |
| EPS grid | 101 points from 0.00 through 1.00; step 0.01 |
| Selected EPS | 0.55 |
| Validation B-cubed F1 | Signature-weighted 0.9563495712; dataset-macro 0.9614285714 |
| Cluster score population | 72,649 known validation signatures across seven benchmarks |
| Total job times | Pairwise stage 80.2 minutes; EPS stage 6.1 minutes; both exit zero |

The native and LightGBM predictions match exactly on 512 actual feature rows for each model.
The calibrated bundle preserves both booster hashes. The selected EPS maximizes the recorded validation objective with the smallest-EPS tie rule.
Unknown signatures remain in cluster inference. Scores exclude their fallback labels.

The calibrated pairwise bundle is `stages/calibrated/production_model_v1.3` beneath the run root.
The [October 13 prefix exclusion](orcid_prefix_exclusion_20261013.md) produces the current runtime copy at `stages/calibrated_no14/production_model_v1.3`.
Both boosters, training records, prediction fixtures, and EPS remain unchanged. The original bundle retains the artifact identity from its fit.
Evidence files are `pairwise_output_checks.json`, `reports/eps_calibration_report.json`, `logs/pairwise.run.json`, and `logs/eps.run.json`.
The clean source passes full local CI. The later held-out uncertainty correction passes 55 focused checks and does not change the fit or EPS results.

This is a pairwise-only bundle. Linker feature generation, linker fit, and held-out evaluation remain pending.
Historical metrics use different data and splits. They do not establish improvement for this release.
Final release acceptance policy and S3 publication remain pending.
