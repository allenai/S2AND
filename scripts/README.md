This folder contains scripts that are a mix of: (a) documentation, (b) internal Semantic Scholar scripts that won't run for anyone outside of AI2, 
(c) experimental scripts for the S2AND paper, and (d) continuous integration scripts.

If you're not internal to AI2, here are scripts you will care about:
- `paper_experiments.sh`: A complete list of command line commands to reproduce all of the paper's results 
- `transfer_experiment_seed_paper.py`: The main script used to run the experiments present in the paper
- `tutorial.ipynb`: A guide to the S2AND pipeline that's easier to look at than the above two scripts.

*Important* notes about `transfer_experiment_seed_paper.py`: 
- It assumes that the S2AND data is in `<code root path>/data/`. If that's not the case, you'll have to modify the `"main_data_dir"` entry in `data/path_config.json`.
- If you have a small to medium amount of RAM, don't use the `--use_cache` flag. Without the cache, it'll be slower, but will not try to fit all of the feature data into memory.

Other scripts in this folder (mostly have `use_cache=True`):
- `full_model_dump.py`: Trains and dumps to disk a full model trained on all of the datasets (including orcid and augmented, which are not released)
- `get_name_counts.py`: Present as documentation for how the name counts metadata was collected (not runnable because it relies on internal Semantic Scholar data)
- `make_augmentation_dataset_a.py`: First step of creating the augmentation dataset (data not released)
- `make_augmentation_dataset_b.py`: Second step of creating the augmentation dataset (data not released)
- `make_s2and_mini_dataset.py`: S2AND is huge and takes a long time. If you want to make a smaller dataset, this script will do it. It skips medline.

Continuous integration scripts:
- `mypy.sh`: Just runs the mypy part of the continuous integration
- `run_ci_locally.sh`: Runs the CI for the repo locally
