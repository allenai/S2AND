This folder contains scripts that are a mix of: (a) documentation, (b) internal Semantic Scholar scripts that won't run for anyone outside of AI2, 
(c) experimental scripts for the ablations, and (d) continuous integration scripts.

If you're not internal to AI2, here are scripts you will care about:
- `paper_experiments.sh`: A complete list of command line commands to reproduce all of the paper's results 
- `transfer_experiment_seed_paper.py`: The main script used to run the experiments present in the paper
- `tutorial.ipynb`: A guide to the S2AND pipeline that's easier to look at than the above two scripts.

*Important* notes about `transfer_experiment_seed_paper.py`: 
- It assumes that the S2AND data is in `<code root path>/data/`. If that's not the case, you'll have to modify the `"main_data_dir"` entry in `data/path_config.json`.
- If you have a small to medium amount of RAM, don't use the `--use_cache` flag. Without the cache, it'll be slower, but will not try to fit all of the feature data into memory.

Continuous integration scripts:
- `mypy.sh`: Just runs the mypy part of the continuous integration
- `run_ci_locally.sh`: Runs the CI for the repo locally
