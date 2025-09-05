This folder contains scripts that are a mix of: (a) documentation, (b) internal Semantic Scholar scripts that won't run for anyone outside of AI2, 
(c) experimental scripts for the S2AND paper, and (d) continuous integration scripts.

If you're not internal to AI2, here are scripts you will care about:
- `paper_experiments.sh`: A complete list of command line commands to reproduce all of the paper's results 
- `sota.py`: Scripts to compute the state-of-the-art results table in the paper
- `transfer_experiment_seed_paper.py`: The main script used to run the experiments present in the paper
- `tutorial_for_predicting_with_the_prod_model.py`: A guide to using the released production model to make predictions on your own data.
- `tutorial.ipynb`: A guide to the S2AND pipeline that's easier to look at than the above two scripts.

*Important* notes about `transfer_experiment_seed_paper.py`: 
- It assumes that the S2AND data is in `<code root path>/data/`. If that's not the case, you'll have to modify the `"main_data_dir"` entry in `data/path_config.json`.
- If you have a small to medium amount of RAM, don't use the `--use_cache` flag. Without the cache, it'll be slower, but will not try to fit all of the feature data into memory.

Other scripts in this folder (mostly have `use_cache=True`):
- `blog_post_eval.py`: Computes min edit distance performance numbers that appear only in the blog post.
- `claims_cluster_eval.py`: Evaluates a model on the Semantic Scholar corrections data (data not released)
- `full_model_dump.py`: Trains and dumps to disk a full model trained on all of the datasets (including orcid and augmented, which are not released)
- `get_orcid_name_prefix_counts.py`: Present as documentation for how the orcid name prefix counts metadata was collected (not runnable because it relies on internal Semantic Scholar data)
- `get_name_counts.py`: Present as documentation for how the name counts metadata was collected (not runnable because it relies on internal Semantic Scholar data)
- `LLM_based_filtering_of_name_tuples.py`: Present as documentation for how the name tuples were filtered using gemini-2.5-pro (runnable, if you want to re-spend the money)
- `make_augmentation_dataset_a.py`: First step of creating the augmentation dataset (data not released)
- `make_augmentation_dataset_b.py`: Second step of creating the augmentation dataset (data not released)
- `make_claims_dataset.py`: Creates datasets for evaluating a model on Semantic Scholar corrections data (not runnable because it relies on internal Semantic Scholar data)
- `make_s2and_name_tuples.py`: Creates the name tuples file of known aliases (included as documentation)
- `make_s2and_mini_dataset.py`: S2AND is huge and takes a long time. If you want to make a smaller dataset, this script will do it. It skips medline.
- `transfer_experiment_internal.py`: A version of `transfer_experiment_seed_paper.py` for internal S2 use (has two unreleased datasets)
- `transform_all_datasets.py`: Transforms an old format of the datasets into the final one (probably not relevant to you)

Continuous integration scripts:
- `run_ci_locally.py`: Runs the CI for the repo locally
