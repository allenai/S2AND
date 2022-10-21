# Ablation experiments
python scripts/paper_experiments.py --n_jobs 25 --random_seed 1 2 3 4 5 --experiment_name ablations/baseline --use_cache

python scripts/paper_experiments.py --n_jobs 25 --random_seed 1 2 3 4 5 --experiment_name ablations/no_monotone_constraints --dont_use_monotone_constraints --use_cache

python scripts/paper_experiments.py --n_jobs 25 --random_seed 1 2 3 4 5 --experiment_name ablations/50k_pairwise_training --n_train_pairs_size 50000 --use_cache
python scripts/paper_experiments.py --n_jobs 25 --random_seed 1 2 3 4 5 --experiment_name ablations/10k_pairwise_training --n_train_pairs_size 10000 --use_cache
python scripts/paper_experiments.py --n_jobs 25 --random_seed 1 2 3 4 5 --experiment_name ablations/5k_pairwise_training --n_train_pairs_size 5000 --use_cache
python scripts/paper_experiments.py --n_jobs 25 --random_seed 1 2 3 4 5 --experiment_name ablations/1k_pairwise_training --n_train_pairs_size 1000 --use_cache

python scripts/paper_experiments.py --n_jobs 25 --random_seed 1 2 3 4 5 --experiment_name ablations/dbscan --use_dbscan --use_cache

python scripts/paper_experiments.py --n_jobs 25 --random_seed 1 2 3 4 5 --experiment_name ablations/linkage_ward --linkage ward --use_cache
python scripts/paper_experiments.py --n_jobs 25 --random_seed 1 2 3 4 5 --experiment_name ablations/linkage_single --linkage single --use_cache
python scripts/paper_experiments.py --n_jobs 25 --random_seed 1 2 3 4 5 --experiment_name ablations/linkage_complete --linkage complete --use_cache

python scripts/paper_experiments.py --n_jobs 25 --random_seed 1 2 3 4 5 --experiment_name ablations/no_authors --feature_groups_to_skip author_similarity --use_cache
python scripts/paper_experiments.py --n_jobs 25 --random_seed 1 2 3 4 5 --experiment_name ablations/no_venue --feature_groups_to_skip venue_similarity --use_cache
python scripts/paper_experiments.py --n_jobs 25 --random_seed 1 2 3 4 5 --experiment_name ablations/no_year --feature_groups_to_skip year_diff --use_cache
python scripts/paper_experiments.py --n_jobs 25 --random_seed 1 2 3 4 5 --experiment_name ablations/no_title --feature_groups_to_skip title_similarity --use_cache
python scripts/paper_experiments.py --n_jobs 25 --random_seed 1 2 3 4 5 --experiment_name ablations/no_abstract --feature_groups_to_skip abstract_similarity --use_cache
python scripts/paper_experiments.py --n_jobs 25 --random_seed 1 2 3 4 5 --experiment_name ablations/no_paper_quality --feature_groups_to_skip paper_quality --use_cache