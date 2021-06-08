# Transfer table with all 5 seeds + standard deviations. B3 and AUC.
python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --experiment_name paper_experiments/multiseed_full 

# We also need leave self in for the table above  but not individual models 
python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --experiment_name paper_experiments/multiseed_union_leave_self_in  --leave_self_in --skip_individual_models

# Ablation experiments
python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --skip_individual_models --experiment_name paper_experiments/ablations/baseline

python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --skip_individual_models --experiment_name paper_experiments/ablations/no_nameless_model --dont_use_nameless_model

python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --skip_individual_models --experiment_name paper_experiments/ablations/no_monotone_constraints --dont_use_monotone_constraints

python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --skip_individual_models --experiment_name paper_experiments/ablations/linear_pairwise_model --use_linear_pairwise_model

python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --skip_individual_models --experiment_name paper_experiments/ablations/50k_pairwise_training --n_train_pairs_size 50000
python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --skip_individual_models --experiment_name paper_experiments/ablations/10k_pairwise_training --n_train_pairs_size 10000
python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --skip_individual_models --experiment_name paper_experiments/ablations/5k_pairwise_training --n_train_pairs_size 5000
python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --skip_individual_models --experiment_name paper_experiments/ablations/1k_pairwise_training --n_train_pairs_size 1000

python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --skip_individual_models --experiment_name paper_experiments/ablations/dbscan --use_dbscan

python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --skip_individual_models --experiment_name paper_experiments/ablations/linkage_ward --linkage ward
python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --skip_individual_models --experiment_name paper_experiments/ablations/linkage_single --linkage single
python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --skip_individual_models --experiment_name paper_experiments/ablations/linkage_complete --linkage complete

python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --skip_individual_models --experiment_name paper_experiments/ablations/no_specter --feature_groups_to_skip embedding_similarity
python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --skip_individual_models --experiment_name paper_experiments/ablations/no_titles --feature_groups_to_skip title_similarity
python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --skip_individual_models --experiment_name paper_experiments/ablations/no_name_counts --feature_groups_to_skip name_counts
python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --skip_individual_models --experiment_name paper_experiments/ablations/no_advanced_name_similarity --feature_groups_to_skip advanced_name_similarity
python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --skip_individual_models --experiment_name paper_experiments/ablations/no_venue --feature_groups_to_skip venue_similarity journal_similarity
python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --skip_individual_models --experiment_name paper_experiments/ablations/no_email --feature_groups_to_skip email_similarity
python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --skip_individual_models --experiment_name paper_experiments/ablations/no_affiliation --feature_groups_to_skip affiliation_similarity
python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --skip_individual_models --experiment_name paper_experiments/ablations/no_year --feature_groups_to_skip year_diff
python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --skip_individual_models --experiment_name paper_experiments/ablations/no_coauthors --feature_groups_to_skip coauthor_similarity
python scripts/transfer_experiment_seed_paper.py --n_jobs 25 --random_seed 1 2 3 4 5 --skip_individual_models --experiment_name paper_experiments/ablations/no_references --feature_groups_to_skip reference_features

# SOTA
python scripts/sota.py --n_jobs 25 --random_seed 42 --inspire_split 0 --inspire_only --experiment_name paper_experiments_sota/inspire_split_0/
python scripts/sota.py --n_jobs 25 --random_seed 42 --inspire_split 1 --inspire_only --experiment_name paper_experiments_sota/inspire_split_1/
python scripts/sota.py --n_jobs 25 --random_seed 42 --inspire_split 2 --inspire_only --experiment_name paper_experiments_sota/inspire_split_2/
python scripts/sota.py --n_jobs 25 --random_seed 42 --aminer_only --experiment_name paper_experiments_sota/aminer/
python scripts/sota.py --n_jobs 25 --random_seed 1 2 3 4 5 --experiment_name paper_experiments_sota/kisti_pubmed_medline/
