import cProfile
import io
import os
import pickle
import pstats
import time


def main() -> None:
    from s2and.data import ANDData
    from s2and.eval import cluster_eval
    from s2and.consts import PROJECT_ROOT_PATH

    n_jobs = 4
    os.environ["OMP_NUM_THREADS"] = f"{n_jobs}"

    data_root = os.path.join(PROJECT_ROOT_PATH, "data", "s2and_mini")
    dataset_name = "kisti"

    with open(os.path.join(PROJECT_ROOT_PATH, "data", "production_model_v1.1.pickle"), "rb") as f:
        clusterer = pickle.load(f)["clusterer"]
        clusterer.use_cache = False
        clusterer.n_jobs = n_jobs

    anddata = ANDData(
        signatures=os.path.join(data_root, dataset_name, dataset_name + "_signatures.json"),
        papers=os.path.join(data_root, dataset_name, dataset_name + "_papers.json"),
        name=dataset_name,
        mode="train",
        specter_embeddings=os.path.join(data_root, dataset_name, dataset_name + "_specter.pickle"),
        clusters=os.path.join(data_root, dataset_name, dataset_name + "_clusters.json"),
        block_type="s2",
        train_pairs=None,
        val_pairs=None,
        test_pairs=None,
        train_pairs_size=100000,
        val_pairs_size=10000,
        test_pairs_size=10000,
        n_jobs=n_jobs,
        load_name_counts=True,
        preprocess=True,
        random_seed=42,
        name_tuples="filtered",
        use_orcid_id=True,
        use_sinonym_overwrite=True,
    )

    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    cluster_metrics, _ = cluster_eval(anddata, clusterer, split="test", use_s2_clusters=False)
    profiler.disable()
    elapsed = time.perf_counter() - start

    print(f"KISTI prod-mode runtime: {elapsed:.2f}s")
    print(cluster_metrics)

    output_path = os.path.join(PROJECT_ROOT_PATH, "scratch", "profile_kisti_rust_prod.txt")
    stats_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream).sort_stats("cumtime")
    stats.print_stats(40)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(stats_stream.getvalue())
        f.write(f"\nTotal runtime: {elapsed:.2f}s\n")
    print(f"Wrote profile to {output_path}")


if __name__ == "__main__":
    main()
