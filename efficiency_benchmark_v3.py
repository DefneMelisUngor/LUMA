import time
import numpy as np
import pandas as pd

from sklearn.datasets import load_digits

# Configuration
SEEDS = [42, 43, 44]
N_EPISODES = 100
N_TIMING_EPISODES = 3
N_WAYS = 5
N_QUERIES = 10
SHOT_COUNTS = [1, 2, 5, 10, 50, 100]

# Timing Configuration
WARMUP_REPEATS = 20
N_BATCHES = 15
SAMPLES_PER_BATCH = 1000
MAX_SHOTS = max(SHOT_COUNTS)

def load_data():
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target
    return X, y

def create_episode(X, y, rng):
    classes = rng.choice(np.unique(y), N_WAYS, replace=False)
    support_X, support_y = [], []
    query_X, query_y = [], []

    for class_id in classes:
        indices = np.where(y == class_id)[0]
        # Veri setinde mevcut olan örnek sayısını kontrol et
        available = len(indices)
        sample_size = min(available, MAX_SHOTS + N_QUERIES)
        
        selected = rng.choice(
            indices,
            sample_size,
            replace=False
        )
        
        # Önce sorgu örneklerini ayır, kalanı destek seti yap (adil benchmark için)
        actual_query_count = min(N_QUERIES, sample_size)
        actual_support_count = sample_size - actual_query_count
        
        support_X.extend(X[selected[:actual_support_count]])
        support_y.extend([class_id] * actual_support_count)
        query_X.extend(X[selected[actual_support_count:]])
        query_y.extend([class_id] * actual_query_count)

    return (
        np.array(support_X),
        np.array(support_y),
        np.array(query_X),
        np.array(query_y)
    )

def prepare_support(sx_pool, sy_pool, shots):
    support_X, support_y = [], []
    for class_id in np.unique(sy_pool):
        indices = np.where(sy_pool == class_id)[0][:shots]
        support_X.extend(sx_pool[indices])
        support_y.extend([class_id] * len(indices))
    return np.array(support_X), np.array(support_y)

def numpy_knn_fit(sx, sy):
    return sx, sy

def numpy_knn_predict(model, qx):
    sx, sy = model
    dists = np.linalg.norm(qx[:, None, :] - sx[None, :, :], axis=2)
    return sy[np.argmin(dists, axis=1)]

def numpy_proto_fit(sx, sy):
    classes = np.unique(sy)
    prototypes = np.array([
        sx[sy == class_id].mean(axis=0)
        for class_id in classes
    ])
    return classes, prototypes

def numpy_proto_predict(model, qx):
    classes, prototypes = model
    dists = np.linalg.norm(qx[:, None, :] - prototypes[None, :, :], axis=2)
    return classes[np.argmin(dists, axis=1)]

def measure_batched(func, args, repeats, batch_size):
    batch_durations = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(batch_size):
            _ = func(*args)
        end = time.perf_counter_ns()
        duration_per_call = ((end - start) / batch_size) / 1_000_000
        batch_durations.append(duration_per_call)
    return np.array(batch_durations)

def measure_single(func, args, repeats):
    durations = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        _ = func(*args)
        end = time.perf_counter_ns()
        durations.append((end - start) / 1_000_000)
    return np.array(durations)

def get_iqr(data):
    return np.percentile(data, 75) - np.percentile(data, 25)

def benchmark_episode(sx, sy, qx, qy, do_timing=True):
    for _ in range(WARMUP_REPEATS):
        knn_m = numpy_knn_fit(sx, sy)
        numpy_knn_predict(knn_m, qx)
        proto_m = numpy_proto_fit(sx, sy)
        numpy_proto_predict(proto_m, qx)

    results = {}
    knn_m = numpy_knn_fit(sx, sy)
    proto_m = numpy_proto_fit(sx, sy)
    
    if len(qy) > 0:
        results["knn_accuracy"] = np.mean(numpy_knn_predict(knn_m, qx) == qy)
        results["proto_accuracy"] = np.mean(numpy_proto_predict(proto_m, qx) == qy)
    else:
        results["knn_accuracy"] = np.nan
        results["proto_accuracy"] = np.nan

    if do_timing:
        knn_fit_times = measure_single(numpy_knn_fit, (sx, sy), N_BATCHES)
        knn_predict_times = measure_batched(numpy_knn_predict, (knn_m, qx), N_BATCHES, SAMPLES_PER_BATCH)
        proto_fit_times = measure_single(numpy_proto_fit, (sx, sy), N_BATCHES)
        proto_predict_times = measure_batched(numpy_proto_predict, (proto_m, qx), N_BATCHES, SAMPLES_PER_BATCH)

        results.update({
            "knn_fit_median_ms": np.median(knn_fit_times),
            "knn_fit_iqr_ms": get_iqr(knn_fit_times),
            "knn_predict_median_ms": np.median(knn_predict_times),
            "knn_predict_iqr_ms": get_iqr(knn_predict_times),
            "proto_fit_median_ms": np.median(proto_fit_times),
            "proto_fit_iqr_ms": get_iqr(proto_fit_times),
            "proto_predict_median_ms": np.median(proto_predict_times),
            "proto_predict_iqr_ms": get_iqr(proto_predict_times)
        })
    else:
        for col in ["knn_fit_median_ms", "knn_fit_iqr_ms", "knn_predict_median_ms", "knn_predict_iqr_ms",
                    "proto_fit_median_ms", "proto_fit_iqr_ms", "proto_predict_median_ms", "proto_predict_iqr_ms"]:
            results[col] = np.nan

    return results

def main():
    X, y = load_data()
    results = []

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for episode in range(N_EPISODES):
            sx_pool, sy_pool, qx, qy = create_episode(X, y, rng)
            do_timing = (episode < N_TIMING_EPISODES)
            for shots in SHOT_COUNTS:
                sx, sy = prepare_support(sx_pool, sy_pool, shots)
                metrics = benchmark_episode(sx, sy, qx, qy, do_timing=do_timing)
                results.append({
                    "seed": seed,
                    "episode": episode + 1,
                    "shots": shots,
                    **metrics
                })
        print(f"Completed seed: {seed}")

    df = pd.DataFrame(results)
    df.to_csv("/Users/defnemelisungor/LUMA/results/day05_efficiency_benchmark_v3.csv", index=False)

    metric_columns = [
        col for col in df.columns if col not in ["seed", "episode", "shots"]
    ]
    summary_data = {}
    for col in metric_columns:
        summary_data[f"{col}_median"] = df.groupby("shots")[col].median()

    summary_df = pd.DataFrame(summary_data).round(5)
    print("\nLUMA — Day 05 Benchmark v3 (NumPy-based, Stress Test)")
    print("--------------------------------")
    print(summary_df.to_string())
    print("\nSaved: /Users/defnemelisungor/LUMA/results/day05_efficiency_benchmark_v3.csv")

if __name__ == "__main__":
    main()