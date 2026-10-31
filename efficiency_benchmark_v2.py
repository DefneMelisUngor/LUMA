
import time
import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

SEEDS = [42, 43, 44]
N_EPISODES = 100
N_WAYS = 5
N_QUERIES = 10
SHOT_COUNTS = [1, 2, 5, 10]

WARMUP_REPEATS = 20
TIMING_REPEATS = 100
MAX_SHOTS = max(SHOT_COUNTS)


def load_data():
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target

    _, X_eval, _, y_eval = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    return X_eval, y_eval


def create_episode(X, y, rng):
    classes = rng.choice(np.unique(y), N_WAYS, replace=False)

    support_X, support_y = [], []
    query_X, query_y = [], []

    for class_id in classes:
        indices = np.where(y == class_id)[0]

        selected = rng.choice(
            indices,
            MAX_SHOTS + N_QUERIES,
            replace=False
        )

        support_X.extend(X[selected[:MAX_SHOTS]])
        support_y.extend([class_id] * MAX_SHOTS)

        query_X.extend(X[selected[MAX_SHOTS:]])
        query_y.extend([class_id] * N_QUERIES)

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
        support_y.extend([class_id] * shots)

    return np.array(support_X), np.array(support_y)


def knn_fit(sx, sy):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(sx, sy)
    return model


def knn_predict(model, qx):
    return model.predict(qx)


def proto_fit(sx, sy):
    classes = np.unique(sy)

    prototypes = np.array([
        sx[sy == class_id].mean(axis=0)
        for class_id in classes
    ])

    return classes, prototypes


def proto_predict(model, qx):
    classes, prototypes = model

    distances = np.linalg.norm(
        qx[:, None, :] - prototypes[None, :, :],
        axis=2
    )

    return classes[np.argmin(distances, axis=1)]


def measure(function, repeats):
    durations = []

    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = function()
        end = time.perf_counter_ns()

        durations.append((end - start) / 1_000_000)

    return result, np.array(durations)


def benchmark_episode(sx, sy, qx, qy):
    # Warm-up
    for _ in range(WARMUP_REPEATS):
        knn = knn_fit(sx, sy)
        knn_predict(knn, qx)

        proto = proto_fit(sx, sy)
        proto_predict(proto, qx)

    # Repeated KNN fit
    knn_model, knn_fit_times = measure(
        lambda: knn_fit(sx, sy),
        TIMING_REPEATS
    )

    # Repeated KNN prediction
    knn_predictions, knn_predict_times = measure(
        lambda: knn_predict(knn_model, qx),
        TIMING_REPEATS
    )

    # Repeated prototype construction
    proto_model, proto_fit_times = measure(
        lambda: proto_fit(sx, sy),
        TIMING_REPEATS
    )

    # Repeated prototype prediction
    proto_predictions, proto_predict_times = measure(
        lambda: proto_predict(proto_model, qx),
        TIMING_REPEATS
    )

    return {
        "knn_accuracy": np.mean(knn_predictions == qy),
        "knn_fit_median_ms": np.median(knn_fit_times),
        "knn_fit_std_ms": np.std(knn_fit_times),
        "knn_predict_median_ms": np.median(knn_predict_times),
        "knn_predict_std_ms": np.std(knn_predict_times),
        "proto_accuracy": np.mean(proto_predictions == qy),
        "proto_fit_median_ms": np.median(proto_fit_times),
        "proto_fit_std_ms": np.std(proto_fit_times),
        "proto_predict_median_ms": np.median(proto_predict_times),
        "proto_predict_std_ms": np.std(proto_predict_times)
    }


def main():
    X, y = load_data()
    results = []

    for seed in SEEDS:
        rng = np.random.default_rng(seed)

        for episode in range(N_EPISODES):
            sx_pool, sy_pool, qx, qy = create_episode(X, y, rng)

            for shots in SHOT_COUNTS:
                sx, sy = prepare_support(
                    sx_pool, sy_pool, shots
                )

                metrics = benchmark_episode(sx, sy, qx, qy)

                results.append({
                    "seed": seed,
                    "episode": episode + 1,
                    "shots": shots,
                    **metrics
                })

        print(f"Completed seed: {seed}")

    df = pd.DataFrame(results)

    df.to_csv(
        "results/day04_efficiency_benchmark_v2.csv",
        index=False
    )

    metric_columns = [
        column for column in df.columns
        if column not in ["seed", "episode", "shots"]
    ]

    summary = df.groupby("shots")[metric_columns].agg(
        ["mean", "median", "std"]
    ).round(5)

    print("\nLUMA — Day 04 Benchmark v2")
    print("--------------------------------")
    print(summary.to_string())

    print("\nSaved: results/day04_efficiency_benchmark_v2.csv")


if __name__ == "__main__":
    main()
