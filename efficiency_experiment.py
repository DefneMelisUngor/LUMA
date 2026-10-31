
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
WARMUP = 5


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


MAX_SHOTS = max(SHOT_COUNTS)


def evaluate_episode(sx, sy, qx, qy):
    # KNN: fit/adaptation time
    start = time.perf_counter()

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(sx, sy)

    knn_fit_time = time.perf_counter() - start

    # KNN: prediction time
    start = time.perf_counter()
    knn_predictions = knn.predict(qx)
    knn_predict_time = time.perf_counter() - start

    knn_accuracy = np.mean(knn_predictions == qy)

    # Prototypical: prototype computation time
    start = time.perf_counter()

    classes = np.unique(sy)
    prototypes = np.array([
        sx[sy == class_id].mean(axis=0)
        for class_id in classes
    ])

    proto_fit_time = time.perf_counter() - start

    # Prototypical: prediction time
    start = time.perf_counter()

    distances = np.linalg.norm(
        qx[:, None, :] - prototypes[None, :, :],
        axis=2
    )

    proto_predictions = classes[np.argmin(distances, axis=1)]

    proto_predict_time = time.perf_counter() - start

    proto_accuracy = np.mean(proto_predictions == qy)

    return {
        "knn_accuracy": knn_accuracy,
        "knn_fit_ms": knn_fit_time * 1000,
        "knn_predict_ms": knn_predict_time * 1000,
        "proto_accuracy": proto_accuracy,
        "proto_fit_ms": proto_fit_time * 1000,
        "proto_predict_ms": proto_predict_time * 1000
    }


def main():
    X, y = load_data()
    rng_results = []

    # Warm-up to reduce first-call overhead
    rng = np.random.default_rng(999)
    sx, sy, qx, qy = create_episode(X, y, rng)

    for _ in range(WARMUP):
        evaluate_episode(sx[:N_WAYS], sy[:N_WAYS], qx, qy)

    for seed in SEEDS:
        rng = np.random.default_rng(seed)

        for episode in range(N_EPISODES):
            sx_pool, sy_pool, qx, qy = create_episode(X, y, rng)

            for shots in SHOT_COUNTS:
                sx, sy = [], []

                for class_id in np.unique(sy_pool):
                    indices = np.where(sy_pool == class_id)[0][:shots]
                    sx.extend(sx_pool[indices])
                    sy.extend([class_id] * shots)

                result = evaluate_episode(
                    np.array(sx),
                    np.array(sy),
                    qx,
                    qy
                )

                rng_results.append({
                    "seed": seed,
                    "episode": episode + 1,
                    "shots": shots,
                    **result
                })

        print(f"Completed seed: {seed}")

    df = pd.DataFrame(rng_results)

    df.to_csv(
        "results/day03_efficiency.csv",
        index=False
    )

    metrics = [
        "knn_accuracy",
        "knn_fit_ms",
        "knn_predict_ms",
        "proto_accuracy",
        "proto_fit_ms",
        "proto_predict_ms"
    ]

    summary = df.groupby("shots")[metrics].mean().round(4)

    print("\nLUMA — Day 03 Efficiency Results")
    print("--------------------------------")
    print(summary)

    print("\nSaved: results/day03_efficiency.csv")


if __name__ == "__main__":
    main()
