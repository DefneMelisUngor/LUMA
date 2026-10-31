
import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Experiment configuration
SEEDS = [42, 43, 44]
N_EPISODES = 100
N_WAYS = 5
N_QUERIES = 10
SHOT_COUNTS = [1, 2, 5, 10]
MAX_SHOTS = max(SHOT_COUNTS)


def load_data():
    digits = load_digits()

    X = digits.data.astype(np.float32) / 16.0
    y = digits.target

    _, X_eval, _, y_eval = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    return X_eval, y_eval


def create_episode(X, y, rng):
    classes = rng.choice(
        np.unique(y),
        size=N_WAYS,
        replace=False
    )

    support_pool = []
    support_labels = []
    query_X = []
    query_y = []

    for class_id in classes:
        indices = np.where(y == class_id)[0]

        selected = rng.choice(
            indices,
            size=MAX_SHOTS + N_QUERIES,
            replace=False
        )

        support_pool.extend(X[selected[:MAX_SHOTS]])
        support_labels.extend([class_id] * MAX_SHOTS)

        query_X.extend(X[selected[MAX_SHOTS:]])
        query_y.extend([class_id] * N_QUERIES)

    return (
        np.array(support_pool),
        np.array(support_labels),
        np.array(query_X),
        np.array(query_y)
    )


def evaluate_episode(sx, sy, qx, qy):
    # Nearest Neighbor
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(sx, sy)
    knn_accuracy = knn.score(qx, qy)

    # Prototypical Classification
    classes = np.unique(sy)

    prototypes = np.array([
        sx[sy == class_id].mean(axis=0)
        for class_id in classes
    ])

    distances = np.linalg.norm(
        qx[:, None, :] - prototypes[None, :, :],
        axis=2
    )

    predictions = classes[np.argmin(distances, axis=1)]
    prototype_accuracy = np.mean(predictions == qy)

    return knn_accuracy, prototype_accuracy


def run_experiment():
    X, y = load_data()
    all_results = []

    for seed in SEEDS:
        rng = np.random.default_rng(seed)

        for episode in range(N_EPISODES):
            sx_pool, sy_pool, qx, qy = create_episode(
                X, y, rng
            )

            for shots in SHOT_COUNTS:
                sx = []
                sy = []

                for class_id in np.unique(sy_pool):
                    class_indices = np.where(
                        sy_pool == class_id
                    )[0]

                    chosen = class_indices[:shots]

                    sx.extend(sx_pool[chosen])
                    sy.extend([class_id] * shots)

                sx = np.array(sx)
                sy = np.array(sy)

                knn_acc, proto_acc = evaluate_episode(
                    sx, sy, qx, qy
                )

                all_results.append({
                    "seed": seed,
                    "episode": episode + 1,
                    "shots": shots,
                    "knn_accuracy": knn_acc,
                    "prototype_accuracy": proto_acc
                })

        print(f"Completed seed: {seed}")

    df = pd.DataFrame(all_results)

    df.to_csv(
        "results/day02_support_experiment.csv",
        index=False
    )

    summary = df.groupby("shots")[
        ["knn_accuracy", "prototype_accuracy"]
    ].agg(["mean", "std"])

    print("\nLUMA — Day 02 Support Experiment")
    print("--------------------------------")
    print(summary.round(4))

    print("\nSaved to results/day02_support_experiment.csv")


if __name__ == "__main__":
    run_experiment()
