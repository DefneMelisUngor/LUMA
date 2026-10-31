
import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


SEED = 42
N_EPISODES = 100
N_WAYS = 5
N_SHOTS = 5
N_QUERIES = 10

rng = np.random.default_rng(SEED)


def load_data():
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=SEED,
        stratify=y
    )

    return X_test, y_test


def run_episode(X, y):
    classes = rng.choice(np.unique(y), size=N_WAYS, replace=False)

    support_X, support_y = [], []
    query_X, query_y = [], []

    for class_id in classes:
        indices = np.where(y == class_id)[0]
        selected = rng.choice(
            indices,
            size=N_SHOTS + N_QUERIES,
            replace=False
        )

        support_indices = selected[:N_SHOTS]
        query_indices = selected[N_SHOTS:]

        support_X.extend(X[support_indices])
        support_y.extend([class_id] * N_SHOTS)

        query_X.extend(X[query_indices])
        query_y.extend([class_id] * N_QUERIES)

    return (
        np.array(support_X),
        np.array(support_y),
        np.array(query_X),
        np.array(query_y)
    )


def evaluate():
    X, y = load_data()
    results = []

    for episode in range(N_EPISODES):
        sx, sy, qx, qy = run_episode(X, y)

        # Method 1: Nearest-neighbor classification
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(sx, sy)
        knn_accuracy = knn.score(qx, qy)

        # Method 2: Prototypical classification
        prototypes = np.array([
            sx[sy == class_id].mean(axis=0)
            for class_id in np.unique(sy)
        ])

        distances = np.linalg.norm(
            qx[:, None, :] - prototypes[None, :, :],
            axis=2
        )

        predicted = np.unique(sy)[np.argmin(distances, axis=1)]
        prototype_accuracy = np.mean(predicted == qy)

        results.append({
            "episode": episode + 1,
            "knn_accuracy": knn_accuracy,
            "prototype_accuracy": prototype_accuracy
        })

    df = pd.DataFrame(results)

    df.to_csv("results/day01_baseline.csv", index=False)

    print("\nLUMA — Day 01 Baseline Results")
    print("--------------------------------")
    print(f"Episodes: {N_EPISODES}")
    print(f"Nearest Neighbor: {df['knn_accuracy'].mean():.4f}")
    print(f"Prototypical:     {df['prototype_accuracy'].mean():.4f}")
    print("\nSaved to results/day01_baseline.csv")


if __name__ == "__main__":
    evaluate()
