import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load kmtest dataset from the local directory
current_dir = os.path.dirname(os.path.abspath(__file__))
kmtest_csv_path = os.path.join(current_dir, "kmtest.csv")

kmtest_data = pd.read_csv(
    kmtest_csv_path,
    header=None,
    delim_whitespace=True,
    encoding="utf-8-sig"
)

# Convert dataset to numerical feature array
features = kmtest_data.values.astype(float)


# Apply z-score normalization to the dataset
def z_score_normalize(data):
    return (data - data.mean(axis=0)) / data.std(axis=0)


# Randomly select initial centroids from the dataset
def initialize_centroids(data, num_clusters):
    indices = np.random.choice(len(data), num_clusters, replace=False)
    return data[indices]


# Compute Euclidean distance between two points
def euclidean_distance(point_a, point_b):
    diff = point_a - point_b
    return np.sqrt(np.sum(diff ** 2))


# Perform K-Means clustering without built-in libraries
def k_means_clustering(data, num_clusters, max_iterations=100, tolerance=1e-4):
    centroids = initialize_centroids(data, num_clusters)

    for _ in range(max_iterations):
        # Assign each data point to the nearest centroid
        cluster_labels = np.zeros(data.shape[0], dtype=int)
        for i in range(data.shape[0]):
            min_distance = float("inf")
            for j in range(num_clusters):
                dist = euclidean_distance(data[i], centroids[j])
                if dist < min_distance:
                    min_distance = dist
                    cluster_labels[i] = j

        # Recalculate centroids based on cluster assignments
        new_centroids = np.zeros_like(centroids)
        for j in range(num_clusters):
            cluster_points = data[cluster_labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[j] = centroids[j]

        # Check for convergence using centroid movement
        max_shift = 0.0
        for j in range(num_clusters):
            shift = euclidean_distance(centroids[j], new_centroids[j])
            max_shift = max(max_shift, shift)

        if max_shift < tolerance:
            break

        centroids = new_centroids

    return centroids, cluster_labels


# Plot clustering results for selected dimensions
def plot_clusters(data, centroids, cluster_labels, title, feature_indices=(0, 1)):
    plt.figure(figsize=(6, 5))

    # Plot each cluster with a different color
    for i in range(len(centroids)):
        cluster_points = data[cluster_labels == i]
        plt.scatter(
            cluster_points[:, feature_indices[0]],
            cluster_points[:, feature_indices[1]],
            label=f"Cluster {i + 1}"
        )

    # Plot cluster centroids
    plt.scatter(
        centroids[:, feature_indices[0]],
        centroids[:, feature_indices[1]],
        c="black",
        marker="x",
        s=100,
        label="Centers"
    )

    plt.title(title)
    plt.legend()
    plt.show()


# Run K-Means on original data without normalization
for k in [2, 3, 4, 5]:
    centroids, cluster_labels = k_means_clustering(features, k)
    plot_clusters(
        features,
        centroids,
        cluster_labels,
        title=f"kmtest – K={k} (No Normalization)"
    )


# Normalize data using z-score normalization
normalized_features = z_score_normalize(pd.DataFrame(features)).values


# Run K-Means on normalized data
for k in [2, 3, 4, 5]:
    centroids, cluster_labels = k_means_clustering(normalized_features, k)
    plot_clusters(
        normalized_features,
        centroids,
        cluster_labels,
        title=f"kmtest – K={k} (Z-Score Normalized)"
    )
