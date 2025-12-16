import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset from the local directory
current_dir = os.path.dirname(os.path.abspath(__file__))
iris_csv_path = os.path.join(current_dir, "iris.csv")

iris_data = pd.read_csv(
    iris_csv_path,
    header=None,
    encoding="utf-8-sig"
)

# Extract feature vectors (columns 1–4) and class labels (column 5)
features = iris_data.iloc[:, 0:4].values.astype(float)
labels = iris_data.iloc[:, 4].values


# Select random data points as initial cluster centroids
def initialize_centroids(data, num_clusters):
    indices = np.random.choice(len(data), num_clusters, replace=False)
    return data[indices]


# Compute Euclidean distance between two points
def euclidean_distance(point_a, point_b):
    diff = point_a - point_b
    return np.sqrt(np.sum(diff ** 2))


# Perform K-Means clustering without using built-in libraries
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

        # Recalculate centroids based on current cluster assignments
        new_centroids = np.zeros_like(centroids)
        for j in range(num_clusters):
            cluster_points = data[cluster_labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[j] = centroids[j]

        # Check if centroid movement is below convergence threshold
        max_shift = 0.0
        for j in range(num_clusters):
            shift = euclidean_distance(centroids[j], new_centroids[j])
            max_shift = max(max_shift, shift)

        if max_shift < tolerance:
            break

        centroids = new_centroids

    return centroids, cluster_labels


# Plot clustering results using selected feature dimensions
def plot_clusters(data, centroids, cluster_labels, title, feature_indices=(2, 3)):
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
        label="Centroids"
    )

    plt.title(title)
    plt.legend()
    plt.show()


# Run K-Means multiple times to observe different initializations
clustering_results = []
for _ in range(5):
    centroids, cluster_labels = k_means_clustering(features, 3)
    clustering_results.append((centroids, cluster_labels))


# Calculate total within-cluster variance for evaluation
def compute_within_cluster_variance(data, centroids, cluster_labels):
    total_variance = 0.0
    for i in range(len(centroids)):
        cluster_points = data[cluster_labels == i]
        diff = cluster_points - centroids[i]
        total_variance += np.sum(diff ** 2)
    return total_variance


# Identify best and worst clustering results
variances = [
    compute_within_cluster_variance(features, ctrs, lbls)
    for ctrs, lbls in clustering_results
]

best_index = np.argmin(variances)
worst_index = np.argmax(variances)

best_centroids, best_labels = clustering_results[best_index]
worst_centroids, worst_labels = clustering_results[worst_index]


# Visualize best and worst clustering outcomes
plot_clusters(features, best_centroids, best_labels,
              title="Iris – Best Clustering", feature_indices=(2, 3))

plot_clusters(features, worst_centroids, worst_labels,
              title="Iris – Worst Clustering", feature_indices=(2, 3))


# Plot original Iris class labels for comparison
plt.figure(figsize=(6, 5))
species_names = ["Setosa", "Versicolor", "Virginica"]

for idx, species in enumerate(iris_data[4].unique()):
    plt.scatter(
        features[labels == species, 2],
        features[labels == species, 3],
        label=species_names[idx]
    )

plt.title("Iris – Original Labels")
plt.legend()
plt.show()


# Compute true class centroids using original labels
true_centroids = np.array([
    features[labels == species].mean(axis=0)
    for species in iris_data[4].unique()
])

# Measure distance between computed centroids and true class centroids
center_distances = np.linalg.norm(
    best_centroids[:, None, :] - true_centroids[None, :, :],
    axis=2
)

print("Distances between best clustering centers and true class centers:")
print(center_distances)
