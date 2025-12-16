# K-Means Clustering Implementation

## Description

This project provides a Python implementation of the K-Means clustering algorithm from scratch. K-Means is a popular unsupervised learning method that groups similar data points together while separating dissimilar ones. This repository allows you to experiment with clustering on multiple datasets and explore the impact of data normalization on clustering performance.
Two datasets are included:
kmtest.csv – a sample dataset for basic clustering experiments.
iris.csv – the classic Iris dataset for clustering analysis without using class labels.
Users can visualize clustering results, compare outcomes with and without normalization, and analyze clustering behavior using different attributes and cluster counts.

## Features

🎯 **K-Means from Scratch**
Implements the K-Means algorithm without using built-in clustering functions.
📊 **Dataset Support**
Supports clustering on kmtest and iris datasets.
⚖️ **Normalization Comparison**
Includes experiments with z-score normalization to show how scaling affects cluster formation.
🎨 **Visual Cluster Plots**  
Generates plots for each clustering result, displaying points by cluster with distinct colors and cluster centers.  
🔁 **Multiple Runs for Robustness**  
Runs clustering multiple times to highlight the effect of random initialization on results, particularly for the Iris dataset.
📏 **Distance Analysis**  
Computes distances between original and computed cluster centers for comparison.

## Getting Started

1. **Clone the Repository:** Clone this repository to your local machine using Git.

  ```
 git clone https://github.com/Khadejia/KMeans_Clustering.git
  ```
2. **Add Data and Scripts:** For clustering the kmtest dataset:

For clustering the kmtest dataset:
  ```
 python kmtest.py
  ```
For clustering the Iris dataset:
  ```
 python iris.py
  ```
3. **Visualize and Analyze:**
- Optimize the K-Means implementation for performance.
- Add alternative initialization methods (e.g., K-Means++).
- Implement additional distance metrics (Manhattan, cosine, etc.).
- Extend visualization for higher-dimensional datasets.
- Add more datasets for testing.

## Contributing

Contributions are welcome! You can:
- Optimize the K-Means implementation for performance.
- Add alternative initialization methods (e.g., K-Means++).
- Implement additional distance metrics (Manhattan, cosine, etc.).
- Extend visualization for higher-dimensional datasets.
- Add more datasets for testing.
Please submit a pull request for any changes or improvements.

## Support
If you find this project useful:
- ⭐ Star this repository
- Share with peers learning machine learning
- Follow me on GitHub for future projects  
Your support is greatly appreciated!
