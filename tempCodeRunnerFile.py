import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "iris.csv")

data = pd.read_csv(
    csv_path,
    header=None,
    encoding="utf-8-sig"
)

X = data.iloc[:, 0:4].values.astype(float)
Y = data.iloc[:, 4].values

def kmeans_ctrs(X, Z):
    indx = np.random.choice(len(X), Z, replace=False)
    return X[indx]

def distance(a, b):
    difference = a - b
    return np.sqrt(np.sum(difference * difference))

def k_means(X, Z, max_iter=100, tol=1e-4):
    ctrs = kmeans_ctrs(X, Z)
    
    for iter in range(max_iter):
        lbls = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            min_dist = float('inf')
            for j in range(Z):
                dist = distance(X[i], ctrs[j])
                if dist < min_dist:
                    min_dist = dist
                    lbls[i] = j
        
        new_ctrs = np.zeros_like(ctrs)
        for j in range(Z):
            pts = []
            for i in range(X.shape[0]):
                if lbls[i] == j:
                    pts.append(X[i])
            if pts:
                new_ctrs[j] = np.mean(pts, axis=0)
            else:
                new_ctrs[j] = ctrs[j] 
        
        max_chg = 0.0
        for j in range(Z):
            chg = distance(ctrs[j], new_ctrs[j])
            if chg > max_chg:
                max_chg = chg
        if max_chg < tol:
            break
        
        ctrs = new_ctrs
    
    return ctrs, lbls

# DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!















def cluster(X, ctrs, lbls, title, dims=(2,3)):
    plt.figure(figsize=(6,5))
    
    for i in range(len(ctrs)):
        cluster_points = X[lbls == i]
        plt.scatter(
            cluster_points[:, dims[0]],
            cluster_points[:, dims[1]],
            'o', label=f"Cluster {i + 1}"
        )
    
    plt.scatter(
        ctrs[:, dims[0]],
        ctrs[:, dims[1]],
        c="black",
        marker="x",
        s=100,
        label="Centers"
    )
    
    plt.title(title)
    plt.legend()
    plt.show()


results = []
for i in range(5): 
    ctrs, lbls = k_means(X, 3)
    results.append((ctrs, lbls))

def variance(X, ctrs, lbls):
    sum = 0.0
    for j in range(len(ctrs)):
        points = X[lbls == j]
        diff = points - ctrs[j]
        sum += np.sum(diff ** 2)
    return sum


variances = []
for ctrs_run, lbls_run in results:
    variances.append(variance(X, ctrs_run, lbls_run))
    
best_idx, worst_idx = np.argmin(variance), np.argmax(variance)
best_ctrs, best_lbls = results[best_idx]
worst_ctrs, worst_lbls = results[worst_idx]

cluster(X, best_ctrs, best_lbls, dims=(2,3), title = "Iris - Best Clustering")
cluster(X, worst_ctrs, worst_lbls, dims=(2,3), title = "Iris - Worst Clustering")

plt.figure(figsize=(6,5))
for lab, name in zip([0,1,2], ["Setosa", "Versicolor", "Virginica"]):
    plt.scatter(
        X[Y == data[4].unique()[lab], 2],
        X[Y == data[4].unique()[lab], 3],
        label=name
    )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#plt.title("Iris - Original Labels")
plt.legend()
plt.show()

true_ctrs = []
for species in data[4].unique():
    true_ctrs.append(X[Y == species].mean(axis=0))
true_ctrs = np.array(true_ctrs)

distances = np.linalg.norm(
    best_ctrs[:, None, :] - true_ctrs[None, :, :],
    axis=2
)

print("Distances between best clustering centers and original centers:")
print(distances)
