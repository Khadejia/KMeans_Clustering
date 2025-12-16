import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "kmtest.csv")


data = pd.read_csv(
    csv_path, 
    header=None, 
    delim_whitespace=True, 
    encoding="utf-8-sig"
)
X = data.values.astype(float)


def z_score_normalize(X):

    return (X - X.mean(axis=0)) / X.std(axis=0)

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

def cluster(X, ctrs, lbls, title, dims=(0,1)):
    plt.figure(figsize=(6,5))
    
    for i in range(len(ctrs)):
        points = X[lbls == i]
        plt.scatter(
            points[:, dims[0]],
            points[:, dims[1]],
            marker='o',
            label=f"Cluster {i + 1}"
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

for k in [2, 3, 4, 5]:
    ctrs, lbls = k_means(X, k)
    cluster(X, ctrs, lbls, f"kmtest - K={k} (no normalization)")

X_norm = z_score_normalize(pd.DataFrame(X)).values
for k in [2, 3, 4, 5]:
    ctrs, lbls = k_means(X_norm, k)
    cluster(X_norm, ctrs, lbls, f"kmtest - K={k} (z-score normalized)")
