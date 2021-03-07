import pandas as pd
import numpy as np
from datetime import datetime


def kmeans_iterate(df, centroids):
    n = df.shape[0]
    k = len(centroids)
    cluster_number = np.zeros(n, dtype=np.int)
    cluster_members = [[] for i in range(k)]
    for i in range(n):
        distances = np.linalg.norm(centroids - df.iloc[i].to_numpy(), axis=1)
        cluster_number[i] = round(np.argmin(distances))
        # print(cluster_number[i])
        cluster_members[cluster_number[i]].append(i)
    new_centroids = centroids.copy()
    for i in range(k):
        new_centroids[i] = df.iloc[cluster_members[i]].to_numpy().mean(axis=0)
    return new_centroids


def kmeans(df, k, maxiter=np.inf):
    n = df.shape[0]
    if k > n or k < 0:
        return -1
    centroid_indices = np.random.choice(n, k, replace=False)
    centroids = df.iloc[centroid_indices].to_numpy()
    new_centroids = kmeans_iterate(df, centroids)
    iter = 0
    while not ((centroids == new_centroids).all()):
        centroids = new_centroids
        new_centroids = kmeans_iterate(df, centroids)
        iter += 1
        if iter >= maxiter:
            break

    cluster_number = np.zeros(n, dtype=np.int)
    for i in range(n):
        distances = np.linalg.norm(centroids - df.iloc[i].to_numpy(), axis=1)
        cluster_number[i] = round(np.argmin(distances))
    return cluster_number, centroids
