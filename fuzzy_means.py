import numpy as np


def cmeans(data, c, m=2.5, epsilon=1e-3, maxiter=int(1e3), seed=42):
    n = data.shape[0]
    if c > n or c < 0:
        return -1
    centroid_indices = np.random.choice(n, c, replace=False)
    centroids = data.iloc[centroid_indices].to_numpy()
    mem_mat = compute_membership(data, centroids, m)
    p = 0
    # print("MEMMAT:", mem_mat)
    for _ in range(maxiter):
        # print("DEBUG:", centroids)
        p += 1
        new_centroids = find_centroids(data, mem_mat, c, m)
        if (np.linalg.norm(centroids - new_centroids, axis=0) < epsilon).all():
            break
        centroids = new_centroids
        mem_mat = compute_membership(data, centroids, m)
    return centroids, mem_mat, p


def compute_membership(data, centroids, m):
    mem_mat = np.zeros((data.shape[0], len(centroids)))
    for d in range(data.shape[0]):
        data_pt = data.iloc[d].to_numpy()
        # print("IN COMP", d)
        for c, cntr in enumerate(centroids):
            denominator = 0
            numer = np.linalg.norm(data_pt - cntr)
            dd = 0
            for o_cntr in centroids:
                if np.linalg.norm(data_pt - o_cntr) < 1e-4:
                    continue
                denominator += (numer /
                                np.linalg.norm(data_pt - o_cntr))**(2 /
                                                                    (m - 1))
                dd += 1
            print("DENO: ", dd)
            mem_mat[d][c] = 1 / denominator
    return mem_mat


def find_centroids(data, membership, c, m):
    new_cntrs = np.zeros((c, data.shape[1]))
    for j in range(c):
        numer = np.zeros(data.shape[1])
        denom = 0
        for i in range(data.shape[0]):
            numer += (membership[i][j]**m) * data.iloc[i].to_numpy()
            denom += membership[i][j]**m

        new_cntrs[j] = numer / denom
    return new_cntrs
