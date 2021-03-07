def defuzz(dist_mat):
    labels = []
    for dist in zip(*dist_mat):
        labels.append(dist.index(min(dist)))
    return labels
