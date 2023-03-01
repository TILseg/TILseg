import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def kmeans_best_n_clusters(patch_path, n_clusters_considered):

    """
    Computes the best number of clusters for Kmeans clustering

    Parameters
    -----
    patch_path: path to the patch to be clustered
    n_clusters_considered: the maximum number of clusters to consider

    Returns
    -----
    best_n: the optimal number of clusters for K means clustering
    Prints a graph of inertia vs number of clusters (if called from a notebook)
    """

    n_list = list(range(1, n_clusters_considered + 1))
    fit_patch = plt.imread(patch_path)
    fit_patch_n = np.float32(fit_patch.reshape((-1, 3))/255.)

    inertias = []

    for n in n_list:
        
        model = KMeans(n_clusters=n, max_iter=5,
                    n_init=3, tol=1e-3)
        model.fit(fit_patch_n)
        inertias.append(model.inertia_)

    plt.plot(n_list, inertias)
    plt.xlabel('number of clusters')
    plt.ylabel('inertia')
    plt.xticks(n_list)
    plt.title('Inertia vs Number of CLusters')

    i = 2
    cur_slope_diff = 1
    prev_slope_diff = 0

    while cur_slope_diff > prev_slope_diff:
        i = i + 1
        cur_slope = inertias[i-1] - inertias[i]
        prev_slope = inertias[i-2] - inertias[i-1]
        prev_prev_slope = inertias[i-3] - inertias[i-2]
        cur_slope_diff = prev_slope - cur_slope
        prev_slope_diff = prev_prev_slope - prev_slope
        
    best_n = n_list[i]

    return best_n