from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN

from .aabb import AABB
from .iou import compute_dist_mat


def cluster_aabbs(aabbs):
    """cluster aabbs using DBSCAN and the Jaccard distance between bounding boxes"""
    if len(aabbs) < 2:
        return aabbs

    dists = compute_dist_mat(aabbs)
    clustering = DBSCAN(eps=0.7, min_samples=3, metric='precomputed').fit(dists)

    clusters = defaultdict(list)
    for i, c in enumerate(clustering.labels_):
        if c == -1:
            continue
        clusters[c].append(aabbs[i])

    res_aabbs = []
    for curr_cluster in clusters.values():
        xmin = np.median([aabb.xmin for aabb in curr_cluster])
        xmax = np.median([aabb.xmax for aabb in curr_cluster])
        ymin = np.median([aabb.ymin for aabb in curr_cluster])
        ymax = np.median([aabb.ymax for aabb in curr_cluster])
        res_aabbs.append(AABB(xmin, xmax, ymin, ymax))

    return res_aabbs
