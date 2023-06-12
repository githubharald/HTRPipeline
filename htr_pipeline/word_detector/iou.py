import numpy as np


def compute_iou(ra, rb):
    """intersection over union of two axis aligned rectangles ra and rb"""
    if ra.xmax < rb.xmin or rb.xmax < ra.xmin or ra.ymax < rb.ymin or rb.ymax < ra.ymin:
        return 0

    l = max(ra.xmin, rb.xmin)
    r = min(ra.xmax, rb.xmax)
    t = max(ra.ymin, rb.ymin)
    b = min(ra.ymax, rb.ymax)

    intersection = (r - l) * (b - t)
    union = ra.area() + rb.area() - intersection

    iou = intersection / union
    return iou


def compute_dist_mat(aabbs):
    """Jaccard distance matrix of all pairs of aabbs"""
    num_aabbs = len(aabbs)

    dists = np.zeros((num_aabbs, num_aabbs))
    for i in range(num_aabbs):
        for j in range(num_aabbs):
            if j > i:
                break

            dists[i, j] = dists[j, i] = 1 - compute_iou(aabbs[i], aabbs[j])

    return dists


def compute_dist_mat_2(aabbs1, aabbs2):
    """Jaccard distance matrix of all pairs of aabbs from lists aabbs1 and aabbs2"""
    num_aabbs1 = len(aabbs1)
    num_aabbs2 = len(aabbs2)

    dists = np.zeros((num_aabbs1, num_aabbs2))
    for i in range(num_aabbs1):
        for j in range(num_aabbs2):
            dists[i, j] = 1 - compute_iou(aabbs1[i], aabbs2[j])

    return dists
