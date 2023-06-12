import cv2
import numpy as np

from .aabb import AABB


class MapOrdering:
    """order of the maps encoding the aabbs around the words"""
    SEG_WORD = 0
    SEG_SURROUNDING = 1
    SEG_BACKGROUND = 2
    GEO_TOP = 3
    GEO_BOTTOM = 4
    GEO_LEFT = 5
    GEO_RIGHT = 6
    NUM_MAPS = 7


def encode(shape, gt, f=1.0):
    gt_map = np.zeros((MapOrdering.NUM_MAPS,) + shape)
    for aabb in gt:
        aabb = aabb.scale(f, f)

        # segmentation map
        aabb_clip = AABB(0, shape[0] - 1, 0, shape[1] - 1)

        aabb_word = aabb.scale_around_center(0.5, 0.5).as_type(int).clip(aabb_clip)
        aabb_sur = aabb.as_type(int).clip(aabb_clip)
        gt_map[MapOrdering.SEG_SURROUNDING, aabb_sur.ymin:aabb_sur.ymax + 1, aabb_sur.xmin:aabb_sur.xmax + 1] = 1
        gt_map[MapOrdering.SEG_SURROUNDING, aabb_word.ymin:aabb_word.ymax + 1, aabb_word.xmin:aabb_word.xmax + 1] = 0
        gt_map[MapOrdering.SEG_WORD, aabb_word.ymin:aabb_word.ymax + 1, aabb_word.xmin:aabb_word.xmax + 1] = 1

        # geometry map TODO vectorize
        for x in range(aabb_word.xmin, aabb_word.xmax + 1):
            for y in range(aabb_word.ymin, aabb_word.ymax + 1):
                gt_map[MapOrdering.GEO_TOP, y, x] = y - aabb.ymin
                gt_map[MapOrdering.GEO_BOTTOM, y, x] = aabb.ymax - y
                gt_map[MapOrdering.GEO_LEFT, y, x] = x - aabb.xmin
                gt_map[MapOrdering.GEO_RIGHT, y, x] = aabb.xmax - x

    gt_map[MapOrdering.SEG_BACKGROUND] = np.clip(1 - gt_map[MapOrdering.SEG_WORD] - gt_map[MapOrdering.SEG_SURROUNDING],
                                                 0, 1)

    return gt_map


def subsample(idx, max_num):
    """restrict fg indices to a maximum number"""
    f = len(idx[0]) / max_num
    if f > 1:
        a = np.asarray([idx[0][int(j * f)] for j in range(max_num)], np.int64)
        b = np.asarray([idx[1][int(j * f)] for j in range(max_num)], np.int64)
        idx = (a, b)
    return idx


def fg_by_threshold(thres, max_num=None):
    """all pixels above threshold are fg pixels, optionally limited to a maximum number"""

    def func(seg_map):
        idx = np.where(seg_map > thres)
        if max_num is not None:
            idx = subsample(idx, max_num)
        return idx

    return func


def fg_by_cc(thres, max_num):
    """take a maximum number of pixels per connected component, but at least 3 (->DBSCAN minPts)"""

    def func(seg_map):
        seg_mask = (seg_map > thres).astype(np.uint8)
        num_labels, label_img = cv2.connectedComponents(seg_mask, connectivity=4)
        max_num_per_cc = max(max_num // (num_labels + 1), 3)  # at least 3 because of DBSCAN clustering

        all_idx = [np.empty(0, np.int64), np.empty(0, np.int64)]
        for curr_label in range(1, num_labels):
            curr_idx = np.where(label_img == curr_label)
            curr_idx = subsample(curr_idx, max_num_per_cc)
            all_idx[0] = np.append(all_idx[0], curr_idx[0])
            all_idx[1] = np.append(all_idx[1], curr_idx[1])
        return tuple(all_idx)

    return func


def decode(pred_map, comp_fg=fg_by_threshold(0.5), f=1):
    idx = comp_fg(pred_map[MapOrdering.SEG_WORD])
    pred_map_masked = pred_map[..., idx[0], idx[1]]
    aabbs = []
    for yc, xc, pred in zip(idx[0], idx[1], pred_map_masked.T):
        t = pred[MapOrdering.GEO_TOP]
        b = pred[MapOrdering.GEO_BOTTOM]
        l = pred[MapOrdering.GEO_LEFT]
        r = pred[MapOrdering.GEO_RIGHT]
        aabb = AABB(xc - l, xc + r, yc - t, yc + b)
        aabbs.append(aabb.scale(f, f))
    return aabbs
