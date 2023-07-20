from collections import defaultdict
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import onnxruntime as ort
from pkg_resources import resource_filename
from sklearn.cluster import DBSCAN

from .aabb import AABB
from .aabb_clustering import cluster_aabbs
from .coding import decode, fg_by_cc, fg_by_threshold
from .iou import compute_iou


def _load_model():
    """Loads model and model metadata."""
    ort_session = ort.InferenceSession(resource_filename('htr_pipeline', 'models/detector.onnx'),
                                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    return ort_session


# global vars holding the model and model metadata
_ORT_SESSION = _load_model()


@dataclass
class DetectorRes:
    img: np.ndarray
    aabb: AABB


def ceil32(val):
    if val % 32 == 0:
        return val
    val = (val // 32 + 1) * 32
    return val


def pad_image(img):
    res = 255 * np.ones([ceil32(img.shape[0]), ceil32(img.shape[1])])
    res[:img.shape[0], :img.shape[1]] = img
    return res


def detect(img: np.ndarray, scale: float, margin: int) -> List[DetectorRes]:
    img_resized = cv2.resize(img, None, fx=scale, fy=scale)
    img_padded = pad_image(img_resized)
    img_batch = img_padded.astype(np.float32)[None, None] / 255 - 0.5

    outputs = _ORT_SESSION.run(None, {'input': img_batch})
    pred_map = outputs[0][0]
    aabbs = decode(pred_map, comp_fg=fg_by_cc(0.5, 100), f=img_batch.shape[2] / pred_map.shape[1])
    aabbs = [aabb.scale(1 / scale, 1 / scale) for aabb in aabbs if aabb.scale(1 / scale, 1 / scale)]
    h, w = img.shape
    aabbs = [aabb.clip(AABB(0, w - 1, 0, h - 1)) for aabb in aabbs]  # bounding box must be inside img
    clustered_aabbs = cluster_aabbs(aabbs)

    res = []
    for aabb in clustered_aabbs:
        aabb = aabb.enlarge(margin)
        aabb = aabb.as_type(int).clip(AABB(0, img.shape[1], 0, img.shape[0]))
        if aabb.area() == 0:
            continue
        crop = img[aabb.ymin:aabb.ymax, aabb.xmin:aabb.xmax]
        res.append(DetectorRes(crop, aabb))

    return res


def _cluster_lines(detections: List[DetectorRes],
                   max_dist: float = 0.7,
                   min_words_per_line: int = 2) -> List[List[DetectorRes]]:
    # compute matrix containing Jaccard distances (which is a proper metric)
    num_bboxes = len(detections)
    dist_mat = np.ones((num_bboxes, num_bboxes))
    for i in range(num_bboxes):
        for j in range(i, num_bboxes):
            a = detections[i].aabb
            b = detections[j].aabb
            if a.ymin > b.ymax or b.ymin > a.ymax:
                continue
            intersection = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
            union = a.height + b.height - intersection
            iou = np.clip(intersection / union if union > 0 else 0, 0, 1)
            dist_mat[i, j] = dist_mat[j, i] = 1 - iou  # Jaccard distance is defined as 1-iou

    dbscan = DBSCAN(eps=max_dist, min_samples=min_words_per_line, metric='precomputed').fit(dist_mat)

    clustered = defaultdict(list)
    for i, cluster_id in enumerate(dbscan.labels_):
        if cluster_id == -1:
            continue
        clustered[cluster_id].append(detections[i])

    res = sorted(clustered.values(), key=lambda line: [det.aabb.ymin + det.aabb.height / 2 for det in line])
    return res


def sort_multiline(detections: List[DetectorRes],
                   max_dist: float = 0.7,
                   min_words_per_line: int = 2) -> List[List[DetectorRes]]:
    """Cluster detections into lines, then sort the lines according to x-coordinates of word centers.

    Args:
        detections: List of detections.
        max_dist: Maximum Jaccard distance (0..1) between two y-projected words to be considered as neighbors.
        min_words_per_line: If a line contains less words than specified, it is ignored.

    Returns:
        List of lines, each line itself a list of detections.
    """
    lines = _cluster_lines(detections, max_dist, min_words_per_line)
    res = []
    for line in lines:
        res += sort_line(line)
    return res


def sort_line(detections: List[DetectorRes]) -> List[List[DetectorRes]]:
    """Sort the list of detections according to x-coordinates of word centers."""
    return [sorted(detections, key=lambda det: det.aabb.xmin + det.aabb.width / 2)]
