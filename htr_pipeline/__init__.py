from dataclasses import dataclass
from typing import List

import numpy as np

from .reader import read
from .word_detector import detect, sort_multiline, prepare_img, BBox


@dataclass
class WordReadout:
    text: str
    bbox: BBox


@dataclass
class DetectorConfig:
    height: int = 1000
    kernel_size: int = 25
    sigma: float = 11
    theta: float = 7
    min_area: int = 100


@dataclass
class LineClusteringConfig:
    min_words_per_line: int = 2
    max_dist: float = 0.7


def read_page(img: np.ndarray,
              detector_config: DetectorConfig = DetectorConfig(1000),
              line_clustering_config=LineClusteringConfig()) -> List[List[WordReadout]]:
    # prepare image
    img, f = prepare_img(img, detector_config.height)

    # detect words
    detections = detect(img,
                        detector_config.kernel_size,
                        detector_config.sigma,
                        detector_config.theta,
                        detector_config.min_area)

    # sort words (cluster into lines and ensure reading order top->bottom and left->right)
    lines = sort_multiline(detections, min_words_per_line=line_clustering_config.min_words_per_line)

    # go through all lines and words and read all of them
    read_lines = []
    for line in lines:
        read_lines.append([])
        for word in line:
            text = read(word.img)
            read_lines[-1].append(WordReadout(text, word.bbox * (1 / f)))

    return read_lines
