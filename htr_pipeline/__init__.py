from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .reader import read
from .reader.ctc import PrefixTree
from .word_detector import detect, sort_multiline, AABB


@dataclass
class WordReadout:
    """Information about a read word: the readout and the bounding box."""
    text: str
    aabb: AABB


@dataclass
class DetectorConfig:
    """Configure size at which word detection is done, and define added margin around word before reading."""
    scale: float = 1.0
    margin: int = 0


@dataclass
class LineClusteringConfig:
    """Configure how word detections are clustered into lines."""
    min_words_per_line: int = 1  # minimum number of words per line, if less, line gets discarded
    max_dist: float = 0.7  # threshold for clustering words into lines, value between 0 and 1


@dataclass
class ReaderConfig:
    """Configure how the detected words are read."""
    decoder: str = 'best_path'  # 'best_path' or 'word_beam_search'
    prefix_tree: Optional[PrefixTree] = None


def read_page(img: np.ndarray,
              detector_config: DetectorConfig = DetectorConfig(),
              line_clustering_config=LineClusteringConfig(),
              reader_config=ReaderConfig()) -> List[List[WordReadout]]:
    """Read a page of handwritten words. Returns a list of lines. Each line is a list of read words."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # detect words
    detections = detect(img, detector_config.scale, detector_config.margin)

    # sort words (cluster into lines and ensure reading order top->bottom and left->right)
    lines = sort_multiline(detections, min_words_per_line=line_clustering_config.min_words_per_line)

    # go through all lines and words and read all of them
    read_lines = []
    for line in lines:
        read_lines.append([])
        for word in line:
            text = read(word.img, reader_config.decoder, reader_config.prefix_tree)
            read_lines[-1].append(WordReadout(text, word.aabb))

    return read_lines
