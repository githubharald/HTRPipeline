import json
import math
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
from pkg_resources import resource_filename

from .ctc import ctc_best_path, ctc_single_word_beam_search, PrefixTree


def _load_model():
    """Loads model and model metadata."""
    ort_session = ort.InferenceSession(resource_filename('htr_pipeline', 'models/reader.onnx'),
                                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    with open(resource_filename('htr_pipeline', 'models/reader.json')) as f:
        chars = json.load(f)['chars']
    return ort_session, chars


def transform(img: np.ndarray) -> np.ndarray:
    """Bring image into suitable shape for the model."""
    target_height = 48
    padding = 32

    # compute shape of target image
    fh = target_height / img.shape[0]
    f = min(fh, 2)
    h = target_height
    w = math.ceil(img.shape[1] * f)
    w = w + (4 - w) % 4
    w += padding

    # create target image
    res = 255 * np.ones((h, w), dtype=np.uint8)

    # copy image into target image
    img = cv2.resize(img, dsize=None, fx=f, fy=f)
    th = (res.shape[0] - img.shape[0]) // 2
    tw = (res.shape[1] - img.shape[1]) // 2
    res[th:img.shape[0] + th, tw:img.shape[1] + tw] = img
    res = res

    return res / 255 - 0.5


# global vars holding the model and model metadata
_ORT_SESSION, _CHARS = _load_model()


def read(img: np.ndarray, decoder: str, prefix_tree: Optional[PrefixTree] = None) -> str:
    """Recognizes text in image."""
    img = transform(img)
    img = img[None, None].astype(np.float32)
    outputs = _ORT_SESSION.run(None, {'input': img})

    if decoder == 'best_path':
        text = ctc_best_path(outputs[0], _CHARS)[0]
    elif decoder == 'word_beam_search':
        text = ctc_single_word_beam_search(outputs[0], _CHARS, 25, prefix_tree)[0]
    else:
        raise Exception('Unknown decoder. Available: "best_path" and "word_beam_search".')

    return text
