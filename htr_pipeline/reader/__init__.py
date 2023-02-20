import numpy as np

from .model import crnn, Batch, Preprocessor


def read(img: np.ndarray) -> str:
    """Recognizes text in image provided by file path."""

    preprocessor = Preprocessor((128, 32), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    text = crnn.infer_batch(batch)
    return text[0]
