"""Helpers shared across encoder classes in ``src/models``."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


ImageInput = Union[str, Path, Image.Image]


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    """L2-normalise each row of ``arr`` to unit length. Zero-norm rows are left
    untouched. Always returns float32."""
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return arr / norms


def to_pil(image: ImageInput) -> Image.Image:
    """Coerce ``str | Path | PIL.Image`` into a fresh RGB PIL image."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    return image.convert("RGB")
