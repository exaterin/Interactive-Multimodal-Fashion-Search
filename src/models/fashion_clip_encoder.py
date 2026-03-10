from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import streamlit as st
from fashion_clip.fashion_clip import FashionCLIP
from PIL import Image


ImageInput = Union[str, Path, Image.Image]


class FashionCLIPEncoder:
    """
    Wrapper around FashionCLIP for encoding text queries
    and uploaded/reference images.
    """

    def __init__(self, model_name: str = "fashion-clip"):
        self.model = FashionCLIP(model_name)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode_text(self, text: str) -> np.ndarray:
        emb = self.model.encode_text([text], batch_size=512)[0]
        return self._normalize(emb)

    def encode_image(self, image: ImageInput) -> np.ndarray:
        """
        Accepts:
        - path string
        - Path object
        - PIL image
        """
        if isinstance(image, Image.Image):
            emb = self.model.encode_images([image], batch_size=1)[0]
        else:
            emb = self.model.encode_images([str(image)], batch_size=1)[0]

        return self._normalize(emb)


def build_fashion_clip_encoder() -> FashionCLIPEncoder:
    return FashionCLIPEncoder()