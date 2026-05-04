from __future__ import annotations

from typing import List

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from src.models.utils import ImageInput, normalize_rows, to_pil


class FashionCLIPEncoder:
    """
    Wrapper around FashionCLIP for encoding text queries, single images, and
    image batches into a shared 512-d L2-normalised embedding space.
    """

    def __init__(self, model_name: str = "patrickjohncyh/fashion-clip"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # `use_safetensors=False` works around a macOS arm64 SIGBUS that happens
        # when matmul touches safetensors-mmapped weights from this checkpoint.
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=False).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs).cpu().numpy()
        return normalize_rows(emb)[0]

    def encode_image(self, image: ImageInput) -> np.ndarray:
        return self.encode_images([image])[0]

    def encode_images(self, images: List[ImageInput]) -> np.ndarray:
        """Batch-embed images. Returns (N, 512) L2-normalised float32."""
        pil_images = [to_pil(im) for im in images]
        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            embs = self.model.get_image_features(**inputs).cpu().numpy()
        return normalize_rows(embs)


def build_fashion_clip_encoder() -> FashionCLIPEncoder:
    return FashionCLIPEncoder()
