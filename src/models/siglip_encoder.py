from __future__ import annotations

from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from src.models.utils import ImageInput, normalize_rows, to_pil


class SigLIPEncoder:
    """
    Wrapper around Google's SigLIP for encoding text queries, single images,
    and image batches into a shared L2-normalised embedding space.
    """

    def __init__(self, model_name: str = "google/siglip-large-patch16-256"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.text_model = self.model.text_model
        self.vision_model = self.model.vision_model

    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            emb = self.text_model(**inputs).pooler_output.cpu().numpy()
        return normalize_rows(emb)[0]

    def encode_image(self, image: ImageInput) -> np.ndarray:
        return self.encode_images([image])[0]

    def encode_images(self, images: List[ImageInput]) -> np.ndarray:
        """Batch-embed images. Returns (N, 1024) L2-normalised float32."""
        pil_images = [to_pil(im) for im in images]
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embs = self.vision_model(**inputs).pooler_output.cpu().numpy()
        return normalize_rows(embs)
