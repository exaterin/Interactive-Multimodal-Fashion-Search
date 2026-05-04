from __future__ import annotations

from typing import List

import numpy as np
import torch
import open_clip

from src.models.utils import ImageInput, normalize_rows, to_pil


class CLIPEncoder:
    """
    Wrapper around an open_clip backbone (default EVA02-B-16) for encoding
    text queries, single images, and image batches into a shared L2-normalised
    embedding space.
    """

    def __init__(self, model_name: str = "EVA02-B-16", pretrained: str = "merged2b_s8b_b131k"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens).cpu().numpy()
        return normalize_rows(emb)[0]

    def encode_image(self, image: ImageInput) -> np.ndarray:
        return self.encode_images([image])[0]

    def encode_images(self, images: List[ImageInput]) -> np.ndarray:
        """Batch-embed images. Returns (N, D) L2-normalised float32."""
        pil_images = [to_pil(im) for im in images]
        tensors = torch.stack([self.preprocess(im) for im in pil_images]).to(self.device)
        with torch.no_grad():
            embs = self.model.encode_image(tensors).cpu().numpy()
        return normalize_rows(embs)
