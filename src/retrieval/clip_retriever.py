from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.data.catalog import Catalog
from src.models.fashion_clip_encoder import FashionCLIPEncoder


def _rank_results(
    scores: np.ndarray,
    catalog: Catalog,
    candidate_indices: np.ndarray,
    top_k: int,
) -> List[dict]:
    top_k = min(top_k, len(scores))
    top_local_indices = np.argsort(-scores)[:top_k]

    results = []
    for rank, local_idx in enumerate(top_local_indices, start=1):
        global_idx = candidate_indices[local_idx]
        image_id = catalog.image_ids[global_idx]
        results.append(
            {
                "rank": rank,
                "image_id": image_id,
                "score": float(scores[local_idx]),
            }
        )
    return results


def search_clip(
    catalog: Catalog,
    encoder: FashionCLIPEncoder,
    query_text: Optional[str] = None,
    query_image=None,
    top_k: int = 12,
    text_weight: float = 0.5,
    image_weight: float = 0.5,
) -> List[dict]:
    embeddings = []

    if query_text and query_text.strip():
        embeddings.append((text_weight, encoder.encode_text(query_text)))

    if query_image is not None:
        embeddings.append((image_weight, encoder.encode_image(query_image)))

    if not embeddings:
        return []

    total_weight = sum(weight for weight, _ in embeddings)
    query_embedding = sum(weight * emb for weight, emb in embeddings) / total_weight

    norm = np.linalg.norm(query_embedding)
    if norm > 0:
        query_embedding = query_embedding / norm

    candidate_indices = np.arange(len(catalog.image_ids))
    scores = catalog.embeddings @ query_embedding

    return _rank_results(scores, catalog, candidate_indices, top_k)