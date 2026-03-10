from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.data.catalog import Catalog
from src.models.fashion_clip_encoder import FashionCLIPEncoder
from src.retrieval.filtering import matches_filters


def search_hybrid(
    catalog: Catalog,
    encoder: FashionCLIPEncoder,
    selected_filters: Dict[str, List[str]],
    query_text: Optional[str] = None,
    query_image=None,
    top_k: int = 12,
    text_weight: float = 0.5,
    image_weight: float = 0.5,
) -> List[dict]:
    candidate_indices = np.array(
        [
            idx
            for idx, image_id in enumerate(catalog.image_ids)
            if matches_filters(
                image_id=image_id,
                catalog=catalog,
                selected_filters=selected_filters,
            )
        ],
        dtype=np.int64,
    )

    if len(candidate_indices) == 0:
        return []

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

    candidate_embeddings = catalog.embeddings[candidate_indices]
    scores = candidate_embeddings @ query_embedding

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