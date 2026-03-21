from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.data.fashionpedia.catalog import FashionpediaCatalog
from src.data.fashionpedia.annotation_parser import (
    _PATTERN_SUPERCATS,
    _SHAPE_SUPERCATS,
    _MATERIAL_SUPERCATS,
    _FINISHING_SUPERCATS,
)
from src.models.fashion_clip_encoder import FashionCLIPEncoder


# Filtering

def matches_filters_fp(
    item_id: str,
    catalog: FashionpediaCatalog,
    selected_filters: Dict[str, List[str]],
) -> bool:
    """
    Return True if the annotation satisfies all non-empty filter criteria.

    Filter keys:
        "category"  — garment category name (shirt, dress, pants, …)
        "pattern"   — textile pattern + animal print attributes
        "shape"     — silhouette / neckline / length / waistline / opening type
        "material"  — non-textile material and leather attributes
        "finishing" — textile finishing / manufacturing techniques
    """
    if selected_filters.get("category"):
        item_cat = catalog.category_annotations.get(item_id, "")
        if item_cat not in selected_filters["category"]:
            return False

    if selected_filters.get("pattern"):
        supercat_dict = catalog.attribute_annotations.get(item_id, {})
        item_patterns: set = set()
        for sc in _PATTERN_SUPERCATS:
            item_patterns.update(supercat_dict.get(sc, set()))
        if not set(selected_filters["pattern"]).issubset(item_patterns):
            return False

    if selected_filters.get("shape"):
        supercat_dict = catalog.attribute_annotations.get(item_id, {})
        item_shapes: set = set()
        for sc in _SHAPE_SUPERCATS:
            item_shapes.update(supercat_dict.get(sc, set()))
        if not set(selected_filters["shape"]).issubset(item_shapes):
            return False

    if selected_filters.get("material"):
        supercat_dict = catalog.attribute_annotations.get(item_id, {})
        item_materials: set = set()
        for sc in _MATERIAL_SUPERCATS:
            item_materials.update(supercat_dict.get(sc, set()))
        if not set(selected_filters["material"]).issubset(item_materials):
            return False

    if selected_filters.get("finishing"):
        supercat_dict = catalog.attribute_annotations.get(item_id, {})
        item_finishings: set = set()
        for sc in _FINISHING_SUPERCATS:
            item_finishings.update(supercat_dict.get(sc, set()))
        if not set(selected_filters["finishing"]).issubset(item_finishings):
            return False

    return True

# Internal ranking helper

def _rank_results(
    scores: np.ndarray,
    catalog: FashionpediaCatalog,
    candidate_indices: np.ndarray,
    top_k: int,
) -> List[dict]:
    top_k = min(top_k, len(scores))
    top_local = np.argsort(-scores)[:top_k]
    results = []
    for rank, local_idx in enumerate(top_local, start=1):
        global_idx = candidate_indices[local_idx]
        item_id = catalog.item_ids[global_idx]
        results.append(
            {
                "rank": rank,
                "image_id": item_id,          # kept as "image_id" for UI compatibility
                "score": float(scores[local_idx]),
            }
        )
    return results


# Search functions

def search_clip_fp(
    catalog: FashionpediaCatalog,
    encoder: FashionCLIPEncoder,
    query_text: Optional[str] = None,
    query_image=None,
    top_k: int = 12,
    text_weight: float = 0.5,
    image_weight: float = 0.5,
) -> List[dict]:
    """Semantic search over Fashionpedia bbox-crop embeddings."""
    raw: list = []

    if query_text and query_text.strip():
        raw.append((text_weight, encoder.encode_text(query_text)))
    if query_image is not None:
        raw.append((image_weight, encoder.encode_image(query_image)))

    if not raw:
        return []

    total = sum(w for w, _ in raw)
    query_emb = sum(w * e for w, e in raw) / total
    norm = np.linalg.norm(query_emb)
    if norm > 0:
        query_emb = query_emb / norm

    candidate_indices = np.arange(len(catalog.item_ids))
    scores = catalog.embeddings @ query_emb

    return _rank_results(scores, catalog, candidate_indices, top_k)


def search_by_attributes_fp(
    catalog: FashionpediaCatalog,
    selected_filters: Dict[str, List[str]],
    top_k: int = 50,
) -> List[dict]:
    """Return annotations that match all selected attribute filters."""
    results = []
    for item_id in catalog.item_ids:
        if matches_filters_fp(item_id, catalog, selected_filters):
            results.append({"image_id": item_id, "score": None})

    results = results[:top_k]
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank

    return results


def search_hybrid_fp(
    catalog: FashionpediaCatalog,
    encoder: FashionCLIPEncoder,
    selected_filters: Dict[str, List[str]],
    query_text: Optional[str] = None,
    query_image=None,
    top_k: int = 12,
    text_weight: float = 0.5,
    image_weight: float = 0.5,
) -> List[dict]:
    """Filter by attributes then rank surviving crops with FashionCLIP."""
    raw: list = []
    if query_text and query_text.strip():
        raw.append((text_weight, encoder.encode_text(query_text)))
    if query_image is not None:
        raw.append((image_weight, encoder.encode_image(query_image)))

    if not raw:
        return []

    total = sum(w for w, _ in raw)
    query_emb = sum(w * e for w, e in raw) / total
    norm = np.linalg.norm(query_emb)
    if norm > 0:
        query_emb = query_emb / norm

    candidate_indices = np.array(
        [
            idx
            for idx, item_id in enumerate(catalog.item_ids)
            if matches_filters_fp(item_id, catalog, selected_filters)
        ]
    )

    if len(candidate_indices) == 0:
        return []

    scores = catalog.embeddings[candidate_indices] @ query_emb
    return _rank_results(scores, catalog, candidate_indices, top_k)
