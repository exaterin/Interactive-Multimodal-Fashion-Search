from __future__ import annotations

from typing import Dict, List

from src.data.catalog import Catalog
from src.retrieval.filtering import matches_filters


def search_by_attributes(
    catalog: Catalog,
    selected_filters: Dict[str, List[str]],
    top_k: int = 50,
) -> List[dict]:
    results = []

    for image_id in catalog.image_ids:
        if matches_filters(
            image_id=image_id,
            catalog=catalog,
            selected_filters=selected_filters,
        ):
            results.append(
                {
                    "image_id": image_id,
                    "score": None,
                }
            )

    results = results[:top_k]

    for rank, item in enumerate(results, start=1):
        item["rank"] = rank

    return results