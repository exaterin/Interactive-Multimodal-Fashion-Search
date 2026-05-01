from __future__ import annotations

from typing import Dict, List

from src.datasets.deepfashion.catalog import Catalog


def matches_filters(
    image_id: str,
    catalog: Catalog,
    selected_filters: Dict[str, List[str]],
) -> bool:
    if selected_filters.get("fabric"):
        item_values = set(catalog.fabric_annotations.get(image_id, {}).values())
        if not set(selected_filters["fabric"]).issubset(item_values):
            return False

    if selected_filters.get("pattern"):
        item_values = set(catalog.pattern_annotations.get(image_id, {}).values())
        if not set(selected_filters["pattern"]).issubset(item_values):
            return False

    if selected_filters.get("shape"):
        item_values = set(catalog.shape_annotations.get(image_id, {}).values())
        if not set(selected_filters["shape"]).issubset(item_values):
            return False

    return True