"""
Catalog Evidence: top-k retrieved items + Context Extraction.

Mirrors the "Catalog Evidence" box in the project schema. Holds the items returned
by the multimodal retriever for the (rewritten) query and renders them via the
chosen ExtractionStrategy (Attributes / Descriptions / Images).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from src.data.fashionpedia.catalog import FashionpediaCatalog
from src.search.context_extraction import (
    ExtractionStrategy,
    ItemContext,
    REFINEMENT_SUPERCATS,
    get_formatter,
)


_CONTEXT_SIZE = 50


@dataclass
class CatalogEvidence:
    total_results: int
    items: List[ItemContext]
    formatter: Any = field(default=None, repr=False, compare=False)

    @property
    def is_multimodal(self) -> bool:
        return getattr(self.formatter, "is_multimodal", False)

    def _header(self) -> str:
        return (
            f"Catalog evidence — {self.total_results} retrieved, "
            f"showing top {len(self.items)}:"
        )

    def to_text(self) -> str:
        return self.formatter.format_text(self._header(), self.items)

    def to_blocks(self) -> list:
        return self.formatter.build_blocks(self._header(), self.items)


def build_catalog_evidence(
    results: List[dict],
    catalog: FashionpediaCatalog,
    strategy: ExtractionStrategy = ExtractionStrategy.ATTRIBUTE,
    context_size: int = _CONTEXT_SIZE,
) -> CatalogEvidence:
    formatter = get_formatter(strategy)
    load_images = strategy == ExtractionStrategy.IMAGE

    if not results:
        return CatalogEvidence(total_results=0, items=[], formatter=formatter)

    items: List[ItemContext] = []
    for result in results[:context_size]:
        item_id = result["image_id"]
        category = catalog.category_annotations.get(item_id, "")
        colors = list(catalog.color_annotations.get(item_id, []))
        raw_attrs = catalog.attribute_annotations.get(item_id, {})
        attributes = {
            supercat: sorted(raw_attrs[supercat])
            for supercat in REFINEMENT_SUPERCATS
            if supercat in raw_attrs
        }
        items.append(ItemContext(
            item_id=item_id,
            category=category,
            colors=colors,
            attributes=attributes,
            image_path=catalog.image_paths.get(item_id) if load_images else None,
            bbox=catalog.bboxes.get(item_id) if load_images else None,
        ))

    return CatalogEvidence(
        total_results=len(results),
        items=items,
        formatter=formatter,
    )
