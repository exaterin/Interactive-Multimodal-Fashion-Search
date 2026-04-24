from __future__ import annotations

from enum import Enum
from typing import List, Optional

from src.data.fashionpedia.catalog import FashionpediaCatalog
from src.search.relevance_feedback import FeedbackItem, build_feedback_context

from .attribute import AttributeGrounding
from .context import GroundingContext, ItemContext, REFINEMENT_SUPERCATS
from .description import DescriptionGrounding
from .image import ImageGrounding


_CONTEXT_SIZE = 50


class GroundingStrategy(str, Enum):
    ATTRIBUTE = "attribute"
    DESCRIPTION = "description"
    IMAGE = "image"


_FORMATTERS = {
    GroundingStrategy.ATTRIBUTE: AttributeGrounding(),
    GroundingStrategy.DESCRIPTION: DescriptionGrounding(),
    GroundingStrategy.IMAGE: ImageGrounding(),
}


def build_grounding_context(
    results: List[dict],
    catalog: FashionpediaCatalog,
    feedback_items: Optional[List[FeedbackItem]] = None,
    strategy: GroundingStrategy = GroundingStrategy.ATTRIBUTE,
    context_size: int = _CONTEXT_SIZE,
) -> GroundingContext:
    """
    Build grounding context from retrieved results and optional relevance feedback.

    When feedback_items is None or empty, context is built from catalog results only.
    """
    formatter = _FORMATTERS[strategy]
    feedback_str = build_feedback_context(feedback_items or [])
    load_images = strategy == GroundingStrategy.IMAGE

    if not results:
        return GroundingContext(
            total_results=0,
            items=[],
            feedback_context=feedback_str,
            formatter=formatter,
        )

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

    return GroundingContext(
        total_results=len(results),
        items=items,
        feedback_context=feedback_str,
        formatter=formatter,
    )
