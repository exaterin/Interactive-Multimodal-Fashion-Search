"""Backward-compatible shim — import from src.search.grounding directly for new code."""
from src.search.grounding import (  # noqa: F401
    AttributeGrounding,
    DescriptionGrounding,
    GroundingContext,
    GroundingStrategy,
    ImageGrounding,
    ItemContext,
    build_grounding_context,
)
from src.search.grounding.builder import _CONTEXT_SIZE
from src.data.fashionpedia.catalog import FashionpediaCatalog
from typing import List


def analyze_results(
    results: List[dict],
    catalog: FashionpediaCatalog,
    context_size: int = _CONTEXT_SIZE,
) -> GroundingContext:
    return build_grounding_context(results, catalog, context_size=context_size)
