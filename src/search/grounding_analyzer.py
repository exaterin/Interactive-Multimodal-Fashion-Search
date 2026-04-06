from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.data.fashionpedia.catalog import FashionpediaCatalog


# Supercategories that are most useful for refinement suggestions
_REFINEMENT_SUPERCATS = (
    "silhouette",
    "length",
    "neckline type",
    "textile pattern",
    "non-textile material type",
    "textile finishing, manufacturing techniques",
)


@dataclass
class GroundingContext:
    total_results: int
    dominant_categories: List[Tuple[str, int]]          # [(name, count), ...]
    dominant_colors: List[Tuple[str, int]]               # [(color, count), ...]
    dominant_attributes: Dict[str, List[Tuple[str, int]]]  # {supercat: [(attr, count), ...]}
    is_diverse: bool                                     # multiple categories present

    def to_prompt_str(self, top_n: int = 5) -> str:
        """Render context as a concise text block for the LLM prompt."""
        lines = [f"Total retrieved items: {self.total_results}"]

        if self.dominant_categories:
            cats = ", ".join(
                f"{c} ({n})" for c, n in self.dominant_categories[:top_n]
            )
            lines.append(f"Categories found: {cats}")

        if self.dominant_colors:
            colors = ", ".join(
                f"{c} ({n})" for c, n in self.dominant_colors[:top_n]
            )
            lines.append(f"Colors found: {colors}")

        for supercat in _REFINEMENT_SUPERCATS:
            attrs = self.dominant_attributes.get(supercat, [])
            if attrs:
                attr_str = ", ".join(f"{a} ({n})" for a, n in attrs[:4])
                lines.append(f"{supercat.capitalize()}: {attr_str}")

        return "\n".join(lines)

    def top_refinement_values(self, top_n: int = 3) -> List[str]:
        """Return a flat list of the most frequent attribute values for quick suggestions."""
        values: List[str] = []
        for supercat in _REFINEMENT_SUPERCATS:
            for attr, _ in self.dominant_attributes.get(supercat, [])[:top_n]:
                values.append(attr)
        return values


def analyze_results(
    results: List[dict],
    catalog: FashionpediaCatalog,
    top_n: int = 6,
) -> GroundingContext:
    if not results:
        return GroundingContext(
            total_results=0,
            dominant_categories=[],
            dominant_colors=[],
            dominant_attributes={},
            is_diverse=False,
        )

    category_counter: Counter = Counter()
    color_counter: Counter = Counter()
    supercat_counters: Dict[str, Counter] = {}

    for item in results:
        item_id = item["image_id"]

        cat = catalog.category_annotations.get(item_id, "")
        if cat:
            category_counter[cat] += 1

        for color in catalog.color_annotations.get(item_id, []):
            color_counter[color] += 1

        for supercat, attrs in catalog.attribute_annotations.get(item_id, {}).items():
            if supercat not in supercat_counters:
                supercat_counters[supercat] = Counter()
            for attr in attrs:
                supercat_counters[supercat][attr] += 1

    dominant_attributes = {
        sc: counter.most_common(top_n)
        for sc, counter in supercat_counters.items()
        if counter
    }

    return GroundingContext(
        total_results=len(results),
        dominant_categories=category_counter.most_common(top_n),
        dominant_colors=color_counter.most_common(top_n),
        dominant_attributes=dominant_attributes,
        is_diverse=len(category_counter) > 2,
    )
