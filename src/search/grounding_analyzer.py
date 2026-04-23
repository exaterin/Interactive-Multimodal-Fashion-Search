from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

from src.data.fashionpedia.catalog import FashionpediaCatalog


_REFINEMENT_SUPERCATS = (
    "silhouette",
    "length",
    "neckline type",
    "textile pattern",
    "non-textile material type",
    "textile finishing, manufacturing techniques",
)

_CONTEXT_SIZE = 50


@dataclass
class ItemContext:
    item_id: str
    category: str
    colors: List[str]
    attributes: Dict[str, List[str]]  # supercategory → sorted attribute values


@dataclass
class GroundingContext:
    total_results: int
    items: List[ItemContext]

    def to_prompt_str(self) -> str:
        lines = [
            f"Total retrieved items: {self.total_results}",
            f"Showing top {len(self.items)} items (each as attribute set):\n",
        ]
        for i, item in enumerate(self.items, 1):
            parts = [f"Item {i}"]
            if item.category:
                parts.append(f"category: {item.category}")
            if item.colors:
                parts.append(f"colors: {', '.join(item.colors)}")
            for supercat in _REFINEMENT_SUPERCATS:
                attrs = item.attributes.get(supercat)
                if attrs:
                    parts.append(f"{supercat}: {', '.join(attrs)}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    def top_refinement_values(self, top_n: int = 3) -> List[str]:
        counter: Counter = Counter()
        for item in self.items:
            for supercat in _REFINEMENT_SUPERCATS:
                for attr in item.attributes.get(supercat, []):
                    counter[attr] += 1
        return [attr for attr, _ in counter.most_common(top_n * len(_REFINEMENT_SUPERCATS))]


def analyze_results(
    results: List[dict],
    catalog: FashionpediaCatalog,
    context_size: int = _CONTEXT_SIZE,
) -> GroundingContext:
    if not results:
        return GroundingContext(total_results=0, items=[])

    items: List[ItemContext] = []
    for result in results[:context_size]:
        item_id = result["image_id"]
        category = catalog.category_annotations.get(item_id, "")
        colors = list(catalog.color_annotations.get(item_id, []))
        raw_attrs = catalog.attribute_annotations.get(item_id, {})
        attributes = {
            supercat: sorted(raw_attrs[supercat])
            for supercat in _REFINEMENT_SUPERCATS
            if supercat in raw_attrs
        }
        items.append(ItemContext(
            item_id=item_id,
            category=category,
            colors=colors,
            attributes=attributes,
        ))

    return GroundingContext(total_results=len(results), items=items)
