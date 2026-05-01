"""
Preference Evidence: user-selected relevant items + Context Extraction.

Mirrors the "Preference Evidence" box in the project schema. Uses the same
ExtractionStrategy as Catalog Evidence so a liked item is rendered as a cropped
image when the system is in image grounding mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.datasets.fashionpedia.catalog import FashionpediaCatalog
from src.search.context_extraction import (
    ExtractionStrategy,
    ItemContext,
    REFINEMENT_SUPERCATS,
    get_formatter,
)


_PREFERENCE_FOOTER = (
    "Treat these as visual preference signals. Extract shared attributes across "
    "liked items and incorporate them into positive_constraints."
)


@dataclass
class PreferenceItem:
    """User-selected item as received from the UI (id + cached attributes)."""
    id: str
    category: str = ""
    attributes: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class PreferenceEvidence:
    items: List[ItemContext]
    formatter: Any = field(default=None, repr=False, compare=False)

    @property
    def is_empty(self) -> bool:
        return not self.items

    @property
    def is_multimodal(self) -> bool:
        return getattr(self.formatter, "is_multimodal", False)

    def _header(self) -> str:
        return (
            f"Preference evidence — user has liked {len(self.items)} item(s) "
            f"as style reference(s):"
        )

    def to_text(self) -> str:
        if self.is_empty:
            return ""
        return f"{self.formatter.format_text(self._header(), self.items)}\n{_PREFERENCE_FOOTER}"

    def to_blocks(self) -> list:
        if self.is_empty:
            return []
        blocks = self.formatter.build_blocks(self._header(), self.items)
        blocks.append({"type": "text", "text": _PREFERENCE_FOOTER})
        return blocks


def build_preference_evidence(
    items: List[PreferenceItem],
    catalog: FashionpediaCatalog,
    strategy: ExtractionStrategy = ExtractionStrategy.ATTRIBUTE,
) -> PreferenceEvidence:
    formatter = get_formatter(strategy)
    if not items:
        return PreferenceEvidence(items=[], formatter=formatter)

    load_images = strategy == ExtractionStrategy.IMAGE
    contexts: List[ItemContext] = []
    for item in items:
        attrs = item.attributes or {}
        colors = list(attrs.get("color", []))
        structured_attrs = {
            supercat: sorted(attrs[supercat])
            for supercat in REFINEMENT_SUPERCATS
            if supercat in attrs
        }
        contexts.append(ItemContext(
            item_id=item.id,
            category=item.category,
            colors=colors,
            attributes=structured_attrs,
            image_path=catalog.image_paths.get(item.id) if load_images else None,
            bbox=catalog.bboxes.get(item.id) if load_images else None,
        ))

    return PreferenceEvidence(items=contexts, formatter=formatter)
