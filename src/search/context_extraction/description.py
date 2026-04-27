from __future__ import annotations

from typing import List

from .item import ItemContext, REFINEMENT_SUPERCATS


class DescriptionFormatter:
    """Renders each item as a natural-language description."""

    is_multimodal = False

    def format_text(self, header: str, items: List[ItemContext]) -> str:
        lines = [header]
        for i, item in enumerate(items, 1):
            desc_parts = []
            if item.category:
                desc_parts.append(item.category)
            if item.colors:
                desc_parts.append(f"in {', '.join(item.colors)}")
            for supercat in REFINEMENT_SUPERCATS:
                attrs = item.attributes.get(supercat)
                if attrs:
                    desc_parts.append(", ".join(attrs))
            description = " ".join(desc_parts) if desc_parts else "(no description available)"
            lines.append(f"Item {i}: {description}")
        return "\n".join(lines)

    def build_blocks(self, header: str, items: List[ItemContext]) -> list:
        return [{"type": "text", "text": self.format_text(header, items)}]
