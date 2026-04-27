from __future__ import annotations

from typing import List

from .item import ItemContext, REFINEMENT_SUPERCATS


class AttributeFormatter:
    """Renders each item as a structured set of fashion attributes."""

    is_multimodal = False

    def format_text(self, header: str, items: List[ItemContext]) -> str:
        lines = [header]
        for i, item in enumerate(items, 1):
            parts = [f"Item {i}"]
            if item.category:
                parts.append(f"category: {item.category}")
            if item.colors:
                parts.append(f"colors: {', '.join(item.colors)}")
            for supercat in REFINEMENT_SUPERCATS:
                attrs = item.attributes.get(supercat)
                if attrs:
                    parts.append(f"{supercat}: {', '.join(attrs)}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    def build_blocks(self, header: str, items: List[ItemContext]) -> list:
        return [{"type": "text", "text": self.format_text(header, items)}]
