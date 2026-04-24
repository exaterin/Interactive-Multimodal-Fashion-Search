from __future__ import annotations

from typing import TYPE_CHECKING

from .context import REFINEMENT_SUPERCATS

if TYPE_CHECKING:
    from .context import GroundingContext


class AttributeGrounding:
    """Represents each retrieved item as a structured set of fashion attributes."""

    is_multimodal = False

    def format_catalog(self, context: GroundingContext) -> str:
        lines = [
            f"Total retrieved items: {context.total_results}",
            f"Showing top {len(context.items)} items (each as attribute set):\n",
        ]
        for i, item in enumerate(context.items, 1):
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

    def build_multimodal_blocks(self, context: GroundingContext) -> list:
        return [{"type": "text", "text": self.format_catalog(context)}]
