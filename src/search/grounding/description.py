from __future__ import annotations

from typing import TYPE_CHECKING

from .context import REFINEMENT_SUPERCATS

if TYPE_CHECKING:
    from .context import GroundingContext


class DescriptionGrounding:
    """Represents each retrieved item as a natural language description."""

    is_multimodal = False

    def format_catalog(self, context: GroundingContext) -> str:
        lines = [
            f"Total retrieved items: {context.total_results}",
            f"Showing top {len(context.items)} items (each as natural language description):\n",
        ]
        for i, item in enumerate(context.items, 1):
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

    def build_multimodal_blocks(self, context: GroundingContext) -> list:
        return [{"type": "text", "text": self.format_catalog(context)}]
