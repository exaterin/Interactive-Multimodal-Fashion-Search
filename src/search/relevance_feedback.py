from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


_FEEDBACK_SUPERCATS = (
    "silhouette",
    "length",
    "neckline type",
    "textile pattern",
    "non-textile material type",
    "textile finishing, manufacturing techniques",
)


@dataclass
class FeedbackItem:
    id: str
    category: str = ""
    attributes: Dict[str, List[str]] = field(default_factory=dict)


def build_feedback_context(items: List[FeedbackItem]) -> str:
    """Format feedback items as per-item attribute rows for the LLM prompt."""
    if not items:
        return ""

    lines = [f"The user has liked {len(items)} item(s) as style reference(s):"]
    for i, item in enumerate(items, 1):
        parts = [f"Item {i}"]
        if item.category:
            parts.append(f"category: {item.category}")
        colors = item.attributes.get("color", [])
        if colors:
            parts.append(f"colors: {', '.join(colors)}")
        for supercat in _FEEDBACK_SUPERCATS:
            attrs = item.attributes.get(supercat, [])
            if attrs:
                parts.append(f"{supercat}: {', '.join(attrs)}")
        lines.append(" | ".join(parts))

    lines.append(
        "Treat these as visual preference signals. Extract shared attributes across "
        "liked items and incorporate them into updated_query and positive_constraints."
    )
    return "\n".join(lines)
