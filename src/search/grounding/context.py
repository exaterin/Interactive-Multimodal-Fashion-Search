from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Shared supercategories used for attribute formatting and refinement stats
REFINEMENT_SUPERCATS = (
    "silhouette",
    "length",
    "neckline type",
    "textile pattern",
    "non-textile material type",
    "textile finishing, manufacturing techniques",
)


@dataclass
class ItemContext:
    item_id: str
    category: str
    colors: List[str]
    attributes: Dict[str, List[str]]  # supercategory → sorted attribute values
    image_path: Optional[Path] = None  # populated for ImageGrounding
    bbox: Optional[List[float]] = None  # populated for ImageGrounding


@dataclass
class GroundingContext:
    total_results: int
    items: List[ItemContext]
    feedback_context: str = ""
    # Formatter is one of AttributeGrounding / DescriptionGrounding / ImageGrounding.
    # typed as Any to avoid a circular import from the strategy modules.
    formatter: Any = field(default=None, repr=False, compare=False)

    def to_prompt_str(self) -> str:
        """Text context for LLM. Feedback is appended when present."""
        catalog_str = self.formatter.format_catalog(self)
        if self.feedback_context:
            return f"{catalog_str}\n\n{self.feedback_context}"
        return catalog_str

    @property
    def is_multimodal(self) -> bool:
        return getattr(self.formatter, "is_multimodal", False)

    def to_multimodal_blocks(self) -> list:
        """
        Multimodal content blocks for ImageGrounding.
        Feedback block is appended here so individual strategies don't need to handle it.
        """
        blocks = self.formatter.build_multimodal_blocks(self)
        if self.feedback_context:
            blocks.append({"type": "text", "text": self.feedback_context})
        return blocks

    def top_refinement_values(self, top_n: int = 3) -> List[str]:
        counter: Counter = Counter()
        for item in self.items:
            for supercat in REFINEMENT_SUPERCATS:
                for attr in item.attributes.get(supercat, []):
                    counter[attr] += 1
        return [attr for attr, _ in counter.most_common(top_n * len(REFINEMENT_SUPERCATS))]
