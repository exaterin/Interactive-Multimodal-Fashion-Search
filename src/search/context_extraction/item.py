from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


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
    """Structured item view used for both catalog and preference evidence."""
    item_id: str
    category: str
    colors: List[str]
    attributes: Dict[str, List[str]]  # supercategory → sorted values
    image_path: Optional[Path] = None
    bbox: Optional[List[float]] = None
