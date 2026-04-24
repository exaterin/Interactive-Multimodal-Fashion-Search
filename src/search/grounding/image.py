from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING, List

from PIL import Image

if TYPE_CHECKING:
    from .context import GroundingContext


class ImageGrounding:
    """Represents each retrieved item as a cropped image sent to a multimodal LLM."""

    is_multimodal = True

    def format_catalog(self, context: GroundingContext) -> str:
        """Text-only fallback when a multimodal LLM is not available."""
        return (
            f"Total retrieved items: {context.total_results}\n"
            f"Showing top {len(context.items)} items as images (attached below):"
        )

    def build_multimodal_blocks(self, context: GroundingContext) -> list:
        blocks: list = [{"type": "text", "text": self.format_catalog(context)}]
        for i, item in enumerate(context.items, 1):
            if item.image_path and item.image_path.exists() and item.bbox:
                img = _crop_bbox(item.image_path, item.bbox)
                blocks.append({"type": "text", "text": f"Item {i}:"})
                blocks.append({"type": "image_url", "image_url": {"url": _to_data_uri(img)}})
        return blocks


def _crop_bbox(image_path: Path, bbox: List[float], padding: float = 0.15) -> Image.Image:
    img = Image.open(str(image_path)).convert("RGB")
    x, y, w, h = (int(v) for v in bbox)
    pad_x, pad_y = int(w * padding), int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img.width, x + w + pad_x)
    y2 = min(img.height, y + h + pad_y)
    return img.crop((x1, y1, x2, y2))


def _to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"
