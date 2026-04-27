from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import List

from PIL import Image

from .item import ItemContext


class ImageFormatter:
    """Renders each item as a cropped image block for a multimodal LLM."""

    is_multimodal = True

    def format_text(self, header: str, items: List[ItemContext]) -> str:
        return f"{header} (images attached below)"

    def build_blocks(self, header: str, items: List[ItemContext]) -> list:
        blocks: list = [{"type": "text", "text": self.format_text(header, items)}]
        for i, item in enumerate(items, 1):
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
