from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from fashion_clip.fashion_clip import FashionCLIP
from PIL import Image
from tqdm import tqdm

# Paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FASHIONPEDIA_DIR = PROJECT_ROOT / "data" / "fashionpedia"
ANNOTATIONS_JSON = FASHIONPEDIA_DIR / "instances_attributes_train2020.json"
IMAGES_DIR = FASHIONPEDIA_DIR / "train"
EMBEDDINGS_DIR = FASHIONPEDIA_DIR / "embeddings"


BATCH_SIZE = 128       # number of crops per FashionCLIP forward pass
MIN_SIDE = 16          # skip crops whose shorter side is below this (pixels)


# Helpers

def _crop(img: Image.Image, bbox: List[float]) -> Image.Image:
    """Crop a PIL image to a COCO [x, y, w, h] bounding box."""
    x, y, w, h = bbox
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(img.width, int(x + w)), min(img.height, int(y + h))
    return img.crop((x1, y1, x2, y2))


def _is_valid_crop(img: Image.Image) -> bool:
    return img.width >= MIN_SIDE and img.height >= MIN_SIDE


def _embed_batch(
    model: FashionCLIP,
    crops: List[Image.Image],
) -> np.ndarray:
    return np.array(model.encode_images(crops, batch_size=len(crops)), dtype=np.float32)


# Main

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load annotation JSON
    print(f"Loading annotations from {ANNOTATIONS_JSON} …")
    with ANNOTATIONS_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    # Group annotations by image so each image is loaded only once
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    print(
        f"Found {len(data['annotations'])} annotations across "
        f"{len(anns_by_image)} images."
    )

    model = FashionCLIP("fashion-clip")
    model.model = model.model.to(device)

    # Accumulate crops across images and flush in BATCH_SIZE chunks
    pending_crops: List[Image.Image] = []
    pending_ids: List[str] = []

    all_embeddings: List[np.ndarray] = []
    all_item_ids: List[str] = []
    skipped = 0

    def flush_batch() -> None:
        if not pending_crops:
            return
        embs = _embed_batch(model, pending_crops)
        all_embeddings.append(embs)
        all_item_ids.extend(pending_ids)
        pending_crops.clear()
        pending_ids.clear()

    for image_id, anns in tqdm(anns_by_image.items(), desc="Images"):
        fname = id_to_filename[image_id]
        img_path = IMAGES_DIR / fname

        if not img_path.exists():
            skipped += len(anns)
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += len(anns)
            continue

        for ann in anns:
            crop = _crop(img, ann["bbox"])
            if not _is_valid_crop(crop):
                skipped += 1
                continue

            pending_crops.append(crop)
            pending_ids.append(str(ann["id"]))

            if len(pending_crops) >= BATCH_SIZE:
                flush_batch()

    flush_batch()

    if not all_embeddings:
        raise RuntimeError("No valid crops were embedded. Check IMAGES_DIR path.")

    embeddings = np.vstack(all_embeddings)
    item_ids = np.array(all_item_ids)

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_DIR / "fashionpedia_embeddings.npy", embeddings)
    np.save(EMBEDDINGS_DIR / "fashionpedia_item_ids.npy", item_ids)

    print(f"\nEmbedded {len(item_ids)} crops  (skipped {skipped}).")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved to: {EMBEDDINGS_DIR}")


if __name__ == "__main__":
    main()
