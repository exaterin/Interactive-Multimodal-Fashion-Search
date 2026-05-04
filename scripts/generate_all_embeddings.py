#!/usr/bin/env python3
"""
Generate Fashionpedia image embeddings for five models:
FashionCLIP, CLIP (ViT-B/32), EVA02-B-16, SigLIP, and Marqo Fashion SigLIP.

For each crop in the Fashionpedia training set, the script crops the bounding
box, encodes it, L2-normalises the resulting vector, and writes
per-model `.npy` files:

    data/fashionpedia/embeddings/embeddings_fashionclip/
    data/fashionpedia/embeddings/embeddings_clip/
    data/fashionpedia/embeddings/embeddings_eva/
    data/fashionpedia/embeddings/embeddings_siglip/
    data/fashionpedia/embeddings/embeddings_marqo/

Each output directory contains:
    fashionpedia_embeddings.npy   (N, D) float32, L2-normalised
    fashionpedia_item_ids.npy     (N,)   annotation IDs as strings
    fashionpedia_filenames.npy    (N,)   source image filenames

Path constants and the row-normalisation helper are imported from
`src.datasets.fashionpedia.loaders` so this script stays in sync with how the
backend loads embeddings.

Usage:
    python scripts/generate_all_embeddings.py                       # all five
    python scripts/generate_all_embeddings.py --models fashionclip
    python scripts/generate_all_embeddings.py --models siglip marqo
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.fashionpedia.loaders import (  # noqa: E402
    ANNOTATIONS_JSON,
    EMBEDDINGS_DIR,
    IMAGES_DIR,
)

MIN_SIDE = 16   # skip crops whose shorter side is below this (pixels)

BATCH_SIZES = {
    "fashionclip": 128,
    "clip":        128,   # ViT-B/32 — smallest/fastest, true baseline
    "eva":          64,   # EVA02-B-16 — stronger, slightly heavier
    "siglip":       32,
    "marqo":        32,
}

OUTPUT_DIRS = {
    "fashionclip": EMBEDDINGS_DIR / "embeddings_fashionclip",
    "clip":        EMBEDDINGS_DIR / "embeddings_clip",
    "eva":         EMBEDDINGS_DIR / "embeddings_eva",
    "siglip":      EMBEDDINGS_DIR / "embeddings_siglip",
    "marqo":       EMBEDDINGS_DIR / "embeddings_marqo",
}


# ---------------------------------------------------------------------------
# Crop helpers
# ---------------------------------------------------------------------------

def _crop(img: Image.Image, bbox: List[float]) -> Image.Image:
    """Crop a PIL image to a COCO [x, y, w, h] bounding box."""
    x, y, w, h = bbox
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(img.width, int(x + w)), min(img.height, int(y + h))
    return img.crop((x1, y1, x2, y2))


def _is_valid(crop: Image.Image) -> bool:
    return crop.width >= MIN_SIDE and crop.height >= MIN_SIDE


# ---------------------------------------------------------------------------
# Encoder loaders
# ---------------------------------------------------------------------------

def _load_encoder(model_key: str):
    print(f"\nLoading {model_key} encoder …")
    if model_key == "fashionclip":
        from src.models.fashion_clip_encoder import FashionCLIPEncoder
        return FashionCLIPEncoder()
    elif model_key == "clip":
        from src.models.clip_encoder import CLIPEncoder
        return CLIPEncoder(model_name="ViT-B-32", pretrained="openai")
    elif model_key == "eva":
        from src.models.clip_encoder import CLIPEncoder
        return CLIPEncoder(model_name="EVA02-B-16", pretrained="merged2b_s8b_b131k")
    elif model_key == "siglip":
        from src.models.siglip_encoder import SigLIPEncoder
        return SigLIPEncoder()
    elif model_key == "marqo":
        from src.models.marqo_fashion_siglip_encoder import MarqoFashionSigLIPEncoder
        return MarqoFashionSigLIPEncoder()
    else:
        raise ValueError(f"Unknown model key: {model_key!r}")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def generate_embeddings(
    model_key: str,
    anns_by_image: dict,
    id_to_filename: dict,
) -> None:
    encoder  = _load_encoder(model_key)
    batch_sz = BATCH_SIZES[model_key]
    out_dir  = OUTPUT_DIRS[model_key]

    pending_crops:     List[Image.Image] = []
    pending_ids:       List[str]         = []
    pending_filenames: List[str]         = []
    all_embeddings:    List[np.ndarray]  = []
    all_item_ids:      List[str]         = []
    all_filenames:     List[str]         = []
    skipped = 0

    def flush() -> None:
        if not pending_crops:
            return
        embs = encoder.encode_images(pending_crops)
        all_embeddings.append(embs)
        all_item_ids.extend(pending_ids)
        all_filenames.extend(pending_filenames)
        pending_crops.clear()
        pending_ids.clear()
        pending_filenames.clear()

    for image_id, anns in tqdm(anns_by_image.items(), desc=f"[{model_key}] images"):
        fname    = id_to_filename[image_id]
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
            if not _is_valid(crop):
                skipped += 1
                continue

            pending_crops.append(crop)
            pending_ids.append(str(ann["id"]))
            pending_filenames.append(fname)

            if len(pending_crops) >= batch_sz:
                flush()

    flush()

    if not all_embeddings:
        raise RuntimeError(
            f"[{model_key}] No embeddings generated. "
            f"Check that {IMAGES_DIR} contains images."
        )

    embeddings = np.vstack(all_embeddings)
    item_ids   = np.array(all_item_ids)
    filenames  = np.array(all_filenames)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "fashionpedia_embeddings.npy",    embeddings)
    np.save(out_dir / "fashionpedia_item_ids.npy",      item_ids)
    np.save(out_dir / "fashionpedia_filenames.npy",     filenames)

    print(f"[{model_key}] Embedded {len(item_ids)} crops  (skipped {skipped})")
    print(f"[{model_key}] Shape:   {embeddings.shape}")
    print(f"[{model_key}] Saved →  {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Fashionpedia crop embeddings for FashionCLIP, CLIP, "
                    "EVA, SigLIP, and Marqo Fashion SigLIP.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(OUTPUT_DIRS.keys()),
        default=list(OUTPUT_DIRS.keys()),
        metavar="MODEL",
        help="Models to run: fashionclip | clip | eva | siglip | marqo  (default: all five)",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow running without CUDA. Off by default because a full run on CPU "
             "takes many hours per model.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    elif args.allow_cpu:
        print("CUDA not available — running on CPU (this will be slow).")
    else:
        raise RuntimeError(
            "CUDA is not available. Re-run with --allow-cpu to run on CPU anyway."
        )

    print(f"Loading annotations from {ANNOTATIONS_JSON} …")
    with ANNOTATIONS_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_filename: dict[int, str] = {img["id"]: img["file_name"] for img in data["images"]}

    anns_by_image: dict[int, list] = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    print(
        f"{len(data['annotations'])} annotations across "
        f"{len(anns_by_image)} images.\n"
    )

    for model_key in args.models:
        generate_embeddings(model_key, anns_by_image, id_to_filename)

    print("\nAll done.")


if __name__ == "__main__":
    main()
