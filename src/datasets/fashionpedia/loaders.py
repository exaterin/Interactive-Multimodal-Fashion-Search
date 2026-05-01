from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import streamlit as st

from src.datasets.fashionpedia.annotation_parser import (
    build_filter_values,
    load_color_annotations,
    parse_fashionpedia_annotations,
)
from src.datasets.fashionpedia.catalog import FashionpediaCatalog


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FASHIONPEDIA_DIR = PROJECT_ROOT / "data" / "fashionpedia"

ANNOTATIONS_JSON = FASHIONPEDIA_DIR / "instances_attributes_train2020.json"
IMAGES_DIR = FASHIONPEDIA_DIR / "train"
EMBEDDINGS_DIR = FASHIONPEDIA_DIR / "embeddings"

EMBEDDINGS_PATH  = EMBEDDINGS_DIR / "fashionpedia_embeddings.npy"
ITEM_IDS_PATH    = EMBEDDINGS_DIR / "fashionpedia_item_ids.npy"
COLOR_ANN_PATH   = FASHIONPEDIA_DIR / "color_ann.json"


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def load_embeddings(
    embeddings_path: Path,
    item_ids_path: Path,
) -> tuple[List[str], np.ndarray]:
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Fashionpedia embeddings not found at: {embeddings_path}\n"
            "Generate them first:\n"
            "  python src/models/generate_fashionpedia_embeddings.py"
        )
    if not item_ids_path.exists():
        raise FileNotFoundError(
            f"Fashionpedia item IDs not found at: {item_ids_path}\n"
            "Generate them first:\n"
            "  python src/models/generate_fashionpedia_embeddings.py"
        )

    embeddings = np.load(embeddings_path).astype(np.float32)
    embeddings = _normalize_rows(embeddings)

    item_ids = [str(x) for x in np.load(item_ids_path, allow_pickle=True).tolist()]

    if len(item_ids) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(item_ids)} item IDs but {len(embeddings)} embeddings."
        )

    return item_ids, embeddings


@st.cache_resource(show_spinner=False)
def load_fashionpedia_catalog() -> FashionpediaCatalog:
    item_ids, embeddings = load_embeddings(EMBEDDINGS_PATH, ITEM_IDS_PATH)

    filename_map, bbox_map, category_map, attribute_map = (
        parse_fashionpedia_annotations(ANNOTATIONS_JSON)
    )

    # Keep only annotations that were successfully embedded
    embedded_set = set(item_ids)
    filename_map  = {k: v for k, v in filename_map.items()  if k in embedded_set}
    bbox_map      = {k: v for k, v in bbox_map.items()      if k in embedded_set}
    category_map  = {k: v for k, v in category_map.items()  if k in embedded_set}
    attribute_map = {k: v for k, v in attribute_map.items() if k in embedded_set}

    color_annotations = {
        k: v
        for k, v in load_color_annotations(COLOR_ANN_PATH).items()
        if k in embedded_set
    }

    available_values = build_filter_values(category_map, attribute_map, color_annotations)

    image_paths: Dict[str, Path] = {
        item_id: IMAGES_DIR / fname
        for item_id, fname in filename_map.items()
    }

    return FashionpediaCatalog(
        item_ids=item_ids,
        embeddings=embeddings,
        image_paths=image_paths,
        bboxes=bbox_map,
        category_annotations=category_map,
        attribute_annotations=attribute_map,
        color_annotations=color_annotations,
        available_values=available_values,
    )
