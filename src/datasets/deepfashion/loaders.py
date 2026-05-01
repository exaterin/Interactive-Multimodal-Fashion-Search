from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import streamlit as st

from src.datasets.deepfashion.annotation_parser import (
    build_filter_values,
    parse_fabric_annotations,
    parse_pattern_annotations,
    parse_shape_annotations,
)
from src.datasets.deepfashion.catalog import Catalog


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "deepfashion"

ANNOTATIONS_DIR = DATA_DIR / "annotations"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
IMAGES_DIR = DATA_DIR / "images"

FABRIC_PATH = ANNOTATIONS_DIR / "fabric_ann.txt"
PATTERN_PATH = ANNOTATIONS_DIR / "pattern_ann.txt"
SHAPE_PATH = ANNOTATIONS_DIR / "shape_anno_all.txt"

EMBEDDINGS_PATH = EMBEDDINGS_DIR / "fashion_embeddings.npy"
IMAGE_IDS_PATH = EMBEDDINGS_DIR / "image_ids.npy"


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def load_embeddings(
    embeddings_path: Path,
    image_ids_path: Path,
) -> tuple[List[str], np.ndarray]:
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    if not image_ids_path.exists():
        raise FileNotFoundError(
            f"Image ids file not found: {image_ids_path}"
        )

    embeddings = np.load(embeddings_path).astype(np.float32)
    embeddings = _normalize_rows(embeddings)

    image_ids_array = np.load(image_ids_path, allow_pickle=True)
    image_ids = [str(x) for x in image_ids_array.tolist()]

    if len(image_ids) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(image_ids)} image ids but {len(embeddings)} embeddings."
        )

    return image_ids, embeddings


def build_image_paths(image_ids: List[str]) -> Dict[str, Path]:
    return {image_id: IMAGES_DIR / image_id for image_id in image_ids}


@st.cache_resource(show_spinner=False)
def load_catalog() -> Catalog:
    image_ids, embeddings = load_embeddings(EMBEDDINGS_PATH, IMAGE_IDS_PATH)

    fabric_annotations = parse_fabric_annotations(FABRIC_PATH)
    pattern_annotations = parse_pattern_annotations(PATTERN_PATH)
    shape_annotations = parse_shape_annotations(SHAPE_PATH)

    available_values = build_filter_values(
        fabric_annotations=fabric_annotations,
        pattern_annotations=pattern_annotations,
        shape_annotations=shape_annotations,
    )

    image_paths = build_image_paths(image_ids)

    return Catalog(
        image_ids=image_ids,
        embeddings=embeddings,
        image_paths=image_paths,
        fabric_annotations=fabric_annotations,
        pattern_annotations=pattern_annotations,
        shape_annotations=shape_annotations,
        available_values=available_values,
    )