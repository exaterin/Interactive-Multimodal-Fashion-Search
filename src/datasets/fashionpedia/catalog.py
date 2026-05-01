from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import numpy as np


@dataclass
class FashionpediaCatalog:
    """
    Annotation-level catalog: each item is a single garment bounding-box
    crop, not a whole image.  item_ids are Fashionpedia annotation IDs
    (stored as strings).
    """

    # One entry per annotation (bbox crop).
    item_ids: List[str]
    embeddings: np.ndarray           # shape (N, D), L2-normalised

    # item_id → full image path (used to load + crop for display)
    image_paths: Dict[str, Path]

    # item_id → COCO bbox [x, y, width, height] in pixels
    bboxes: Dict[str, List[float]]

    # item_id → garment category name  ("shirt, blouse", "pants", …)
    category_annotations: Dict[str, str]

    # item_id → supercategory → set of attribute names
    # supercategories kept: "textile pattern", "animal", "silhouette",
    # "neckline type", "length", "waistline", "opening type",
    # "non-textile material type", "leather",
    # "textile finishing, manufacturing techniques"
    attribute_annotations: Dict[str, Dict[str, Set[str]]]

    # item_id → list of color labels (ordered by cosine similarity, threshold-filtered)
    color_annotations: Dict[str, List[str]]

    # filter dropdown options: keys are "category", "pattern", "shape", "material", "finishing", "color"
    available_values: Dict[str, List[str]]
